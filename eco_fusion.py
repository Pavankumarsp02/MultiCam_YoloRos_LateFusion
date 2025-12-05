#!/usr/bin/env python3
"""
yolo_fusion_node.py (detections-only, optimized copies)

Behavior summary:
 - Detections (append-only): only accept detections that arrived *since the last publish*.
   If none arrived in the cycle, publish an empty yolo_msgs/DetectionArray with header.frame_id='fused'.
 - Preserves per-detection bbox3d.frame_id / keypoints3d.frame_id to record source camera.
 - Heavy use of deepcopy removed: uses references + shallow copies (copy.copy) and only copies nested objects when mutated.
"""

from __future__ import annotations
import argparse
import sys
import traceback
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import copy as pycopy

from builtin_interfaces.msg import Time
from yolo_msgs.msg import DetectionArray

DEFAULT_DET_INPUTS = [
    '/yolo/detections3',
    '/yolo/detections4',
    '/yolo/detections5',
]
DEFAULT_OUTPUT_DET = '/fused/detections'
DEFAULT_RATE = 10.0
NEUTRAL_FUSED_FRAME_ID = 'fused'  # fused header.frame_id

class YoloFusionNode(Node):
    def __init__(self,
                 det_input_topics: list[str],
                 output_det_topic: str,
                 publish_rate_hz: float):
        super().__init__('late_fusion_node')

        self.get_logger().info('YOLO fusion node starting (freshness-safe, detections-only, optimized).')

        self.det_input_topics = det_input_topics
        self.output_det_topic = output_det_topic
        self.publish_rate_hz = float(publish_rate_hz)

        qos = QoSProfile(depth=10)

        # Latest messages storage
        # For detections we track a receive counter per-topic so we can decide "fresh since last publish"
        # Store references to the incoming messages (no deepcopy)
        self.latest_det_msgs: dict[str, DetectionArray | None] = {t: None for t in self.det_input_topics}
        self.recv_seq: dict[str, int] = {t: 0 for t in self.det_input_topics}           # incremented on each msg recv
        self.last_published_seq: dict[str, int] = {t: 0 for t in self.det_input_topics}  # snapshot after publish

        # Subscriptions: detections
        for topic in self.det_input_topics:
            try:
                self.create_subscription(
                    DetectionArray,
                    topic,
                    lambda msg, topic=topic: self._det_callback(msg, topic),
                    qos
                )
                self.get_logger().info(f'Subscribed to detection topic: {topic}')
            except Exception as e:
                self.get_logger().error(f'Failed to subscribe to detection topic {topic}: {e}')

        # Publisher for fused detections
        self.det_pub = self.create_publisher(DetectionArray, self.output_det_topic, qos)

        # Timer
        self.timer = self.create_timer(1.0 / float(self.publish_rate_hz), self._on_timer)
        self.get_logger().info(f'Publishing fused detections to {self.output_det_topic} @ {self.publish_rate_hz} Hz')

    # Detections callback: store message reference and increment seq (no deepcopy)
    def _det_callback(self, msg: DetectionArray, topic: str) -> None:
        # store reference to latest message (avoid deepcopy)
        self.latest_det_msgs[topic] = msg
        # mark that this topic has a new message
        self.recv_seq[topic] += 1

    # Timer loop: produce fused detections
    def _on_timer(self) -> None:
        try:
            self._publish_fused_detections()
        except Exception as e:
            self.get_logger().error(f'Error in timer loop: {e}\n{traceback.format_exc()}')

    # -------------------------
    # Detections: freshness-enforced append-only
    # -------------------------
    def _publish_fused_detections(self) -> None:
        # Collect only messages that are NEW since last publish (recv_seq > last_published_seq)
        msgs_to_use: list[tuple[str, DetectionArray]] = []
        for topic in self.det_input_topics:
            if self.recv_seq.get(topic, 0) > self.last_published_seq.get(topic, 0):
                msg = self.latest_det_msgs.get(topic)
                if msg is not None:
                    msgs_to_use.append((topic, msg))

        # If no new messages across all detection topics -> publish empty fused message
        if not msgs_to_use:
            empty_msg = DetectionArray()
            # stamp with current ROS time (fresh), and neutral fused frame_id
            now = self.get_clock().now().to_msg()
            empty_msg.header.stamp = now  # reference is fine
            empty_msg.header.frame_id = NEUTRAL_FUSED_FRAME_ID
            empty_msg.detections = []
            try:
                self.det_pub.publish(empty_msg)
                self.get_logger().debug('Published EMPTY fused detections (no fresh inputs this cycle)')
            except Exception as e:
                self.get_logger().error(f'Failed to publish empty fused detections: {e}')
            # Important: advance last_published_seq to current recv_seq for all topics so old messages won't be reused
            for t in self.det_input_topics:
                self.last_published_seq[t] = self.recv_seq.get(t, 0)
            return

        # There are new messages; build fused message from only those new messages
        # Find newest stamp among msgs_to_use (we'll use reference to that stamp)
        newest_msg = None
        try:
            newest_msg = max((m for (_, m) in msgs_to_use), key=lambda m: (m.header.stamp.sec, m.header.stamp.nanosec))
        except Exception:
            newest_msg = msgs_to_use[0][1]

        out_msg = DetectionArray()
        out_msg.detections = []

        # Append shallow copies of detections; only shallow-copy nested objects we mutate
        for topic, parent_msg in msgs_to_use:
            for det in parent_msg.detections:
                # shallow copy the detection object (cheap)
                try:
                    det_copy = pycopy.copy(det)
                except Exception:
                    # fallback: use original reference (should rarely happen)
                    det_copy = det

                # If bbox3d present and we need to set frame_id, shallow-copy bbox3d before mutating
                try:
                    if hasattr(det_copy, 'bbox3d') and getattr(det_copy, 'bbox3d') is not None:
                        try:
                            bbox = det_copy.bbox3d
                            # shallow copy nested bbox object to avoid mutating publisher's copy
                            try:
                                bbox_copy = pycopy.copy(bbox)
                                bbox_copy.frame_id = parent_msg.header.frame_id
                                det_copy.bbox3d = bbox_copy
                            except Exception:
                                # if copying failed, attempt to set on a new blank object if available, else skip mutating
                                try:
                                    bbox.frame_id = parent_msg.header.frame_id
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

                # If keypoints3d present and we need to set frame_id, shallow-copy keypoints3d before mutating
                try:
                    if hasattr(det_copy, 'keypoints3d') and getattr(det_copy, 'keypoints3d') is not None:
                        try:
                            kp = det_copy.keypoints3d
                            try:
                                kp_copy = pycopy.copy(kp)
                                kp_copy.frame_id = parent_msg.header.frame_id
                                det_copy.keypoints3d = kp_copy
                            except Exception:
                                try:
                                    kp.frame_id = parent_msg.header.frame_id
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

                out_msg.detections.append(det_copy)

        # set header (stamp = newest among used msgs)
        try:
            out_msg.header.stamp = newest_msg.header.stamp  # use reference
        except Exception:
            out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = NEUTRAL_FUSED_FRAME_ID

        # Publish fused detections
        try:
            self.det_pub.publish(out_msg)
            self.get_logger().debug(f'Published fused detections ({len(out_msg.detections)} detections) from fresh inputs')
        except Exception as e:
            self.get_logger().error(f'Failed to publish fused detections: {e}')

        # After publishing, mark last_published_seq = current recv_seq for those topics consumed (and for all topics to prevent reuse)
        for t in self.det_input_topics:
            self.last_published_seq[t] = self.recv_seq.get(t, 0)

def parse_args(argv):
    parser = argparse.ArgumentParser(description='YOLO fusion node (detections-only, optimized)')
    parser.add_argument('--det_inputs', nargs=3, metavar=('D1','D2','D3'),
                        help='Three detection input topics (default cam3,cam4,cam5).',
                        default=DEFAULT_DET_INPUTS)
    parser.add_argument('--output_det', default=DEFAULT_OUTPUT_DET,
                        help='Fused detection output topic (default: /fused/detections)')
    parser.add_argument('--rate', '-r', type=float, default=DEFAULT_RATE,
                        help='Publish rate in Hz (default: 10)')
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    det_inputs = args.det_inputs
    if len(det_inputs) != 3:
        print('ERROR: provide exactly 3 detection topics', file=sys.stderr)
        return 2

    rclpy.init()
    node = None
    try:
        node = YoloFusionNode(det_inputs, args.output_det, args.rate)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Node crashed: {e}\n{traceback.format_exc()}', file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
