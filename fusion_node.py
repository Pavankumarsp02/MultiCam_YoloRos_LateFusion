#!/usr/bin/env python3
"""
fusion_node.py - consume-once fusion (no stale buffering)

Behavior:
 - For each input topic, a message is only valid for the very next publish cycle after it arrives.
 - After a publish cycle, pending messages are cleared (no reuse).
 - If NO detection messages arrived in a cycle, the node publishes an empty
   yolo_msgs/DetectionArray with header.frame_id='fused' and header.stamp = now(), and detections=[].
   (This is published every cycle until any detection arrives again.)
 - If NO image messages arrived in a cycle, the node does NOT publish a stitched image.
 - If at least one image arrived in the cycle, we stitch images side-by-side (1x3), filling missing slots
   with black tiles for that cycle only, and publish the panorama.
 - Per-detection source is preserved by writing parent header.frame_id into det.bbox3d.frame_id and det.keypoints3d.frame_id when present.
 - No stale-timeouts, no retention beyond one publish cycle.
"""

from __future__ import annotations
import argparse
from copy import deepcopy
import sys
import traceback
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image as RosImage
from yolo_msgs.msg import DetectionArray

# OpenCV + cv_bridge
try:
    import cv2
    from cv_bridge import CvBridge, CvBridgeError
except Exception:
    cv2 = None
    CvBridge = None
    CvBridgeError = Exception

DEFAULT_DET_INPUTS = ['/yolo_cam2/detections', '/yolo_cam3/detections', '/yolo_cam4/detections']
DEFAULT_IMG_INPUTS = ['/yolo_cam2/dbg_image', '/yolo_cam3/dbg_image', '/yolo_cam4/dbg_image']
DEFAULT_OUTPUT_DET = '/fused/detections'
DEFAULT_OUTPUT_IMG = '/fused/dbg_image'
DEFAULT_RATE = 10.0
NEUTRAL_FUSED_FRAME_ID = 'fused'

class FusionNode(Node):
    def __init__(self,
                 det_inputs: list[str],
                 img_inputs: list[str],
                 output_det: str,
                 output_img: str,
                 rate_hz: float):
        super().__init__('yolo_fusion_node_single')

        self.get_logger().info('Fusion node (consume-once) starting.')

        self.det_inputs = det_inputs
        self.img_inputs = img_inputs
        self.output_det = output_det
        self.output_img = output_img
        self.rate_hz = float(rate_hz)

        qos = QoSProfile(depth=10)

        # Pending messages: only valid until next publish cycle
        self.pending_dets: dict[str, DetectionArray | None] = {t: None for t in self.det_inputs}
        self.pending_imgs: dict[str, RosImage | None] = {t: None for t in self.img_inputs}

        # CV bridge
        self.bridge = CvBridge() if CvBridge is not None else None
        if self.bridge is None:
            self.get_logger().warn('cv_bridge not available: image stitching disabled until cv_bridge + OpenCV installed.')

        # Subscriptions (callbacks store pending msg; they do NOT persist across cycles)
        for t in self.det_inputs:
            try:
                self.create_subscription(
                    DetectionArray,
                    t,
                    lambda msg, topic=t: self._det_cb(msg, topic),
                    qos
                )
                self.get_logger().info(f'Subscribed detection: {t}')
            except Exception as e:
                self.get_logger().error(f'Failed to subscribe detection {t}: {e}')

        for t in self.img_inputs:
            try:
                self.create_subscription(
                    RosImage,
                    t,
                    lambda msg, topic=t: self._img_cb(msg, topic),
                    qos
                )
                self.get_logger().info(f'Subscribed image: {t}')
            except Exception as e:
                self.get_logger().error(f'Failed to subscribe image {t}: {e}')

        # Publishers
        self.det_pub = self.create_publisher(DetectionArray, self.output_det, qos)
        self.img_pub = self.create_publisher(RosImage, self.output_img, qos)

        # Timer
        self.timer = self.create_timer(1.0 / float(self.rate_hz), self._on_timer)
        self.get_logger().info(f'Publishing at {self.rate_hz} Hz. Detections-> {self.output_det}, Images-> {self.output_img}')

    # Callbacks: place message into pending (overwriting any earlier pending for that topic)
    def _det_cb(self, msg: DetectionArray, topic: str) -> None:
        # store a deepcopy to avoid mutability issues
        try:
            self.pending_dets[topic] = deepcopy(msg)
        except Exception:
            self.pending_dets[topic] = msg

    def _img_cb(self, msg: RosImage, topic: str) -> None:
        try:
            self.pending_imgs[topic] = deepcopy(msg)
        except Exception:
            self.pending_imgs[topic] = msg

    # Publish cycle: use only messages in pending_* dicts, then clear them
    def _on_timer(self) -> None:
        try:
            self._publish_detections_consume_once()
            self._publish_images_consume_once()
        except Exception as e:
            self.get_logger().error(f'Error in timer loop: {e}\n{traceback.format_exc()}')

    # Detections: use pending messages only; if none pending across all topics -> publish empty DetectionArray
    def _publish_detections_consume_once(self) -> None:
        # collect pending messages (these are messages received since last cycle)
        pending_list = [m for m in self.pending_dets.values() if m is not None]

        if not pending_list:
            # No inputs this cycle -> publish empty DetectionArray with fused header
            empty = DetectionArray()
            # header.stamp = now()
            now_msg = self.get_clock().now().to_msg()
            empty.header.stamp = deepcopy(now_msg)
            empty.header.frame_id = NEUTRAL_FUSED_FRAME_ID
            empty.detections = []
            try:
                self.det_pub.publish(empty)
                self.get_logger().debug('Published EMPTY fused DetectionArray (no pending inputs this cycle).')
            except Exception as e:
                self.get_logger().error(f'Failed to publish empty fused detections: {e}')
            # clear pending (already empty effectively)
            for k in self.pending_dets.keys():
                self.pending_dets[k] = None
            return

        # There are pending messages -> merge detections from pending messages only
        try:
            # choose newest pending message as template for other header fields
            newest = max(pending_list, key=lambda m: (m.header.stamp.sec, m.header.stamp.nanosec))
        except Exception:
            newest = pending_list[0]

        out = deepcopy(newest)
        out.detections = []

        # Append detections from pending topics only, and mark their source
        for topic in self.det_inputs:
            pm = self.pending_dets.get(topic)
            if pm is None:
                continue
            for det in pm.detections:
                det_copy = deepcopy(det)
                try:
                    if hasattr(det_copy, 'bbox3d'):
                        det_copy.bbox3d.frame_id = pm.header.frame_id
                except Exception:
                    pass
                try:
                    if hasattr(det_copy, 'keypoints3d'):
                        det_copy.keypoints3d.frame_id = pm.header.frame_id
                except Exception:
                    pass
                out.detections.append(det_copy)

        # header stamp = newest among pending
        newest_stamp = Time()
        newest_stamp.sec = 0
        newest_stamp.nanosec = 0
        for m in pending_list:
            try:
                s = m.header.stamp
                if (s.sec, s.nanosec) > (newest_stamp.sec, newest_stamp.nanosec):
                    newest_stamp = deepcopy(s)
            except Exception:
                pass
        out.header.stamp = newest_stamp
        out.header.frame_id = NEUTRAL_FUSED_FRAME_ID

        try:
            self.det_pub.publish(out)
            self.get_logger().debug(f'Published fused detections (from pending msgs): {len(out.detections)} detections.')
        except Exception as e:
            self.get_logger().error(f'Failed to publish fused detections: {e}')

        # Clear pending detections (consume-once)
        for k in self.pending_dets.keys():
            self.pending_dets[k] = None

    # Images: if at least one pending image exists this cycle -> stitch using only pending images; else skip publishing
    def _publish_images_consume_once(self) -> None:
        if self.bridge is None or cv2 is None:
            # image path disabled
            return

        pending_imgs_list = [self.pending_imgs[t] for t in self.img_inputs if self.pending_imgs.get(t) is not None]
        if not pending_imgs_list:
            # No images received this cycle -> do NOT publish (outputs stop immediately)
            # Clear pending image map (they are already None)
            for k in self.pending_imgs.keys():
                self.pending_imgs[k] = None
            self.get_logger().debug('No pending images this cycle -> skipping panorama publish.')
            return

        # Convert pending images (only those present this cycle) to cv2 and prepare slots
        imgs_cv = []
        for topic in self.img_inputs:
            pm = self.pending_imgs.get(topic)
            if pm is None:
                imgs_cv.append(None)
                continue
            try:
                cv_img = None
                try:
                    cv_img = self.bridge.imgmsg_to_cv2(pm, desired_encoding='bgr8')
                except CvBridgeError:
                    cv_img_raw = self.bridge.imgmsg_to_cv2(pm, desired_encoding='passthrough')
                    if cv_img_raw is None:
                        cv_img = None
                    else:
                        if len(cv_img_raw.shape) == 2:
                            cv_img = cv2.cvtColor(cv_img_raw, cv2.COLOR_GRAY2BGR)
                        elif cv_img_raw.shape[2] == 3:
                            cv_img = cv2.cvtColor(cv_img_raw, cv2.COLOR_RGB2BGR)
                        else:
                            cv_img = cv_img_raw
                imgs_cv.append(cv_img)
            except Exception as e:
                self.get_logger().warning(f'Failed to convert pending image from {topic}: {e}')
                imgs_cv.append(None)

        # Build panorama from imgs_cv: for missing slots (None) insert black tiles (only for this cycle)
        available_cv = [im for im in imgs_cv if im is not None]
        if not available_cv:
            # unexpected but handle: skip publish
            for k in self.pending_imgs.keys():
                self.pending_imgs[k] = None
            self.get_logger().debug('All pending images failed conversion -> skipping panorama publish.')
            return

        heights = [im.shape[0] for im in available_cv if im is not None and im.shape[0] > 0]
        target_h = min(heights) if heights else 240
        widths = [im.shape[1] for im in available_cv if im is not None and im.shape[1] > 0]
        median_w = int(sorted(widths)[len(widths)//2]) if widths else 320

        prepared = []
        for im in imgs_cv:
            if im is None:
                black = np.zeros((target_h, median_w, 3), dtype=np.uint8)
                prepared.append(black)
            else:
                h, w = im.shape[0], im.shape[1]
                if h != target_h:
                    scale = target_h / float(h)
                    new_w = max(1, int(round(w * scale)))
                    try:
                        imr = cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA)
                    except Exception:
                        imr = cv2.resize(im, (new_w, target_h))
                else:
                    imr = im
                prepared.append(imr)

        # normalize channels & dtype
        for i, p in enumerate(prepared):
            if len(p.shape) == 2:
                prepared[i] = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
            elif p.shape[2] == 4:
                prepared[i] = cv2.cvtColor(p, cv2.COLOR_BGRA2BGR)
            if prepared[i].dtype != np.uint8:
                prepared[i] = prepared[i].astype(np.uint8)

        # concat horizontally
        try:
            panorama = cv2.hconcat(prepared)
        except Exception:
            try:
                prepared2 = [ (p.astype('uint8') if p.dtype != np.uint8 else p) for p in prepared ]
                panorama = cv2.hconcat(prepared2)
            except Exception as e:
                self.get_logger().error(f'Failed to create panorama: {e}')
                # clear pending and return
                for k in self.pending_imgs.keys():
                    self.pending_imgs[k] = None
                return

        # convert to ROS Image
        try:
            ros_img = self.bridge.cv2_to_imgmsg(panorama, encoding='bgr8')
        except Exception:
            try:
                ros_img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB), encoding='rgb8')
            except Exception as e:
                self.get_logger().error(f'cv_bridge failed to convert panorama: {e}')
                for k in self.pending_imgs.keys():
                    self.pending_imgs[k] = None
                return

        # header stamp -> choose newest stamp among pending images and pending detections if any
        newest_stamp = Time()
        newest_stamp.sec = 0
        newest_stamp.nanosec = 0
        # check pending dets first
        for m in [m for m in self.pending_dets.values() if m is not None]:
            try:
                s = m.header.stamp
                if (s.sec, s.nanosec) > (newest_stamp.sec, newest_stamp.nanosec):
                    newest_stamp = deepcopy(s)
            except Exception:
                pass
        # then pending imgs
        for m in [m for m in self.pending_imgs.values() if m is not None]:
            try:
                s = m.header.stamp
                if (s.sec, s.nanosec) > (newest_stamp.sec, newest_stamp.nanosec):
                    newest_stamp = deepcopy(s)
            except Exception:
                pass
        # fallback to now
        if newest_stamp.sec == 0 and newest_stamp.nanosec == 0:
            newest_stamp = deepcopy(self.get_clock().now().to_msg())

        ros_img.header.stamp = newest_stamp
        ros_img.header.frame_id = NEUTRAL_FUSED_FRAME_ID

        # publish panorama
        try:
            self.img_pub.publish(ros_img)
            self.get_logger().debug('Published fused panorama (consume-once).')
        except Exception as e:
            self.get_logger().error(f'Failed to publish panorama: {e}')

        # Clear pending images (consume-once)
        for k in self.pending_imgs.keys():
            self.pending_imgs[k] = None

def parse_args(argv):
    p = argparse.ArgumentParser(description='Consume-once fusion node (detections + dbg image stitching)')
    p.add_argument('--det_inputs', nargs=3, default=DEFAULT_DET_INPUTS,
                   help='Three detection input topics (default cam2,cam3,cam4).')
    p.add_argument('--img_inputs', nargs=3, default=DEFAULT_IMG_INPUTS,
                   help='Three debug image input topics (default cam2,cam3,cam4).')
    p.add_argument('--output_det', default=DEFAULT_OUTPUT_DET, help='Output detection topic')
    p.add_argument('--output_img', default=DEFAULT_OUTPUT_IMG, help='Output image topic')
    p.add_argument('--rate', '-r', type=float, default=DEFAULT_RATE, help='Publish rate (Hz)')
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if len(args.det_inputs) != 3 or len(args.img_inputs) != 3:
        print('ERROR: require exactly 3 detection topics and 3 image topics', file=sys.stderr)
        return 2

    rclpy.init()
    node = None
    try:
        node = FusionNode(args.det_inputs, args.img_inputs, args.output_det, args.output_img, args.rate)
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








