#!/usr/bin/env python3
import os
import time
import threading
from typing import Optional, Set

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# vision_msgs (Humble)
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


def _reliability_from_str(s: str) -> ReliabilityPolicy:
    s = (s or "").strip().lower()
    if s in ("reliable", "rel"):
        return ReliabilityPolicy.RELIABLE
    return ReliabilityPolicy.BEST_EFFORT


def _parse_class_ids(s: str) -> Optional[Set[int]]:
    """
    "ALL" -> None (tidak filter)
    "0,1,2" -> {0,1,2}
    """
    if s is None:
        return None
    s = s.strip().upper()
    if s == "" or s == "ALL":
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        out.add(int(part))
    return out


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("detector_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("sub_image", "/camera/image_raw_reliable")
        self.declare_parameter("pub_image", "/camera/image_annotated")
        self.declare_parameter("pub_det", "/camera/detections")

        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_detections", True)

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("device", "cpu")   # di Jetson biasanya "0" / "cuda:0"
        self.declare_parameter("imgsz", 416)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("class_ids", "ALL")

        self.declare_parameter("max_fps", 10.0)      # 0 = unlimit (tidak disarankan di WSL)
        self.declare_parameter("qos_depth", 1)

        self.declare_parameter("sub_reliability", "reliable")
        self.declare_parameter("pub_image_reliability", "reliable")
        self.declare_parameter("pub_det_reliability", "reliable")

        self.declare_parameter("draw_boxes", True)
        self.declare_parameter("draw_labels", True)
        self.declare_parameter("draw_proc_ms", True)
        self.declare_parameter("box_thickness", 2)
        self.declare_parameter("font_scale", 0.7)

        # ---------------- Read params ----------------
        self.sub_image = self.get_parameter("sub_image").value
        self.pub_image = self.get_parameter("pub_image").value
        self.pub_det = self.get_parameter("pub_det").value

        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.publish_detections = bool(self.get_parameter("publish_detections").value)

        self.model_path = str(self.get_parameter("model_path").value)
        self.device = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.class_ids_raw = str(self.get_parameter("class_ids").value)
        self.class_ids = _parse_class_ids(self.class_ids_raw)

        self.max_fps = float(self.get_parameter("max_fps").value)
        self.qos_depth = int(self.get_parameter("qos_depth").value)

        self.sub_rel = _reliability_from_str(self.get_parameter("sub_reliability").value)
        self.pub_img_rel = _reliability_from_str(self.get_parameter("pub_image_reliability").value)
        self.pub_det_rel = _reliability_from_str(self.get_parameter("pub_det_reliability").value)

        self.draw_boxes = bool(self.get_parameter("draw_boxes").value)
        self.draw_labels = bool(self.get_parameter("draw_labels").value)
        self.draw_proc_ms = bool(self.get_parameter("draw_proc_ms").value)
        self.box_thickness = int(self.get_parameter("box_thickness").value)
        self.font_scale = float(self.get_parameter("font_scale").value)

        # ---------------- QoS ----------------
        qos_sub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self.qos_depth),
            reliability=self.sub_rel
        )
        qos_pub_img = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self.qos_depth),
            reliability=self.pub_img_rel
        )
        qos_pub_det = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self.qos_depth),
            reliability=self.pub_det_rel
        )

        # ---------------- ROS IO ----------------
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.sub_image, self._on_image, qos_sub
        )
        self.pub_img = self.create_publisher(Image, self.pub_image, qos_pub_img)
        self.pub_det = self.create_publisher(Detection2DArray, self.pub_det, qos_pub_det)

        # ---------------- Model ----------------
        weights_path = self._resolve_model_path(self.model_path)
        self.get_logger().info(f"Loading YOLO model: {weights_path} (device={self.device}, imgsz={self.imgsz})")
        self.model = YOLO(weights_path)
        self.names = self.model.names if hasattr(self.model, "names") else {}

        # ---------------- Latest-frame processing (anti-delay) ----------------
        self._lock = threading.Lock()
        self._latest_msg: Optional[Image] = None

        # stats
        self._proc_ema_ms = 0.0
        self._ema_alpha = 0.15
        self._frames = 0
        self._last_stat_t = time.time()

        # timer loop
        if self.max_fps and self.max_fps > 0.0:
            period = 1.0 / self.max_fps
        else:
            period = 0.0  # fastest possible, still uses timer
        period = max(0.001, period)  # safety
        self.timer = self.create_timer(period, self._process_latest)

        self.get_logger().info(
            f"Started | sub={self.sub_image}({self.sub_rel.name}) | "
            f"pub_img={self.pub_image}({self.pub_img_rel.name}) | "
            f"pub_det={self.pub_det}({self.pub_det_rel.name}) | "
            f"class_ids={self.class_ids_raw} | max_fps={self.max_fps}"
        )

    def _resolve_model_path(self, p: str) -> str:
        # kalau user kasih path absolut / relatif yang valid
        if p and os.path.exists(p):
            return p

        # fallback: coba cari di folder paket (seano_vision/)
        # asumsi: weights ada di root package yang sama dengan setup.py atau di seano_vision/
        candidates = [
            os.path.join(os.getcwd(), p),
            os.path.join(os.path.dirname(__file__), p),
            os.path.join(os.path.dirname(__file__), "..", p),
        ]
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c

        # terakhir: balikin apa adanya (biar YOLO bisa download kalau itu nama model umum)
        return p

    def _on_image(self, msg: Image) -> None:
        # simpan frame terbaru aja (drop backlog)
        with self._lock:
            self._latest_msg = msg

    def _process_latest(self) -> None:
        with self._lock:
            msg = self._latest_msg
            self._latest_msg = None

        if msg is None:
            return

        t0 = time.time()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert failed: {e}")
            return

        # YOLO inference
        try:
            res = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False
            )[0]
        except Exception as e:
            self.get_logger().error(f"YOLO predict failed: {e}")
            return

        det_array = Detection2DArray()
        det_array.header = msg.header

        annotated = frame.copy()

        # parse boxes
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), c_id, score in zip(xyxy, cls, confs):
                if self.class_ids is not None and c_id not in self.class_ids:
                    continue

                x1 = float(np.clip(x1, 0, frame.shape[1] - 1))
                y1 = float(np.clip(y1, 0, frame.shape[0] - 1))
                x2 = float(np.clip(x2, 0, frame.shape[1] - 1))
                y2 = float(np.clip(y2, 0, frame.shape[0] - 1))

                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                # --- Detection2D ---
                d = Detection2D()
                d.header = msg.header
                # IMPORTANT: vision_msgs/Pose2D punya .position.x/.position.y (bukan x,y langsung)
                d.bbox.center.position.x = float(cx)
                d.bbox.center.position.y = float(cy)
                d.bbox.center.theta = 0.0
                d.bbox.size_x = float(w)
                d.bbox.size_y = float(h)

                hyp = ObjectHypothesisWithPose()
                # ObjectHypothesisWithPose.hypothesis.class_id dan score (format umum vision_msgs)
                hyp.hypothesis.class_id = str(c_id)
                hyp.hypothesis.score = float(score)
                d.results.append(hyp)

                det_array.detections.append(d)

                # --- Draw ---
                if self.draw_boxes:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(annotated, p1, p2, (0, 255, 0), self.box_thickness)

                if self.draw_labels:
                    name = self.names.get(c_id, str(c_id)) if isinstance(self.names, dict) else str(c_id)
                    label = f"{name} {score:.2f}"
                    cv2.putText(
                        annotated,
                        label,
                        (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        (0, 255, 0),
                        2
                    )

        proc_ms = (time.time() - t0) * 1000.0
        self._proc_ema_ms = proc_ms if self._frames == 0 else (
            self._ema_alpha * proc_ms + (1.0 - self._ema_alpha) * self._proc_ema_ms
        )
        self._frames += 1

        if self.draw_proc_ms:
            cv2.putText(
                annotated,
                f"proc={self._proc_ema_ms:.1f}ms",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # publish
        if self.publish_detections:
            self.pub_det.publish(det_array)

        if self.publish_annotated:
            try:
                out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out_msg.header = msg.header
                self.pub_img.publish(out_msg)
            except Exception as e:
                self.get_logger().error(f"Publish annotated failed: {e}")

        # stat log tiap ~1 detik
        now = time.time()
        if now - self._last_stat_t >= 1.0:
            self.get_logger().info(
                f"proc_ema={self._proc_ema_ms:.1f}ms | frames={self._frames}"
            )
            self._last_stat_t = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
