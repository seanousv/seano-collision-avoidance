#!/usr/bin/env python3
"""
SEANO - detector_node.py
ROS2 Humble | YOLOv8 detector

Subscribe:
  - sub_image (sensor_msgs/Image)

Publish:
  - pub_det   (vision_msgs/Detection2DArray)
  - pub_image (sensor_msgs/Image) [annotated]

Tujuan:
  - QoS stabil (reliable / best_effort configurable)
  - Anti-delay: proses hanya frame terbaru (drop backlog)
  - Throttling via timer (max_fps)
  - Parameter bisa diubah saat runtime untuk tuning (conf/iou/imgsz/classes/drawing/max_fps, dll)
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Optional, Set, Tuple, List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False


def _reliability_from_str(s: str) -> ReliabilityPolicy:
    s = (s or "").strip().lower()
    if s in ("reliable", "rel"):
        return ReliabilityPolicy.RELIABLE
    return ReliabilityPolicy.BEST_EFFORT


def _parse_class_ids(s: str) -> Optional[Set[int]]:
    """
    "ALL" / "" -> None (tidak filter)
    "0,1,2"    -> {0,1,2}
    """
    if s is None:
        return None
    s = str(s).strip().upper()
    if s == "" or s == "ALL":
        return None
    out: Set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            pass
    return out if out else None


def _safe_to_numpy(x) -> np.ndarray:
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("detector_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("sub_image", "/camera/image_raw_reliable")
        self.declare_parameter("pub_image", "/camera/image_annotated")
        self.declare_parameter("pub_det", "/camera/detections")

        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_detections", True)
        self.declare_parameter("publish_empty_detections", True)

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("device", "cpu")      # contoh: "cpu", "cuda:0", "0"
        self.declare_parameter("imgsz", 416)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("class_ids", "ALL")

        self.declare_parameter("max_det", 50)        # batasi output deteksi
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("half", False)        # fp16 (biasanya untuk CUDA)
        self.declare_parameter("warmup", True)

        self.declare_parameter("max_fps", 10.0)      # 0 = secepatnya (tidak disarankan di WSL)
        self.declare_parameter("qos_depth", 1)

        self.declare_parameter("sub_reliability", "reliable")
        self.declare_parameter("pub_image_reliability", "reliable")
        self.declare_parameter("pub_det_reliability", "reliable")

        self.declare_parameter("draw_boxes", True)
        self.declare_parameter("draw_labels", True)
        self.declare_parameter("draw_label_bg", True)
        self.declare_parameter("draw_proc_ms", True)

        self.declare_parameter("box_thickness", 2)
        self.declare_parameter("font_scale", 0.6)
        self.declare_parameter("font_thickness", 2)

        self.declare_parameter("annotated_overlay_color_bgr", [0, 255, 0])
        self.declare_parameter("proc_text_color_bgr", [255, 255, 255])
        self.declare_parameter("label_bg_alpha", 0.35)

        self.declare_parameter("stats_period", 1.0)  # seconds (0 = off)

        # ---------------- Read params ----------------
        self._read_static_params()
        self._read_runtime_params()

        # ---------------- QoS ----------------
        depth = max(1, int(self.qos_depth))
        qos_sub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=self.sub_rel,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_pub_img = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=self.pub_img_rel,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_pub_det = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=self.pub_det_rel,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------------- ROS IO ----------------
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.sub_image, self._on_image, qos_sub)
        self.pub_img = self.create_publisher(Image, self.pub_image, qos_pub_img)
        self.pub_det_pub = self.create_publisher(Detection2DArray, self.pub_det, qos_pub_det)

        # ---------------- Model ----------------
        if not _HAS_YOLO:
            self.get_logger().error("Ultralytics YOLO tidak tersedia. Pastikan 'ultralytics' sudah terpasang.")
            raise RuntimeError("ultralytics not available")

        weights_path = self._resolve_model_path(self.model_path)
        self.get_logger().info(
            f"Loading YOLO model: {weights_path} | device={self.device or 'auto'} | imgsz={self.imgsz}"
        )
        self.model = YOLO(weights_path)
        self.names = self.model.names if hasattr(self.model, "names") else {}

        if bool(self.warmup):
            self._warmup_model()

        # ---------------- Latest-frame processing (anti-delay) ----------------
        self._lock = threading.Lock()
        self._latest_msg: Optional[Image] = None

        # prevent re-entrancy (kalau executor multi-thread)
        self._infer_lock = threading.Lock()

        # stats
        self._proc_ema_ms = 0.0
        self._ema_alpha = 0.15
        self._frames = 0
        self._last_stat_t = time.time()
        self._last_det_count = 0

        # timer loop
        self._timer = None
        self._create_or_update_timer()

        # params callback (runtime tuning)
        self.add_on_set_parameters_callback(self._on_params)

        self.get_logger().info(
            f"Started | sub={self.sub_image}({self.sub_rel.name}) | "
            f"pub_img={self.pub_image}({self.pub_img_rel.name}) | "
            f"pub_det={self.pub_det}({self.pub_det_rel.name}) | "
            f"class_ids={self.class_ids_raw} | max_fps={self.max_fps}"
        )

    # ---------------- Parameter read ----------------
    def _read_static_params(self) -> None:
        self.sub_image = str(self.get_parameter("sub_image").value)
        self.pub_image = str(self.get_parameter("pub_image").value)
        self.pub_det = str(self.get_parameter("pub_det").value)

        self.model_path = str(self.get_parameter("model_path").value)
        self.device = str(self.get_parameter("device").value).strip()

        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.sub_rel = _reliability_from_str(str(self.get_parameter("sub_reliability").value))
        self.pub_img_rel = _reliability_from_str(str(self.get_parameter("pub_image_reliability").value))
        self.pub_det_rel = _reliability_from_str(str(self.get_parameter("pub_det_reliability").value))

    def _read_runtime_params(self) -> None:
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.publish_detections = bool(self.get_parameter("publish_detections").value)
        self.publish_empty_detections = bool(self.get_parameter("publish_empty_detections").value)

        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)

        self.class_ids_raw = str(self.get_parameter("class_ids").value)
        self.class_ids = _parse_class_ids(self.class_ids_raw)

        self.max_det = int(self.get_parameter("max_det").value)
        self.agnostic_nms = bool(self.get_parameter("agnostic_nms").value)
        self.half = bool(self.get_parameter("half").value)
        self.warmup = bool(self.get_parameter("warmup").value)

        self.max_fps = float(self.get_parameter("max_fps").value)

        self.draw_boxes = bool(self.get_parameter("draw_boxes").value)
        self.draw_labels = bool(self.get_parameter("draw_labels").value)
        self.draw_label_bg = bool(self.get_parameter("draw_label_bg").value)
        self.draw_proc_ms = bool(self.get_parameter("draw_proc_ms").value)

        self.box_thickness = int(self.get_parameter("box_thickness").value)
        self.font_scale = float(self.get_parameter("font_scale").value)
        self.font_thickness = int(self.get_parameter("font_thickness").value)

        self.stats_period = float(self.get_parameter("stats_period").value)

        self.overlay_color = self._parse_color(self.get_parameter("annotated_overlay_color_bgr").value, (0, 255, 0))
        self.proc_text_color = self._parse_color(self.get_parameter("proc_text_color_bgr").value, (255, 255, 255))
        self.label_bg_alpha = float(self.get_parameter("label_bg_alpha").value)
        self.label_bg_alpha = max(0.0, min(0.9, self.label_bg_alpha))

        # clamp aman
        self.imgsz = max(160, int(self.imgsz))
        self.conf = max(0.0, min(1.0, float(self.conf)))
        self.iou = max(0.0, min(1.0, float(self.iou)))
        self.max_det = max(1, min(300, int(self.max_det)))
        self.box_thickness = max(1, min(8, int(self.box_thickness)))
        self.font_thickness = max(1, min(6, int(self.font_thickness)))
        self.font_scale = max(0.3, min(1.5, float(self.font_scale)))
        self.max_fps = float(self.max_fps)

    def _on_params(self, params: List[Parameter]) -> SetParametersResult:
        """
        Runtime tuning yang aman tanpa restart:
          - conf, iou, imgsz
          - class_ids
          - max_fps (timer akan diupdate)
          - draw flags, warna overlay, thickness, dsb.
        Perubahan QoS/topic/model_path/device sebaiknya restart node.
        """
        names = {p.name for p in params}
        timer_related = ("max_fps" in names)

        try:
            self._read_runtime_params()
            if timer_related:
                self._create_or_update_timer()
        except Exception as e:
            return SetParametersResult(successful=False, reason=str(e))

        return SetParametersResult(successful=True, reason="ok")

    # ---------------- Timer ----------------
    def _create_or_update_timer(self) -> None:
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass
            self._timer = None

        if self.max_fps and self.max_fps > 0.0:
            period = 1.0 / float(self.max_fps)
        else:
            period = 0.001  # tetap jalan, tapi jangan 0 (biar aman)
        period = max(0.001, float(period))

        self._timer = self.create_timer(period, self._process_latest)

    # ---------------- Utilities ----------------
    @staticmethod
    def _parse_color(v, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        try:
            if isinstance(v, (list, tuple)) and len(v) == 3:
                b = int(v[0]); g = int(v[1]); r = int(v[2])
                b = max(0, min(255, b))
                g = max(0, min(255, g))
                r = max(0, min(255, r))
                return (b, g, r)
        except Exception:
            pass
        return fallback

    def _resolve_model_path(self, p: str) -> str:
        if p and os.path.exists(p):
            return p

        candidates = [
            os.path.join(os.getcwd(), p),
            os.path.join(os.path.dirname(__file__), p),
            os.path.join(os.path.dirname(__file__), "..", p),
        ]
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c

        try:
            from ament_index_python.packages import get_package_share_directory  # type: ignore
            share_dir = Path(get_package_share_directory("seano_vision"))
            share_try_1 = share_dir / "models" / Path(p).name
            share_try_2 = share_dir / p
            if share_try_1.exists():
                return str(share_try_1)
            if share_try_2.exists():
                return str(share_try_2)
        except Exception:
            pass

        return p

    def _warmup_model(self) -> None:
        try:
            dummy = np.zeros((int(self.imgsz), int(self.imgsz), 3), dtype=np.uint8)
            device_arg = self.device if self.device != "" else None
            _ = self.model.predict(
                source=dummy,
                imgsz=int(self.imgsz),
                conf=float(self.conf),
                iou=float(self.iou),
                device=device_arg,
                verbose=False,
                half=bool(self.half),
                agnostic_nms=bool(self.agnostic_nms),
                max_det=int(self.max_det),
                classes=(sorted(list(self.class_ids)) if self.class_ids is not None else None),
            )
            self.get_logger().info("YOLO warmup done")
        except Exception as e:
            self.get_logger().warn(f"YOLO warmup skip: {e}")

    # ---------------- ROS callbacks ----------------
    def _on_image(self, msg: Image) -> None:
        # simpan frame terbaru saja (drop backlog)
        with self._lock:
            self._latest_msg = msg

    # ---------------- Core processing ----------------
    def _process_latest(self) -> None:
        # cegah overlap kalau executor multi-thread
        if not self._infer_lock.acquire(blocking=False):
            return

        try:
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

            if frame is None:
                return

            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return

            # YOLO inference
            try:
                device_arg = self.device if self.device != "" else None
                classes_arg = (sorted(list(self.class_ids)) if self.class_ids is not None else None)

                result = self.model.predict(
                    source=frame,
                    imgsz=int(self.imgsz),
                    conf=float(self.conf),
                    iou=float(self.iou),
                    device=device_arg,
                    verbose=False,
                    half=bool(self.half),
                    agnostic_nms=bool(self.agnostic_nms),
                    max_det=int(self.max_det),
                    classes=classes_arg,
                )[0]
            except Exception as e:
                self.get_logger().error(f"YOLO predict failed: {e}")
                return

            det_array = Detection2DArray()
            det_array.header = msg.header

            annotated = None
            if self.publish_annotated:
                annotated = frame.copy()

            det_count = 0

            if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                xyxy = _safe_to_numpy(boxes.xyxy)
                cls = _safe_to_numpy(boxes.cls).astype(int)
                confs = _safe_to_numpy(boxes.conf).astype(float)

                # urutkan dari confidence terbesar (biar konsisten + cap max_det jika perlu)
                if len(confs) > 1:
                    order = np.argsort(-confs)
                    xyxy = xyxy[order]
                    cls = cls[order]
                    confs = confs[order]

                for (x1, y1, x2, y2), c_id, score in zip(xyxy, cls, confs):
                    # filter ulang (double safety, kalau classes_arg None tapi class_ids diubah runtime)
                    if self.class_ids is not None and int(c_id) not in self.class_ids:
                        continue

                    x1 = float(np.clip(x1, 0, w - 1))
                    y1 = float(np.clip(y1, 0, h - 1))
                    x2 = float(np.clip(x2, 0, w - 1))
                    y2 = float(np.clip(y2, 0, h - 1))

                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    cx = x1 + bw / 2.0
                    cy = y1 + bh / 2.0

                    d = Detection2D()
                    d.header = msg.header
                    d.bbox.center.position.x = float(cx)
                    d.bbox.center.position.y = float(cy)
                    d.bbox.center.theta = 0.0
                    d.bbox.size_x = float(bw)
                    d.bbox.size_y = float(bh)

                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(int(c_id))
                    hyp.hypothesis.score = float(score)
                    d.results.append(hyp)

                    det_array.detections.append(d)
                    det_count += 1

                    # draw
                    if annotated is not None:
                        if self.draw_boxes:
                            p1 = (int(x1), int(y1))
                            p2 = (int(x2), int(y2))
                            cv2.rectangle(annotated, p1, p2, self.overlay_color, int(self.box_thickness))

                        if self.draw_labels:
                            if isinstance(self.names, dict):
                                name = self.names.get(int(c_id), str(int(c_id)))
                            else:
                                name = str(int(c_id))
                            label = f"{name} {float(score):.2f}"

                            tx = int(x1)
                            ty = max(18, int(y1) - 6)

                            if self.draw_label_bg:
                                (tw, th), base = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, float(self.font_scale), int(self.font_thickness)
                                )
                                bx1 = tx
                                by1 = max(0, ty - th - base - 6)
                                bx2 = min(w - 1, tx + tw + 8)
                                by2 = min(h - 1, ty + 4)

                                # alpha background
                                overlay = annotated.copy()
                                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                                annotated[:] = cv2.addWeighted(
                                    overlay, float(self.label_bg_alpha), annotated, 1.0 - float(self.label_bg_alpha), 0
                                )

                            cv2.putText(
                                annotated,
                                label,
                                (tx + 4, ty),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                float(self.font_scale),
                                self.overlay_color,
                                int(self.font_thickness),
                                cv2.LINE_AA
                            )

            proc_ms = (time.time() - t0) * 1000.0
            self._proc_ema_ms = proc_ms if self._frames == 0 else (
                self._ema_alpha * proc_ms + (1.0 - self._ema_alpha) * self._proc_ema_ms
            )
            self._frames += 1
            self._last_det_count = det_count

            if annotated is not None and self.draw_proc_ms:
                txt = f"proc={self._proc_ema_ms:.1f}ms"
                cv2.putText(
                    annotated,
                    txt,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    self.proc_text_color,
                    2,
                    cv2.LINE_AA
                )

            # publish detections
            if self.publish_detections:
                if det_count > 0 or self.publish_empty_detections:
                    self.pub_det_pub.publish(det_array)

            # publish annotated
            if self.publish_annotated and annotated is not None:
                try:
                    out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                    out_msg.header = msg.header
                    self.pub_img.publish(out_msg)
                except Exception as e:
                    self.get_logger().error(f"Publish annotated failed: {e}")

            # stats log
            self._maybe_log_stats()

        finally:
            try:
                self._infer_lock.release()
            except Exception:
                pass

    def _maybe_log_stats(self) -> None:
        if not self.stats_period or self.stats_period <= 0.0:
            return
        now = time.time()
        if now - self._last_stat_t < float(self.stats_period):
            return
        self._last_stat_t = now
        self.get_logger().info(
            f"det={self._last_det_count} | proc_ema={self._proc_ema_ms:.1f}ms | frames={self._frames} "
            f"| conf={self.conf:.2f} iou={self.iou:.2f} imgsz={self.imgsz} classes={self.class_ids_raw}"
        )


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
