#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO — False Positive Guard Node (FINAL)

Tujuan:
- Mengurangi false positive dari hasil deteksi YOLO sebelum
  masuk ke risk evaluator.
- Menggunakan filter ringan namun stabil:
    1) Score threshold
    2) Area (size) threshold
    3) Konsistensi temporal (N-of-M)
    4) Matching antar frame berbasis IoU
    5) (Opsional) Waterline gating (maritim)

Input:
- /camera/detections (vision_msgs/Detection2DArray)

Output:
- /camera/detections_filtered (vision_msgs/Detection2DArray)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32
from vision_msgs.msg import Detection2D, Detection2DArray


def det_to_xyxy(det: Detection2D) -> Tuple[float, float, float, float]:
    cx = float(det.bbox.center.position.x)
    cy = float(det.bbox.center.position.y)
    w = float(det.bbox.size_x)
    h = float(det.bbox.size_y)
    return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 1e-9 else float(inter / denom)


def best_class_score(det: Detection2D) -> Tuple[str, float]:
    if not det.results:
        return "unknown", 0.0
    h = det.results[0].hypothesis
    return str(h.class_id), float(h.score)


@dataclass
class Track:
    tid: int
    class_id: str
    bbox: Tuple[float, float, float, float]
    hit_hist: Deque[int] = field(default_factory=lambda: deque(maxlen=8))
    miss: int = 0
    last_det: Optional[Detection2D] = None


class FalsePositiveGuardNode(Node):
    def __init__(self) -> None:
        super().__init__("false_positive_guard_node")

        # ---------- Parameters ----------
        self.declare_parameter("enabled", True)
        self.declare_parameter("input_topic", "/camera/detections")
        self.declare_parameter("output_topic", "/camera/detections_filtered")

        self.declare_parameter("window_size", 8)
        self.declare_parameter("min_hits", 3)

        self.declare_parameter("iou_match", 0.35)
        self.declare_parameter("max_miss", 4)

        self.declare_parameter("min_score", 0.25)
        self.declare_parameter("min_area_px", 900.0)

        self.declare_parameter("use_waterline", True)
        self.declare_parameter("waterline_topic", "/vision/waterline_y")
        self.declare_parameter("waterline_margin_px", 15)

        # ---------- State ----------
        self._tracks: Dict[int, Track] = {}
        self._next_tid = 1
        self._waterline_y: Optional[int] = None

        # ---------- QoS ----------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.in_topic = self.get_parameter("input_topic").value
        self.out_topic = self.get_parameter("output_topic").value

        self.sub = self.create_subscription(Detection2DArray, self.in_topic, self.on_det, qos)
        self.pub = self.create_publisher(Detection2DArray, self.out_topic, qos)

        self.sub_wl = self.create_subscription(
            Int32,
            self.get_parameter("waterline_topic").value,
            self.on_waterline,
            10,
        )

        self.get_logger().info(f"[fp_guard] Ready | in={self.in_topic} out={self.out_topic}")

    def on_waterline(self, msg: Int32) -> None:
        self._waterline_y = int(msg.data)

    def _area(self, bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _passes_basic_filters(
        self, det: Detection2D, score: float, min_score: float, min_area: float
    ) -> bool:
        if score < min_score:
            return False
        return self._area(det_to_xyxy(det)) >= min_area

    def _passes_waterline(self, det: Detection2D, use_wl: bool, margin_px: int) -> bool:
        if not use_wl or self._waterline_y is None:
            return True
        _, _, _, y2 = det_to_xyxy(det)
        return float(y2) >= float(self._waterline_y - max(0, margin_px))

    def on_det(self, msg: Detection2DArray) -> None:
        if not bool(self.get_parameter("enabled").value):
            self.pub.publish(msg)
            return

        window_size = max(2, int(self.get_parameter("window_size").value))
        min_hits = max(1, int(self.get_parameter("min_hits").value))
        iou_match = float(self.get_parameter("iou_match").value)
        max_miss = max(0, int(self.get_parameter("max_miss").value))
        min_score = float(self.get_parameter("min_score").value)
        min_area = float(self.get_parameter("min_area_px").value)
        use_wl = bool(self.get_parameter("use_waterline").value)
        margin_px = int(self.get_parameter("waterline_margin_px").value)

        # -------- aging --------
        for tr in self._tracks.values():
            if tr.hit_hist.maxlen != window_size:
                tr.hit_hist = deque(list(tr.hit_hist), maxlen=window_size)
            tr.hit_hist.append(0)
            tr.miss += 1

        used = set()
        dets = sorted(
            msg.detections,
            key=lambda d: best_class_score(d)[1],
            reverse=True,
        )

        # -------- matching --------
        for det in dets:
            class_id, score = best_class_score(det)

            if not self._passes_basic_filters(det, score, min_score, min_area):
                continue
            if not self._passes_waterline(det, use_wl, margin_px):
                continue

            bbox = det_to_xyxy(det)

            best_tid, best_iou = None, 0.0
            for tid, tr in self._tracks.items():
                if tid in used or tr.class_id != class_id:
                    continue
                v = iou_xyxy(tr.bbox, bbox)
                if v >= iou_match and v > best_iou:
                    best_tid, best_iou = tid, v

            if best_tid is None:
                tid = self._next_tid
                self._next_tid += 1
                self._tracks[tid] = Track(
                    tid=tid,
                    class_id=class_id,
                    bbox=bbox,
                    hit_hist=deque([1], maxlen=window_size),
                    miss=0,
                    last_det=det,
                )
                used.add(tid)
            else:
                tr = self._tracks[best_tid]
                tr.bbox = bbox
                tr.miss = 0
                tr.hit_hist.pop()
                tr.hit_hist.append(1)
                tr.last_det = det
                used.add(best_tid)

        # -------- prune --------
        for tid in [t for t, tr in self._tracks.items() if tr.miss > max_miss]:
            self._tracks.pop(tid, None)

        # -------- publish confirmed --------
        out = Detection2DArray()
        out.header = msg.header
        for tr in self._tracks.values():
            if sum(tr.hit_hist) >= min_hits and tr.last_det is not None:
                out.detections.append(tr.last_det)

        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FalsePositiveGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
