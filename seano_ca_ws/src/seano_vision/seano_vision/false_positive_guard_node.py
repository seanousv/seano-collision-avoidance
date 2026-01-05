#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO — false_positive_guard_node (N-of-M + Size/Score + Waterline Gating)

Tujuan:
- Turunin false positive sebelum masuk risk evaluator.
- Filter sederhana tapi efektif:
  1) min_score
  2) min_area_px
  3) N-of-M consistency (min_hits dari window_size)
  4) matching antar frame pakai IoU
  5) (opsional) waterline gating: drop deteksi yang jelas-jelas ada di area langit (di atas waterline)

Catatan penting:
- Gating waterline dibuat "aman": deteksi TETAP lolos kalau bbox menyentuh/di bawah waterline.
  Drop hanya kalau bbox bawah (y2) masih jauh di atas waterline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from vision_msgs.msg import Detection2DArray, Detection2D
from std_msgs.msg import Int32


def det_to_xyxy(det: Detection2D) -> Tuple[float, float, float, float]:
    # sesuai output ros2 topic echo kamu: bbox.center.position.x/y dan bbox.size_x/size_y
    cx = float(det.bbox.center.position.x)
    cy = float(det.bbox.center.position.y)
    w = float(det.bbox.size_x)
    h = float(det.bbox.size_y)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 1e-9 else float(inter / denom)


def best_class_score(det: Detection2D) -> Tuple[str, float]:
    if not det.results:
        return "unknown", 0.0
    # ambil yang pertama (umumnya best)
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

        # -------- params --------
        self.declare_parameter("enabled", True)
        self.declare_parameter("input_topic", "/camera/detections")
        self.declare_parameter("output_topic", "/camera/detections_filtered")

        # N-of-M
        self.declare_parameter("window_size", 8)   # M
        self.declare_parameter("min_hits", 3)      # N

        # matching
        self.declare_parameter("iou_match", 0.35)
        self.declare_parameter("max_miss", 4)

        # size/score
        self.declare_parameter("min_score", 0.25)
        self.declare_parameter("min_area_px", 900.0)

        # waterline gating
        self.declare_parameter("use_waterline", True)
        self.declare_parameter("waterline_topic", "/vision/waterline_y")
        self.declare_parameter("waterline_margin_px", 15)
        # RULE: drop kalau y2 < (waterline_y - margin)
        # artinya bbox bener-bener di atas waterline (langit)

        self._waterline_y: Optional[int] = None

        # -------- qos --------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.in_topic = str(self.get_parameter("input_topic").value)
        self.out_topic = str(self.get_parameter("output_topic").value)

        self.sub = self.create_subscription(Detection2DArray, self.in_topic, self.on_det, qos)
        self.pub = self.create_publisher(Detection2DArray, self.out_topic, qos)

        self.sub_wl = self.create_subscription(
            Int32,
            str(self.get_parameter("waterline_topic").value),
            self.on_waterline,
            10,
        )

        self._tracks: Dict[int, Track] = {}
        self._next_tid = 1

        self.get_logger().info(
            f"fp_guard ready | in={self.in_topic} out={self.out_topic} wl_topic={self.get_parameter('waterline_topic').value}"
        )

    def on_waterline(self, msg: Int32) -> None:
        self._waterline_y = int(msg.data)

    def _area(self, bbox: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _passes_score_size(self, det: Detection2D, score: float, min_score: float, min_area: float) -> bool:
        if score < min_score:
            return False
        bbox = det_to_xyxy(det)
        return self._area(bbox) >= min_area

    def _passes_waterline(self, det: Detection2D, use_wl: bool, margin_px: int) -> bool:
        if not use_wl:
            return True
        if self._waterline_y is None:
            # kalau waterline belum ada, jangan drop dulu
            return True

        _, _, _, y2 = det_to_xyxy(det)
        wl = float(self._waterline_y)
        margin = float(max(0, margin_px))

        # DROP kalau bbox bawah masih di atas (wl - margin)
        # KEEP kalau bbox menyentuh / di bawah waterline
        return float(y2) >= (wl - margin)

    def on_det(self, msg: Detection2DArray) -> None:
        # -------- ambil param runtime (biar gampang tuning) --------
        enabled = bool(self.get_parameter("enabled").value)

        window_size = max(2, int(self.get_parameter("window_size").value))
        min_hits = max(1, int(self.get_parameter("min_hits").value))

        iou_match = float(self.get_parameter("iou_match").value)
        max_miss = max(0, int(self.get_parameter("max_miss").value))

        min_score = float(self.get_parameter("min_score").value)
        min_area = float(self.get_parameter("min_area_px").value)

        use_wl = bool(self.get_parameter("use_waterline").value)
        margin_px = int(self.get_parameter("waterline_margin_px").value)

        # -------- passthrough --------
        if not enabled:
            self.pub.publish(msg)
            return

        # -------- aging --------
        for t in self._tracks.values():
            if t.hit_hist.maxlen != window_size:
                t.hit_hist = deque(list(t.hit_hist), maxlen=window_size)
            t.hit_hist.append(0)
            t.miss += 1

        # -------- match dets -> tracks --------
        dets = list(msg.detections)
        dets_sorted = sorted(dets, key=lambda d: best_class_score(d)[1], reverse=True)

        used_tracks = set()

        for det in dets_sorted:
            class_id, score = best_class_score(det)
            if not self._passes_score_size(det, score, min_score, min_area):
                continue
            if not self._passes_waterline(det, use_wl, margin_px):
                continue

            bbox = det_to_xyxy(det)

            best_tid = None
            best_iou = 0.0
            for tid, tr in self._tracks.items():
                if tid in used_tracks:
                    continue
                if tr.class_id != class_id:
                    continue
                v = iou_xyxy(tr.bbox, bbox)
                if v >= iou_match and v > best_iou:
                    best_iou = v
                    best_tid = tid

            if best_tid is None:
                tid = self._next_tid
                self._next_tid += 1
                tr = Track(
                    tid=tid,
                    class_id=class_id,
                    bbox=bbox,
                    hit_hist=deque([1], maxlen=window_size),
                    miss=0,
                    last_det=det,
                )
                self._tracks[tid] = tr
                used_tracks.add(tid)
            else:
                tr = self._tracks[best_tid]
                tr.bbox = bbox
                tr.miss = 0
                # set hit di frame ini jadi 1
                if tr.hit_hist:
                    tr.hit_hist.pop()
                tr.hit_hist.append(1)
                tr.last_det = det
                used_tracks.add(best_tid)

        # -------- prune dead tracks --------
        dead = [tid for tid, tr in self._tracks.items() if tr.miss > max_miss]
        for tid in dead:
            self._tracks.pop(tid, None)

        # -------- publish confirmed only --------
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
