#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO — multi_target_fusion_node
ROS2 Humble

Tujuan:
- Menstabilkan pemilihan target utama dan mengurangi jitter pada CA.
- Input : Detection2DArray (mis. /camera/detections_filtered)
- Output: Detection2DArray (mis. /camera/detections_fused) yang sudah:
  - di-sort berdasarkan "threat score" (paling bahaya di index awal), atau
  - hanya top-K detections (lebih ringan untuk risk evaluator)

Threat score (default):
- lebih tinggi kalau:
  1) bbox bawah lebih dekat ke bawah frame (y2 tinggi) -> lebih "dekat" secara proxy
  2) bbox area lebih besar
  3) bbox lebih dekat ke tengah (corridor-ish)

Node ini ringan (tanpa OpenCV), hanya baca metadata image untuk normalisasi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D


def det_to_xyxy(det: Detection2D) -> Tuple[float, float, float, float]:
    cx = float(det.bbox.center.position.x)
    cy = float(det.bbox.center.position.y)
    w = float(det.bbox.size_x)
    h = float(det.bbox.size_y)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return x1, y1, x2, y2


def best_class_score(det: Detection2D) -> Tuple[str, float]:
    # ambil hypothesis pertama (umumnya best)
    if not det.results:
        return "unknown", 0.0
    h = det.results[0].hypothesis
    return str(h.class_id), float(h.score)


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


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class Track:
    tid: int
    class_id: str
    bbox: Tuple[float, float, float, float]
    miss: int = 0
    score_ema: float = 0.0
    det_last: Optional[Detection2D] = None


class MultiTargetFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("multi_target_fusion_node")

        # ---------------- Parameters ----------------
        self.declare_parameter("enabled", True)

        self.declare_parameter("input_topic", "/camera/detections_filtered")
        self.declare_parameter("output_topic", "/camera/detections_fused")

        # for normalization
        self.declare_parameter("image_topic", "/camera/image_raw_reliable")

        # output mode
        # "topk" -> output hanya top_k detections
        # "sort_all" -> output semua detections tapi urut threat tinggi ke rendah
        self.declare_parameter("output_mode", "topk")
        self.declare_parameter("top_k", 3)

        # filter small junk (opsional)
        self.declare_parameter("min_score", 0.0)
        self.declare_parameter("min_area_px", 0.0)

        # threat weights
        self.declare_parameter("w_bottom", 0.55)
        self.declare_parameter("w_area", 0.25)
        self.declare_parameter("w_center", 0.20)
        self.declare_parameter("w_det_score", 0.05)

        # tracking (light)
        self.declare_parameter("use_tracking", True)
        self.declare_parameter("iou_match", 0.35)
        self.declare_parameter("max_miss", 6)
        self.declare_parameter("bbox_ema_alpha", 0.40)
        self.declare_parameter("score_ema_alpha", 0.30)

        # ---------------- State ----------------
        self.enabled = bool(self.get_parameter("enabled").value)
        self.in_topic = str(self.get_parameter("input_topic").value)
        self.out_topic = str(self.get_parameter("output_topic").value)
        self.img_topic = str(self.get_parameter("image_topic").value)

        self.last_w: Optional[int] = None
        self.last_h: Optional[int] = None

        self.tracks: Dict[int, Track] = {}
        self.next_tid = 1

        # ---------------- QoS ----------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_img = self.create_subscription(Image, self.img_topic, self.on_image, qos)
        self.sub_det = self.create_subscription(Detection2DArray, self.in_topic, self.on_det, qos)
        self.pub = self.create_publisher(Detection2DArray, self.out_topic, qos)

        self.get_logger().info(
            f"multi_target_fusion_node started | enabled={self.enabled} in={self.in_topic} out={self.out_topic} img={self.img_topic}"
        )

    def on_image(self, msg: Image) -> None:
        # cuma butuh width/height untuk normalisasi
        if msg.width > 0 and msg.height > 0:
            self.last_w = int(msg.width)
            self.last_h = int(msg.height)

    def _threat_score(self, det: Detection2D) -> float:
        # base values
        x1, y1, x2, y2 = det_to_xyxy(det)
        cx = float(det.bbox.center.position.x)
        class_id, det_score = best_class_score(det)

        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        # normalization (kalau belum tau image size, fallback kasar)
        if self.last_w is None or self.last_h is None:
            y2n = clamp(y2 / 480.0, 0.0, 1.0)
            arean = clamp(area / (640.0 * 480.0), 0.0, 1.0)
            cxn = clamp(cx / 640.0, 0.0, 1.0)
        else:
            W = float(self.last_w)
            H = float(self.last_h)
            y2n = clamp(y2 / H, 0.0, 1.0)
            arean = clamp(area / (W * H), 0.0, 1.0)
            cxn = clamp(cx / W, 0.0, 1.0)

        # center score 0..1 (1 = center)
        center = 1.0 - min(1.0, abs(cxn - 0.5) * 2.0)

        w_bottom = float(self.get_parameter("w_bottom").value)
        w_area = float(self.get_parameter("w_area").value)
        w_center = float(self.get_parameter("w_center").value)
        w_det = float(self.get_parameter("w_det_score").value)

        # area pakai sqrt supaya tidak over-dominant
        score = (
            w_bottom * y2n +
            w_area * (arean ** 0.5) +
            w_center * center +
            w_det * clamp(det_score, 0.0, 1.0)
        )
        return float(score)

    def _update_tracks(self, dets: List[Detection2D]) -> List[Tuple[Detection2D, int, float]]:
        """
        Return list of (det, tid, threat_score).
        Kalau tracking disabled: tid=-1.
        """
        use_tracking = bool(self.get_parameter("use_tracking").value)
        if not use_tracking:
            out = []
            for d in dets:
                out.append((d, -1, self._threat_score(d)))
            return out

        iou_match = float(self.get_parameter("iou_match").value)
        max_miss = int(self.get_parameter("max_miss").value)
        a_bbox = float(self.get_parameter("bbox_ema_alpha").value)
        a_score = float(self.get_parameter("score_ema_alpha").value)

        # aging tracks
        for t in self.tracks.values():
            t.miss += 1

        assigned: Dict[int, bool] = {}

        results: List[Tuple[Detection2D, int, float]] = []

        # match per detection
        for det in dets:
            cid, _ = best_class_score(det)
            bbox = det_to_xyxy(det)
            sc = self._threat_score(det)

            best_tid = None
            best_iou = 0.0

            for tid, tr in self.tracks.items():
                if assigned.get(tid, False):
                    continue
                if tr.class_id != cid:
                    continue
                v = iou_xyxy(tr.bbox, bbox)
                if v >= iou_match and v > best_iou:
                    best_iou = v
                    best_tid = tid

            if best_tid is None:
                tid = self.next_tid
                self.next_tid += 1
                tr = Track(tid=tid, class_id=cid, bbox=bbox, miss=0, score_ema=sc, det_last=det)
                self.tracks[tid] = tr
                assigned[tid] = True
                results.append((det, tid, sc))
            else:
                tr = self.tracks[best_tid]
                # EMA bbox
                x1o, y1o, x2o, y2o = tr.bbox
                x1n, y1n, x2n, y2n = bbox
                tr.bbox = (
                    (1 - a_bbox) * x1o + a_bbox * x1n,
                    (1 - a_bbox) * y1o + a_bbox * y1n,
                    (1 - a_bbox) * x2o + a_bbox * x2n,
                    (1 - a_bbox) * y2o + a_bbox * y2n,
                )
                tr.miss = 0
                tr.score_ema = (1 - a_score) * tr.score_ema + a_score * sc
                tr.det_last = det
                assigned[best_tid] = True
                results.append((det, best_tid, tr.score_ema))

        # prune dead
        dead = [tid for tid, tr in self.tracks.items() if tr.miss > max_miss]
        for tid in dead:
            self.tracks.pop(tid, None)

        return results

    def on_det(self, msg: Detection2DArray) -> None:
        self.enabled = bool(self.get_parameter("enabled").value)

        if not self.enabled:
            # passthrough
            self.pub.publish(msg)
            return

        output_mode = str(self.get_parameter("output_mode").value).strip().lower()
        top_k = int(self.get_parameter("top_k").value)

        min_score = float(self.get_parameter("min_score").value)
        min_area = float(self.get_parameter("min_area_px").value)

        dets_in = []
        for d in msg.detections:
            cid, sc = best_class_score(d)
            if sc < min_score:
                continue
            x1, y1, x2, y2 = det_to_xyxy(d)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < min_area:
                continue
            dets_in.append(d)

        tracked = self._update_tracks(dets_in)

        # sort by score (desc)
        tracked_sorted = sorted(tracked, key=lambda x: x[2], reverse=True)

        if output_mode == "topk":
            if top_k <= 0:
                tracked_sorted = []
            else:
                tracked_sorted = tracked_sorted[:top_k]
        else:
            # sort_all
            pass

        out = Detection2DArray()
        out.header = msg.header

        # copy detections in new order, set det.id to track id (optional)
        for det, tid, sc in tracked_sorted:
            dd = det  # we can reuse object; but safer make a shallow copy by constructing new
            # karena Detection2D bukan dataclass, paling aman: pakai langsung dan set id
            if tid >= 0:
                dd.id = f"id{tid}"
            out.detections.append(dd)

        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MultiTargetFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
