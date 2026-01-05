#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO — waterline_horizon_node (Camera-Only)
ROS2 Humble

Tujuan:
- Estimasi waterline/horizon sederhana untuk ROI maritim (buang area atas/langit)
- Publish:
  1) /vision/waterline_y   (std_msgs/Int32) => posisi y waterline (pixel)
  2) /vision/roi_mask      (sensor_msgs/Image mono8) => 255 di bawah waterline, 0 di atas
  3) /vision/waterline_debug (sensor_msgs/Image bgr8) => overlay garis waterline + info

Metode (ringan, demo-friendly):
- Downscale gray -> blur -> canny -> HoughLinesP (cari garis hampir horizontal)
- Pilih garis dengan skor terbaik (panjang * kedekatan horizontal)
- Ambil waterline_y dari garis (y di tengah gambar)
- EMA smoothing supaya stabil
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Int32

_HAS_CV = True
try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge
except Exception:
    _HAS_CV = False
    cv2 = None
    np = None
    CvBridge = None


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class WaterlineHorizonNode(Node):
    def __init__(self) -> None:
        super().__init__("waterline_horizon_node")

        # -------- Parameters --------
        self.declare_parameter("input_topic", "/camera/image_raw_reliable")

        self.declare_parameter("waterline_topic", "/vision/waterline_y")
        self.declare_parameter("roi_mask_topic", "/vision/roi_mask")
        self.declare_parameter("debug_topic", "/vision/waterline_debug")

        self.declare_parameter("enable_debug", True)
        self.declare_parameter("publish_mask", True)

        # fallback & smoothing
        self.declare_parameter("default_ratio", 0.35)   # kalau gagal deteksi: y = ratio * H
        self.declare_parameter("ema_alpha", 0.25)       # smoothing 0..1 (lebih kecil = lebih halus)

        # performance
        self.declare_parameter("process_every_n", 1)    # proses tiap N frame
        self.declare_parameter("downscale_width", 480)  # biar ringan

        # Hough/Canny tuning
        self.declare_parameter("canny1", 60)
        self.declare_parameter("canny2", 160)
        self.declare_parameter("hough_thresh", 60)
        self.declare_parameter("min_line_length", 80)
        self.declare_parameter("max_line_gap", 12)
        self.declare_parameter("max_abs_slope", 0.25)   # makin kecil makin horizontal

        # search region (agar tidak kebawa garis bawah)
        self.declare_parameter("search_y_min_ratio", 0.10)
        self.declare_parameter("search_y_max_ratio", 0.70)

        # -------- State --------
        self.in_topic = str(self.get_parameter("input_topic").value)
        self.waterline_topic = str(self.get_parameter("waterline_topic").value)
        self.mask_topic = str(self.get_parameter("roi_mask_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.enable_debug = bool(self.get_parameter("enable_debug").value)
        self.publish_mask = bool(self.get_parameter("publish_mask").value)

        self.default_ratio = float(self.get_parameter("default_ratio").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)

        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))
        self.down_w = max(160, int(self.get_parameter("downscale_width").value))

        self.frame_i = 0
        self.last_y: Optional[float] = None
        self.last_w = 0
        self.last_h = 0
        self.last_t = time.time()

        self.bridge = CvBridge() if _HAS_CV else None

        # -------- QoS --------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(Image, self.in_topic, self.on_image, qos)
        self.pub_y = self.create_publisher(Int32, self.waterline_topic, 10)
        self.pub_mask = self.create_publisher(Image, self.mask_topic, 10)
        self.pub_dbg = self.create_publisher(Image, self.debug_topic, 10)

        self.get_logger().info(
            f"waterline_horizon_node started | cv={_HAS_CV} in={self.in_topic} out_y={self.waterline_topic}"
        )

    def _fallback_y(self, h: int) -> int:
        r = _clamp(self.default_ratio, 0.05, 0.95)
        return int(round(r * float(h)))

    def _pick_best_line(self, lines, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Pilih garis terbaik:
        - slope mendekati horizontal
        - panjang terbesar
        - berada di range y tertentu (search band)
        """
        if lines is None or len(lines) == 0:
            return None

        y_min = int(round(_clamp(float(self.get_parameter("search_y_min_ratio").value), 0.0, 1.0) * h))
        y_max = int(round(_clamp(float(self.get_parameter("search_y_max_ratio").value), 0.0, 1.0) * h))
        max_abs_slope = float(self.get_parameter("max_abs_slope").value)

        best = None
        best_score = -1.0

        for l in lines:
            x1, y1, x2, y2 = l[0]
            if x2 == x1:
                continue
            dy = float(y2 - y1)
            dx = float(x2 - x1)
            slope = dy / dx
            if abs(slope) > max_abs_slope:
                continue

            ym = int(round((y1 + y2) * 0.5))
            if ym < y_min or ym > y_max:
                continue

            length = math.hypot(dx, dy)
            # score: panjang * (1 - abs(slope)/max_abs_slope)
            score = float(length) * (1.0 - min(1.0, abs(slope) / max_abs_slope))

            if score > best_score:
                best_score = score
                best = (int(x1), int(y1), int(x2), int(y2))

        return best

    def _y_at_x(self, line: Tuple[int, int, int, int], x: float) -> float:
        x1, y1, x2, y2 = line
        if x2 == x1:
            return float((y1 + y2) * 0.5)
        t = (x - float(x1)) / (float(x2) - float(x1))
        return float(y1) + t * (float(y2) - float(y1))

    def _estimate_waterline(self, img_bgr) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
        h, w = img_bgr.shape[:2]
        if not _HAS_CV:
            return self._fallback_y(h), None

        # downscale
        if w > self.down_w:
            scale = float(self.down_w) / float(w)
            dw = self.down_w
            dh = int(round(h * scale))
            small = cv2.resize(img_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            small = img_bgr

        sh, sw = small.shape[:2]

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        c1 = int(self.get_parameter("canny1").value)
        c2 = int(self.get_parameter("canny2").value)
        edges = cv2.Canny(gray, c1, c2)

        thr = int(self.get_parameter("hough_thresh").value)
        min_len = int(self.get_parameter("min_line_length").value)
        gap = int(self.get_parameter("max_line_gap").value)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180.0,
            threshold=thr,
            minLineLength=min_len,
            maxLineGap=gap
        )

        best = self._pick_best_line(lines, sw, sh)
        if best is None:
            # fallback
            y = int(round(float(self._fallback_y(sh))))
        else:
            # waterline_y: y di tengah x
            y = int(round(self._y_at_x(best, x=float(sw) * 0.5)))
            y = int(_clamp(y, 0, sh - 1))

        # scale back to original
        if w > self.down_w:
            scale_back = float(w) / float(sw)
            y_full = int(round(float(y) * scale_back * (float(sh) / float(h)) * (float(h) / float(sh))))
            # (di atas terlihat redundant, tapi aman; intinya y_full = y * (h/sh))
            y_full = int(round(float(y) * (float(h) / float(sh))))
        else:
            y_full = y

        y_full = int(_clamp(y_full, 0, h - 1))
        return y_full, best

    def on_image(self, msg: Image) -> None:
        self.frame_i += 1
        if (self.frame_i % self.process_every_n) != 0:
            return

        h = int(msg.height)
        w = int(msg.width)
        if h <= 0 or w <= 0:
            return

        self.last_w, self.last_h = w, h

        # default
        y_est = self._fallback_y(h)
        best_line = None

        if _HAS_CV and self.bridge is not None:
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                y_est, best_line = self._estimate_waterline(img)
            except Exception:
                y_est = self._fallback_y(h)
                best_line = None

        # EMA smoothing
        if self.last_y is None:
            y_s = float(y_est)
        else:
            a = _clamp(self.ema_alpha, 0.0, 1.0)
            y_s = (1.0 - a) * float(self.last_y) + a * float(y_est)

        self.last_y = _clamp(y_s, 0.0, float(h - 1))
        y_out = int(round(self.last_y))

        # publish y
        m = Int32()
        m.data = int(y_out)
        self.pub_y.publish(m)

        # publish mask
        if self.publish_mask:
            if _HAS_CV and self.bridge is not None:
                try:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    mask[y_out:, :] = 255
                    out = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
                    out.header = msg.header
                    self.pub_mask.publish(out)
                except Exception:
                    pass

        # publish debug overlay
        if self.enable_debug and _HAS_CV and self.bridge is not None:
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.line(img, (0, y_out), (w - 1, y_out), (0, 255, 255), 2)
                cv2.putText(
                    img,
                    f"waterline_y={y_out}  (default_ratio={self.default_ratio:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out.header = msg.header
                self.pub_dbg.publish(out)
            except Exception:
                pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaterlineHorizonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
