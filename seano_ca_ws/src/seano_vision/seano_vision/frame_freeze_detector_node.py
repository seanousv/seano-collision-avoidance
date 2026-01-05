#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO — Frame Freeze Detector Node

Tujuan:
- Deteksi kondisi "freeze" pada stream kamera:
  frame konten hampir tidak berubah selama N frame berturut-turut.
- Berguna untuk memicu state LOST PERCEPTION di layer CA.

Input:
- /camera/image_raw_reliable (sensor_msgs/Image)

Output:
- /vision/freeze        (std_msgs/Bool)    True jika terdeteksi freeze
- /vision/freeze_score  (std_msgs/Float32) 0..1, makin tinggi makin "frozen"
- /vision/freeze_reason (std_msgs/String)  "still" / "timeout" / "init"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class _FreezeState:
    prev_gray_small: Optional[np.ndarray] = None
    still_count: int = 0
    last_mean_diff: float = 999.0
    last_frame_wall: float = 0.0
    frozen: bool = False
    reason: str = "init"


class FrameFreezeDetectorNode(Node):
    """
    Metode deteksi freeze (ringan dan stabil):
    - Downsample frame -> grayscale
    - Hitung mean absolute difference (MAD) antara frame sekarang vs sebelumnya
    - Jika MAD < diff_threshold selama consecutive_frames berturut-turut -> freeze True

    Tambahan:
    - Timer "no_frame_timeout_s": kalau tidak ada frame masuk selama timeout -> freeze True (reason=timeout)
      Ini berguna sebagai safety signal sederhana (meski LOST utama tetap lebih cocok di risk evaluator).
    """

    def __init__(self) -> None:
        super().__init__("frame_freeze_detector")

        # ---------- Parameters ----------
        self.declare_parameter("input_topic", "/camera/image_raw_reliable")
        self.declare_parameter("freeze_topic", "/vision/freeze")
        self.declare_parameter("score_topic", "/vision/freeze_score")
        self.declare_parameter("reason_topic", "/vision/freeze_reason")

        self.declare_parameter("downsample_w", 160)          # makin kecil makin ringan
        self.declare_parameter("diff_threshold", 2.0)        # MAD threshold (0..255)
        self.declare_parameter("consecutive_frames", 15)     # jumlah frame berturut2
        self.declare_parameter("min_dt_s", 0.001)            # ignore duplicate ultra cepat

        self.declare_parameter("no_frame_timeout_s", 2.0)    # jika >0, publish freeze=True kalau timeout
        self.declare_parameter("timer_hz", 5.0)              # frequency cek timeout

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.freeze_topic = str(self.get_parameter("freeze_topic").value)
        self.score_topic = str(self.get_parameter("score_topic").value)
        self.reason_topic = str(self.get_parameter("reason_topic").value)

        self.downsample_w = int(self.get_parameter("downsample_w").value)
        self.diff_threshold = float(self.get_parameter("diff_threshold").value)
        self.consecutive_frames = int(self.get_parameter("consecutive_frames").value)
        self.min_dt_s = float(self.get_parameter("min_dt_s").value)

        self.no_frame_timeout_s = float(self.get_parameter("no_frame_timeout_s").value)
        self.timer_hz = float(self.get_parameter("timer_hz").value)

        # ---------- QoS ----------
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------- ROS IO ----------
        self.bridge = CvBridge()
        self.state = _FreezeState()
        self.state.last_frame_wall = time.time()

        self.sub = self.create_subscription(Image, self.input_topic, self.on_image, qos_img)
        self.pub_freeze = self.create_publisher(Bool, self.freeze_topic, 10)
        self.pub_score = self.create_publisher(Float32, self.score_topic, 10)
        self.pub_reason = self.create_publisher(String, self.reason_topic, 10)

        # Timer untuk cek timeout (opsional)
        if self.timer_hz > 0:
            self.create_timer(1.0 / self.timer_hz, self.on_timer)

        self.get_logger().info(
            "[freeze] Ready | "
            f"in={self.input_topic} thr={self.diff_threshold} N={self.consecutive_frames} "
            f"down_w={self.downsample_w} timeout={self.no_frame_timeout_s}s"
        )

    def on_timer(self) -> None:
        """Jika tidak ada frame masuk terlalu lama -> publish freeze True (reason=timeout)."""
        if self.no_frame_timeout_s <= 0:
            return

        now = time.time()
        dt = now - self.state.last_frame_wall
        if dt > self.no_frame_timeout_s:
            # Only force if not already frozen, biar tidak spam perubahan status
            self._publish(frozen=True, score=1.0, reason="timeout")

    def on_image(self, msg: Image) -> None:
        now = time.time()
        dt = now - self.state.last_frame_wall
        self.state.last_frame_wall = now

        # ignore ultra-fast duplicates
        if dt < self.min_dt_s:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"[freeze] cv_bridge failed: {e}")
            return

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return

        # --- Downsample keep aspect ratio ---
        target_w = max(48, int(self.downsample_w))
        target_h = max(36, int(h * (target_w / float(w))))

        small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.state.prev_gray_small is None:
            self.state.prev_gray_small = gray
            self.state.still_count = 0
            self.state.last_mean_diff = 999.0
            self._publish(frozen=False, score=0.0, reason="init")
            return

        # --- Mean absolute difference ---
        diff = cv2.absdiff(gray, self.state.prev_gray_small)
        mean_diff = float(np.mean(diff))
        self.state.last_mean_diff = mean_diff
        self.state.prev_gray_small = gray

        # Update still_count
        if mean_diff < self.diff_threshold:
            self.state.still_count += 1
        else:
            self.state.still_count = 0

        frozen = self.state.still_count >= self.consecutive_frames

        # Freeze score (0..1):
        # 1 kalau mean_diff=0 (benar2 diam),
        # 0 kalau mean_diff >= diff_threshold.
        base = 1.0 - clamp(mean_diff / max(1e-6, self.diff_threshold), 0.0, 1.0)

        # Kalau belum nyampe N frame, score dinaikkan bertahap sesuai progress still_count
        progress = clamp(self.state.still_count / max(1.0, float(self.consecutive_frames)), 0.0, 1.0)
        score = base * progress

        reason = "still" if frozen else "moving"
        self._publish(frozen=frozen, score=score, reason=reason)

    def _publish(self, frozen: bool, score: float, reason: str) -> None:
        # publish only if change? (tetap publish terus juga gapapa, tapi biar rapi kita publish tiap callback)
        self.state.frozen = bool(frozen)
        self.state.reason = str(reason)

        b = Bool()
        b.data = bool(frozen)
        self.pub_freeze.publish(b)

        s = Float32()
        s.data = float(clamp(score, 0.0, 1.0))
        self.pub_score.publish(s)

        r = String()
        r.data = str(reason)
        self.pub_reason.publish(r)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrameFreezeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
