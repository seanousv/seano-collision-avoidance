#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEANO — Frame Freeze Detector Node (UPDATED)

Perubahan inti (fix false-positive "still"):
- Freeze TRUE hanya jika frame REPEAT (konten nyaris identik/identik) selama N frame berturut-turut.
- "Stillness" (perubahan kecil) tidak dianggap freeze (freeze tetap FALSE), hanya jadi status/debug.

Metode:
- Downsample -> grayscale
- Hitung MAD (Mean Absolute Difference) antar frame
- Hitung hash frame (adler32) dari gray bytes
- Repeat terdeteksi jika hash sama berturut-turut (dan MAD kecil)

Input:
- /camera/image_raw_synced (sensor_msgs/Image) (default dari time_sync_node) :contentReference[oaicite:1]{index=1}

Output:
- /vision/freeze (std_msgs/Bool)  -> TRUE hanya jika repeat/stuck
- /vision/freeze_score (std_msgs/Float32) 0..1
- /vision/freeze_reason (std_msgs/String) init / moving / still / repeat / timeout
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional
import zlib

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class FreezeState:
    prev_gray: Optional[np.ndarray] = None
    prev_hash: Optional[int] = None

    still_count: int = 0  # MAD kecil (scene statis)
    repeat_count: int = 0  # hash sama (frame repeat/stuck)

    last_mean_diff: float = 999.0
    last_frame_wall: float = 0.0

    frozen: bool = False
    reason: str = "init"


class FrameFreezeDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("frame_freeze_detector_node")

        # ---------- Parameters ----------
        self.declare_parameter("input_topic", "/camera/image_raw_synced")
        self.declare_parameter("freeze_topic", "/vision/freeze")
        self.declare_parameter("score_topic", "/vision/freeze_score")
        self.declare_parameter("reason_topic", "/vision/freeze_reason")

        # Downsample width untuk perhitungan MAD/hash
        self.declare_parameter("downsample_w", 160)

        # Threshold “stillness” (MAD < ini berarti perubahan kecil)
        self.declare_parameter("diff_threshold", 2.0)

        # Freeze (repeat) jika hash sama >= N frame
        self.declare_parameter("consecutive_frames", 15)

        # Untuk menghindari double-callback terlalu rapat
        self.declare_parameter("min_dt_s", 0.001)

        # Jika tidak ada frame masuk
        self.declare_parameter("no_frame_timeout_s", 2.0)
        self.declare_parameter("timer_hz", 5.0)

        # QoS input (bisa disesuaikan jika perlu)
        self.declare_parameter("sub_reliability", "reliable")  # reliable | best_effort

        # Optional: kalau mau menganggap stillness sebagai freeze (TIDAK disarankan untuk lapangan)
        self.declare_parameter("freeze_on_stillness", False)

        # ---------- Read Parameters ----------
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

        self.sub_reliability = str(self.get_parameter("sub_reliability").value).strip().lower()
        self.freeze_on_stillness = bool(self.get_parameter("freeze_on_stillness").value)

        # ---------- QoS ----------
        rel = ReliabilityPolicy.RELIABLE
        if self.sub_reliability in ("best_effort", "besteffort", "be"):
            rel = ReliabilityPolicy.BEST_EFFORT

        qos_img = QoSProfile(
            reliability=rel,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------- ROS IO ----------
        self.bridge = CvBridge()
        self.state = FreezeState()
        self.state.last_frame_wall = time.time()

        self.sub = self.create_subscription(Image, self.input_topic, self.on_image, qos_img)
        self.pub_freeze = self.create_publisher(Bool, self.freeze_topic, 10)
        self.pub_score = self.create_publisher(Float32, self.score_topic, 10)
        self.pub_reason = self.create_publisher(String, self.reason_topic, 10)

        if self.timer_hz > 0.0:
            self.create_timer(1.0 / self.timer_hz, self.on_timer)

        self.get_logger().info(
            f"[freeze] Ready | in={self.input_topic} rel={self.sub_reliability} "
            f"diff_thr={self.diff_threshold} N={self.consecutive_frames} "
            f"freeze_on_stillness={self.freeze_on_stillness} timeout={self.no_frame_timeout_s}s"
        )

    def on_timer(self) -> None:
        if self.no_frame_timeout_s <= 0.0:
            return
        dt = time.time() - self.state.last_frame_wall
        if dt > self.no_frame_timeout_s and not self.state.frozen:
            self._publish(frozen=True, score=1.0, reason="timeout")

    def on_image(self, msg: Image) -> None:
        now = time.time()
        dt = now - self.state.last_frame_wall
        self.state.last_frame_wall = now
        if dt < self.min_dt_s:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"[freeze] cv_bridge error: {e}")
            return

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return

        target_w = max(48, self.downsample_w)
        target_h = max(36, int(h * (target_w / float(w))))

        small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # hash untuk deteksi repeat (frame stuck)
        cur_hash = int(zlib.adler32(gray.tobytes()) & 0xFFFFFFFF)

        if self.state.prev_gray is None or self.state.prev_hash is None:
            self.state.prev_gray = gray
            self.state.prev_hash = cur_hash
            self.state.still_count = 0
            self.state.repeat_count = 0
            self._publish(frozen=False, score=0.0, reason="init")
            return

        diff = cv2.absdiff(gray, self.state.prev_gray)
        mean_diff = float(np.mean(diff))
        self.state.last_mean_diff = mean_diff

        # update still_count (MAD kecil)
        if mean_diff < self.diff_threshold:
            self.state.still_count += 1
        else:
            self.state.still_count = 0

        # update repeat_count (hash sama) + juga wajib MAD kecil biar aman
        if (cur_hash == self.state.prev_hash) and (mean_diff < self.diff_threshold):
            self.state.repeat_count += 1
        else:
            self.state.repeat_count = 0

        # update prev
        self.state.prev_gray = gray
        self.state.prev_hash = cur_hash

        # keputusan freeze
        frozen_repeat = self.state.repeat_count >= self.consecutive_frames
        frozen_still = self.state.still_count >= self.consecutive_frames
        frozen = frozen_repeat or (self.freeze_on_stillness and frozen_still)

        # scoring (0..1)
        progress = clamp(
            self.state.repeat_count / max(float(self.consecutive_frames), 1.0), 0.0, 1.0
        )
        base = 1.0 - clamp(mean_diff / max(self.diff_threshold, 1e-6), 0.0, 1.0)
        score = clamp(base * progress, 0.0, 1.0)

        if frozen_repeat:
            reason = "repeat"
        elif self.freeze_on_stillness and frozen_still:
            reason = "still"
        else:
            # status non-fatal
            reason = "still" if mean_diff < self.diff_threshold else "moving"

        self._publish(frozen=frozen, score=score, reason=reason)

    def _publish(self, frozen: bool, score: float, reason: str) -> None:
        self.state.frozen = frozen
        self.state.reason = reason
        self.pub_freeze.publish(Bool(data=bool(frozen)))
        self.pub_score.publish(Float32(data=float(clamp(score, 0.0, 1.0))))
        self.pub_reason.publish(String(data=str(reason)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrameFreezeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
