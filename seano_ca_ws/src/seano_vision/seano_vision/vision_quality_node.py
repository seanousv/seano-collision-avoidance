#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO — Vision Quality Node

Tujuan:
- Menghitung kualitas citra (0..1) untuk "Perception Health".
- Dipakai untuk mode CAUTION (mis: low-light, blur, glare).

Input:
- /camera/image_raw_reliable (sensor_msgs/Image)

Output:
- /vision/quality         (std_msgs/Float32) 0..1
- /vision/quality_detail  (std_msgs/String)  JSON ringkas (opsional untuk debug)

Metode (ringan & stabil):
- Blur: Laplacian variance
- Brightness: mean intensity
- Contrast: std intensity
- Glare: rasio pixel sangat terang (>= glare_pixel)

Skor per komponen dinormalisasi 0..1 lalu digabung dengan bobot.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class VisionQualityNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_quality")

        # ---------- Parameters ----------
        self.declare_parameter("input_topic", "/camera/image_raw_reliable")
        self.declare_parameter("quality_topic", "/vision/quality")
        self.declare_parameter("detail_topic", "/vision/quality_detail")
        self.declare_parameter("publish_detail", True)

        # Speed control
        self.declare_parameter("downsample_w", 320)  # lebih kecil -> lebih cepat

        # Blur (Laplacian variance) tuning
        self.declare_parameter("blur_bad", 30.0)     # <= ini sangat blur
        self.declare_parameter("blur_good", 140.0)   # >= ini dianggap tajam

        # Brightness tuning (mean grayscale)
        self.declare_parameter("brightness_good_min", 60.0)
        self.declare_parameter("brightness_good_max", 190.0)

        # Contrast tuning (std grayscale)
        self.declare_parameter("contrast_bad", 15.0)
        self.declare_parameter("contrast_good", 50.0)

        # Glare tuning
        self.declare_parameter("glare_pixel", 245)       # pixel dianggap glare jika >= ini
        self.declare_parameter("glare_bad_ratio", 0.18)  # glare ratio >= ini = buruk

        # Weights
        self.declare_parameter("w_blur", 0.35)
        self.declare_parameter("w_brightness", 0.25)
        self.declare_parameter("w_contrast", 0.20)
        self.declare_parameter("w_glare", 0.20)

        # Rate limit publish (optional)
        self.declare_parameter("max_hz", 15.0)  # 0 = no limit

        # Read params
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.quality_topic = str(self.get_parameter("quality_topic").value)
        self.detail_topic = str(self.get_parameter("detail_topic").value)
        self.publish_detail = bool(self.get_parameter("publish_detail").value)

        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.input_topic, self.on_image, qos_img)
        self.pub_q = self.create_publisher(Float32, self.quality_topic, 10)
        self.pub_detail = self.create_publisher(String, self.detail_topic, 10)

        self._last_pub_t = 0.0

        self.get_logger().info(f"[vq] Ready | in={self.input_topic} -> {self.quality_topic}")

    def on_image(self, msg: Image) -> None:
        # Optional rate limit
        max_hz = float(self.get_parameter("max_hz").value)
        now = time.time()
        if max_hz > 0:
            min_dt = 1.0 / max_hz
            if (now - self._last_pub_t) < min_dt:
                return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return

        # Downsample for speed
        down_w = int(self.get_parameter("downsample_w").value)
        down_w = max(80, down_w)
        if w > down_w:
            down_h = max(60, int(h * (down_w / float(w))))
            frame = cv2.resize(frame, (down_w, down_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------- Compute components ----------
        # Blur
        blur_bad = float(self.get_parameter("blur_bad").value)
        blur_good = float(self.get_parameter("blur_good").value)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_var = float(lap.var())
        blur_score = clamp((blur_var - blur_bad) / max(1e-6, (blur_good - blur_bad)), 0.0, 1.0)

        # Brightness
        mean = float(np.mean(gray))
        bmin = float(self.get_parameter("brightness_good_min").value)
        bmax = float(self.get_parameter("brightness_good_max").value)

        if mean < bmin:
            bright_score = clamp(mean / max(1e-6, bmin), 0.0, 1.0)
        elif mean > bmax:
            bright_score = clamp(1.0 - ((mean - bmax) / max(1e-6, (255.0 - bmax))), 0.0, 1.0)
        else:
            bright_score = 1.0

        # Contrast
        contrast_bad = float(self.get_parameter("contrast_bad").value)
        contrast_good = float(self.get_parameter("contrast_good").value)
        std = float(np.std(gray))
        contrast_score = clamp((std - contrast_bad) / max(1e-6, (contrast_good - contrast_bad)), 0.0, 1.0)

        # Glare
        glare_pixel = int(self.get_parameter("glare_pixel").value)
        glare_bad_ratio = float(self.get_parameter("glare_bad_ratio").value)
        glare_ratio = float(np.mean(gray >= glare_pixel))
        glare_score = clamp(1.0 - (glare_ratio / max(1e-6, glare_bad_ratio)), 0.0, 1.0)

        # ---------- Combine ----------
        w_blur = float(self.get_parameter("w_blur").value)
        w_bri = float(self.get_parameter("w_brightness").value)
        w_con = float(self.get_parameter("w_contrast").value)
        w_gla = float(self.get_parameter("w_glare").value)

        wsum = max(1e-6, (w_blur + w_bri + w_con + w_gla))
        vq = (w_blur * blur_score + w_bri * bright_score + w_con * contrast_score + w_gla * glare_score) / wsum
        vq = float(clamp(vq, 0.0, 1.0))

        # Publish
        self.pub_q.publish(Float32(data=vq))
        self._last_pub_t = now

        if self.publish_detail:
            detail = {
                "vq": vq,
                "blur_var": blur_var,
                "blur_score": blur_score,
                "mean": mean,
                "bright_score": bright_score,
                "std": std,
                "contrast_score": contrast_score,
                "glare_ratio": glare_ratio,
                "glare_score": glare_score,
            }
            self.pub_detail.publish(String(data=json.dumps(detail)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionQualityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
