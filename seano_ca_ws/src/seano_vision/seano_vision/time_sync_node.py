#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO - time_sync_node
Tujuan:
- Menjaga header.stamp Image supaya "waras":
  - tidak 0
  - tidak mundur (backward)
  - tidak loncat jauh (jump)
  - tidak terlalu beda dengan jam ROS (skew)
- Kalau stamp tidak valid -> diganti dengan now() (dan dijaga tetap monotonik).
Output:
- Image tersinkron: /camera/image_raw_reliable (atau sesuai param)
- /vision/time_ok (Bool)
- /vision/time_status (String) ringkas untuk debug
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from builtin_interfaces.msg import Time


def time_to_sec(t: Time) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9


def sec_to_time(x: float) -> Time:
    if x < 0.0:
        x = 0.0
    sec = int(math.floor(x))
    nsec = int(round((x - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    msg = Time()
    msg.sec = sec
    msg.nanosec = nsec
    return msg


@dataclass
class SyncStats:
    total: int = 0
    corrected: int = 0
    last_reason: str = "init"
    last_time_ok: bool = True


class TimeSyncNode(Node):
    def __init__(self) -> None:
        super().__init__("time_sync_node")

        # -------- Parameters --------
        self.declare_parameter("input_topic", "/camera/image_raw_reliable")
        self.declare_parameter("output_topic", "/camera/image_raw_synced")
        self.declare_parameter("time_ok_topic", "/vision/time_ok")
        self.declare_parameter("time_status_topic", "/vision/time_status")

        self.declare_parameter("expected_fps", 15.0)
        self.declare_parameter("max_backward_sec", 0.05)   # stamp mundur lebih dari ini = invalid
        self.declare_parameter("max_jump_sec", 2.0)         # loncat maju lebih dari ini = invalid
        self.declare_parameter("max_skew_sec", 5.0)         # beda dengan now() lebih dari ini = invalid
        self.declare_parameter("force_monotonic", True)     # pastikan output stamp selalu naik
        self.declare_parameter("status_hz", 2.0)            # publish status summary rate

        self.in_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.out_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.time_ok_topic = self.get_parameter("time_ok_topic").get_parameter_value().string_value
        self.time_status_topic = self.get_parameter("time_status_topic").get_parameter_value().string_value

        self.expected_fps = float(self.get_parameter("expected_fps").value)
        self.max_backward = float(self.get_parameter("max_backward_sec").value)
        self.max_jump = float(self.get_parameter("max_jump_sec").value)
        self.max_skew = float(self.get_parameter("max_skew_sec").value)
        self.force_monotonic = bool(self.get_parameter("force_monotonic").value)
        self.status_hz = float(self.get_parameter("status_hz").value)

        self.min_dt = 1.0 / max(self.expected_fps, 1.0)

        # -------- QoS --------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        # -------- Pub/Sub --------
        self.sub = self.create_subscription(Image, self.in_topic, self.on_image, qos)
        self.pub_img = self.create_publisher(Image, self.out_topic, qos)
        self.pub_ok = self.create_publisher(Bool, self.time_ok_topic, 10)
        self.pub_status = self.create_publisher(String, self.time_status_topic, 10)

        self.stats = SyncStats()
        self.last_in_stamp: float | None = None
        self.last_out_stamp: float | None = None

        if self.status_hz > 0.0:
            period = 1.0 / self.status_hz
            self.create_timer(period, self.publish_status)

        self.get_logger().info(
            f"[time_sync_node] in={self.in_topic} out={self.out_topic} "
            f"expected_fps={self.expected_fps:.1f} max_skew={self.max_skew:.2f}s"
        )

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def validate_stamp(self, stamp_sec: float, now_sec: float) -> tuple[bool, str]:
        if stamp_sec <= 0.0:
            return False, "zero_stamp"

        if self.last_in_stamp is not None:
            if stamp_sec < self.last_in_stamp - self.max_backward:
                return False, "backward"
            if stamp_sec > self.last_in_stamp + self.max_jump:
                return False, "jump"

        if abs(stamp_sec - now_sec) > self.max_skew:
            return False, "skew"

        return True, "ok"

    def enforce_monotonic(self, out_stamp: float) -> tuple[float, bool]:
        """Return (stamp, adjusted_flag)"""
        if not self.force_monotonic:
            return out_stamp, False
        if self.last_out_stamp is None:
            return out_stamp, False
        if out_stamp <= self.last_out_stamp:
            return self.last_out_stamp + self.min_dt, True
        return out_stamp, False

    def on_image(self, msg: Image) -> None:
        self.stats.total += 1

        in_stamp = time_to_sec(msg.header.stamp)
        now = self.now_sec()

        ok, reason = self.validate_stamp(in_stamp, now)

        out_stamp = in_stamp
        corrected = False

        if not ok:
            out_stamp = now
            corrected = True

        out_stamp, adjusted = self.enforce_monotonic(out_stamp)
        if adjusted:
            # Ini bukan fatal; cuma biar tidak “mundur” saat downstream pakai stamp utk matching.
            reason = "monotonic_adjust" if ok else reason
            corrected = True

        # publish time_ok
        ok_msg = Bool()
        ok_msg.data = bool(ok)
        self.pub_ok.publish(ok_msg)

        # publish image (deepcopy biar aman)
        out_msg = copy.deepcopy(msg)
        out_msg.header.stamp = sec_to_time(out_stamp)
        self.pub_img.publish(out_msg)

        # update state
        self.last_in_stamp = in_stamp
        self.last_out_stamp = out_stamp

        self.stats.last_reason = reason
        self.stats.last_time_ok = bool(ok)
        if corrected:
            self.stats.corrected += 1

    def publish_status(self) -> None:
        ratio = 0.0 if self.stats.total == 0 else (100.0 * self.stats.corrected / self.stats.total)
        s = String()
        s.data = (
            f"total={self.stats.total} corrected={self.stats.corrected} "
            f"({ratio:.1f}%) last_reason={self.stats.last_reason} time_ok={self.stats.last_time_ok}"
        )
        self.pub_status.publish(s)


def main() -> None:
    rclpy.init()
    node = TimeSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
