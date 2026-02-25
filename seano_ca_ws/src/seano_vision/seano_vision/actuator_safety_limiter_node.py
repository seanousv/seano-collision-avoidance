#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ActuatorSafetyLimiterNode (SEANO) - FINAL LIMITER AFTER MUX

Tujuan:
- Node ini adalah "pagar terakhir" sebelum bridge ke MAVROS.
- Dia mengambil output dari MUX (/seano/selected/left_cmd & right_cmd),
  lalu menerapkan:
  1) failsafe dari watchdog (/ca/failsafe_active)
  2) timeout input (kalau command berhenti publish -> STOP)
  3) clamp + slew-rate limiter (gerak halus)

Arsitektur yang disarankan:
teleop/manual + auto -> command_mux -> selected left/right -> [THIS NODE] -> /seano/left_cmd /seano/right_cmd -> mavros_rc_override_bridge

Default:
- Input:  /seano/selected/left_cmd, /seano/selected/right_cmd
- Output: /seano/left_cmd, /seano/right_cmd
- Failsafe: /ca/failsafe_active (Bool)
"""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def slew_limit(current: float, target: float, rate_per_s: float, dt: float) -> float:
    if rate_per_s <= 0.0 or dt <= 0.0:
        return target
    max_delta = rate_per_s * dt
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


def is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


class ActuatorSafetyLimiterNode(Node):
    def __init__(self) -> None:
        super().__init__("actuator_safety_limiter_node")

        # ---------- Topics ----------
        self.declare_parameter("in_left_topic", "/seano/selected/left_cmd")
        self.declare_parameter("in_right_topic", "/seano/selected/right_cmd")
        self.declare_parameter("failsafe_active_topic", "/ca/failsafe_active")

        self.declare_parameter("out_left_topic", "/seano/left_cmd")
        self.declare_parameter("out_right_topic", "/seano/right_cmd")

        # Debug/info
        self.declare_parameter("reason_topic", "/seano/limiter_reason")
        self.declare_parameter("log_stats_sec", 1.0)

        # ---------- Safety / timing ----------
        self.declare_parameter("input_timeout_s", 0.6)  # kalau selected cmd stale -> STOP
        self.declare_parameter("failsafe_timeout_s", 2.0)  # kalau failsafe topic stale
        self.declare_parameter("failsafe_stale_is_active", True)

        self.declare_parameter("loop_hz", 20.0)

        # ---------- Output shaping ----------
        self.declare_parameter("allow_reverse", False)  # default USV kamu 0..1
        self.declare_parameter("deadband", 0.0)  # misal 0.02 kalau mau
        self.declare_parameter("slew_left_per_s", 1.2)
        self.declare_parameter("slew_right_per_s", 1.2)

        # ---------- State ----------
        self._last_left_in = 0.0
        self._last_right_in = 0.0
        self._t_left = 0.0
        self._t_right = 0.0

        self._failsafe_active = False
        self._t_failsafe = 0.0

        self._left_out = 0.0
        self._right_out = 0.0

        self._last_tick = time.time()
        self._last_log = 0.0
        self._last_reason = ""

        # ---------- I/O ----------
        in_left = str(self.get_parameter("in_left_topic").value)
        in_right = str(self.get_parameter("in_right_topic").value)
        fs_topic = str(self.get_parameter("failsafe_active_topic").value)

        out_left = str(self.get_parameter("out_left_topic").value)
        out_right = str(self.get_parameter("out_right_topic").value)
        reason_topic = str(self.get_parameter("reason_topic").value)

        self.create_subscription(Float32, in_left, self._on_left, 10)
        self.create_subscription(Float32, in_right, self._on_right, 10)
        self.create_subscription(Bool, fs_topic, self._on_failsafe, 10)

        self.pub_left = self.create_publisher(Float32, out_left, 10)
        self.pub_right = self.create_publisher(Float32, out_right, 10)
        self.pub_reason = self.create_publisher(String, reason_topic, 10)

        loop_hz = float(self.get_parameter("loop_hz").value)
        loop_hz = 20.0 if loop_hz <= 0 else loop_hz
        self.create_timer(1.0 / loop_hz, self._on_tick)

        self.get_logger().info(
            "Started | "
            f"in=({in_left},{in_right}) fs={fs_topic} "
            f"out=({out_left},{out_right}) loop={loop_hz:.1f}Hz"
        )

    def _on_left(self, msg: Float32) -> None:
        v = float(msg.data)
        if is_finite(v):
            self._last_left_in = v
            self._t_left = time.time()

    def _on_right(self, msg: Float32) -> None:
        v = float(msg.data)
        if is_finite(v):
            self._last_right_in = v
            self._t_right = time.time()

    def _on_failsafe(self, msg: Bool) -> None:
        self._failsafe_active = bool(msg.data)
        self._t_failsafe = time.time()

    def _on_tick(self) -> None:
        now = time.time()
        dt = max(1e-3, now - self._last_tick)
        self._last_tick = now

        input_timeout = float(self.get_parameter("input_timeout_s").value)
        fs_timeout = float(self.get_parameter("failsafe_timeout_s").value)
        fs_stale_is_active = bool(self.get_parameter("failsafe_stale_is_active").value)

        allow_reverse = bool(self.get_parameter("allow_reverse").value)
        deadband = float(self.get_parameter("deadband").value)

        slew_l = float(self.get_parameter("slew_left_per_s").value)
        slew_r = float(self.get_parameter("slew_right_per_s").value)

        # Ages
        left_age = now - self._t_left if self._t_left > 0 else 1e9
        right_age = now - self._t_right if self._t_right > 0 else 1e9
        fs_age = now - self._t_failsafe if self._t_failsafe > 0 else 1e9

        input_stale = (left_age > input_timeout) or (right_age > input_timeout)
        fs_stale = fs_age > fs_timeout

        failsafe_active = self._failsafe_active or (fs_stale and fs_stale_is_active)

        # Decide target
        reason = "ok"
        if failsafe_active:
            target_left = 0.0
            target_right = 0.0
            reason = "failsafe_true" if self._failsafe_active else f"failsafe_stale({fs_age:.2f}s)"
        elif input_stale:
            target_left = 0.0
            target_right = 0.0
            reason = f"input_stale(L={left_age:.2f}s R={right_age:.2f}s)"
        else:
            target_left = float(self._last_left_in)
            target_right = float(self._last_right_in)

        # Clamp range
        lo, hi = (-1.0, 1.0) if allow_reverse else (0.0, 1.0)
        target_left = clamp(target_left, lo, hi)
        target_right = clamp(target_right, lo, hi)

        # Deadband
        if abs(target_left) < deadband:
            target_left = 0.0
        if abs(target_right) < deadband:
            target_right = 0.0

        # Slew
        self._left_out = slew_limit(self._left_out, target_left, slew_l, dt)
        self._right_out = slew_limit(self._right_out, target_right, slew_r, dt)

        # Publish
        self.pub_left.publish(Float32(data=float(self._left_out)))
        self.pub_right.publish(Float32(data=float(self._right_out)))

        # Publish reason only when changed (biar tidak spam)
        if reason != self._last_reason:
            self._last_reason = reason
            self.pub_reason.publish(String(data=reason))

        # Periodic log
        log_stats_sec = float(self.get_parameter("log_stats_sec").value)
        if log_stats_sec > 0 and (now - self._last_log) > log_stats_sec:
            self._last_log = now
            self.get_logger().info(
                f"reason={reason} | in_age L={left_age:.2f}s R={right_age:.2f}s fs_age={fs_age:.2f}s | "
                f"out L={self._left_out:.2f} R={self._right_out:.2f}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActuatorSafetyLimiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
