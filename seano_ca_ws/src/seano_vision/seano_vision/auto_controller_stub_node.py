#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTO Controller (stub) -> Differential Left/Right for SEANO

Input topics (std_msgs/Float32):
  - /seano/auto/desired_speed : 0.0 .. 1.0  (maju)
  - /seano/auto/desired_turn  : -1.0 .. 1.0 (kiri negatif, kanan positif)

Output topics (std_msgs/Float32):
  - /seano/auto/left_cmd
  - /seano/auto/right_cmd

Fitur penting:
- Timeout terpisah: speed dan turn.
  Jika turn timeout -> turn otomatis 0 (kembali lurus), speed tetap.
  Jika speed timeout -> speed=0 (stop).
- diff_mix_gain: left = speed + gain*turn, right = speed - gain*turn
- Bisa auto-enable mode AUTO via /seano/auto_enable (Bool) kalau diaktifkan param.
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class AutoControllerStub(Node):
    def __init__(self):
        super().__init__("auto_controller_stub_node")

        # Topics
        self.declare_parameter("speed_topic", "/seano/auto/desired_speed")
        self.declare_parameter("turn_topic", "/seano/auto/desired_turn")
        self.declare_parameter("out_left_topic", "/seano/auto/left_cmd")
        self.declare_parameter("out_right_topic", "/seano/auto/right_cmd")

        # Optional: toggle AUTO mode
        self.declare_parameter("auto_enable_topic", "/seano/auto_enable")
        self.declare_parameter("auto_enable_on_start", False)
        self.declare_parameter("auto_enable_keepalive_hz", 1.0)

        # Behavior
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("diff_mix_gain", 0.7)

        self.declare_parameter("speed_timeout_s", 0.8)
        self.declare_parameter("turn_timeout_s", 0.25)

        # Safety clamps
        self.declare_parameter("speed_max", 0.60)   # batas aman test
        self.declare_parameter("turn_max", 1.00)
        self.declare_parameter("allow_reverse", False)  # default USV test

        self.declare_parameter("log_period_s", 1.5)

        # State
        self.speed = 0.0
        self.turn = 0.0
        self.t_speed = 0.0
        self.t_turn = 0.0

        # Pub/Sub
        self.pub_left = self.create_publisher(Float32, self.get_parameter("out_left_topic").value, 10)
        self.pub_right = self.create_publisher(Float32, self.get_parameter("out_right_topic").value, 10)

        self.create_subscription(Float32, self.get_parameter("speed_topic").value, self._cb_speed, 10)
        self.create_subscription(Float32, self.get_parameter("turn_topic").value, self._cb_turn, 10)

        self.pub_auto_enable = self.create_publisher(Bool, self.get_parameter("auto_enable_topic").value, 10)

        hz = float(self.get_parameter("rate_hz").value)
        if hz <= 0:
            hz = 20.0
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self._tick)

        keep_hz = float(self.get_parameter("auto_enable_keepalive_hz").value)
        if keep_hz > 0:
            self.create_timer(1.0 / keep_hz, self._auto_enable_keepalive)

        self._last_log = time.time()

        self.get_logger().info("AutoControllerStub ready.")
        self.get_logger().info("Inputs: desired_speed, desired_turn -> Outputs: /seano/auto/left_cmd, /seano/auto/right_cmd")

        if bool(self.get_parameter("auto_enable_on_start").value):
            self.pub_auto_enable.publish(Bool(data=True))
            self.get_logger().info("auto_enable_on_start=true -> requesting AUTO via /seano/auto_enable")

    def _cb_speed(self, msg: Float32):
        self.speed = float(msg.data)
        self.t_speed = time.time()

    def _cb_turn(self, msg: Float32):
        self.turn = float(msg.data)
        self.t_turn = time.time()

    def _auto_enable_keepalive(self):
        if bool(self.get_parameter("auto_enable_on_start").value):
            self.pub_auto_enable.publish(Bool(data=True))

    def _tick(self):
        now = time.time()

        speed_timeout = float(self.get_parameter("speed_timeout_s").value)
        turn_timeout = float(self.get_parameter("turn_timeout_s").value)

        speed_max = float(self.get_parameter("speed_max").value)
        turn_max = float(self.get_parameter("turn_max").value)
        gain = float(self.get_parameter("diff_mix_gain").value)
        allow_reverse = bool(self.get_parameter("allow_reverse").value)

        # apply timeouts
        speed = self.speed
        turn = self.turn

        if (now - self.t_speed) > speed_timeout:
            speed = 0.0
        if (now - self.t_turn) > turn_timeout:
            turn = 0.0

        # clamp input
        if allow_reverse:
            speed = clamp(speed, -1.0, 1.0)
        else:
            speed = clamp(speed, 0.0, 1.0)

        speed = clamp(speed, 0.0 if not allow_reverse else -1.0, speed_max)
        turn = clamp(turn, -turn_max, turn_max)

        # mix to left/right
        left = speed + gain * turn
        right = speed - gain * turn

        if allow_reverse:
            left = clamp(left, -1.0, 1.0)
            right = clamp(right, -1.0, 1.0)
        else:
            left = clamp(left, 0.0, 1.0)
            right = clamp(right, 0.0, 1.0)

        self.pub_left.publish(Float32(data=float(left)))
        self.pub_right.publish(Float32(data=float(right)))

        # periodic log
        period = float(self.get_parameter("log_period_s").value)
        if period > 0 and (now - self._last_log) >= period:
            self._last_log = now
            self.get_logger().info(
                f"speed={speed:.2f} turn={turn:.2f} -> left={left:.2f} right={right:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = AutoControllerStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.pub_left.publish(Float32(data=0.0))
            node.pub_right.publish(Float32(data=0.0))
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()