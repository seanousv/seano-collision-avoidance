#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapter: thr_steer -> AUTO left_right

Input (std_msgs/Float32):
  - /seano/throttle_cmd : 0.0 .. 1.0 (default)
  - /seano/rudder_cmd   : -1.0 .. 1.0

Output (std_msgs/Float32):
  - /seano/auto/left_cmd
  - /seano/auto/right_cmd

Behavior:
- throttle timeout -> throttle=0 (stop)
- steer timeout -> steer=0 (back to straight)
- mix:
    left  = throttle + gain*steer
    right = throttle - gain*steer
- default no reverse: clamp output 0..1
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ThrSteerToAutoLeftRight(Node):
    def __init__(self):
        super().__init__("thrsteer_to_auto_left_right_node")

        # Topics
        self.declare_parameter("thr_topic", "/seano/throttle_cmd")
        self.declare_parameter("steer_topic", "/seano/rudder_cmd")
        self.declare_parameter("out_left_topic", "/seano/auto/left_cmd")
        self.declare_parameter("out_right_topic", "/seano/auto/right_cmd")

        # Optional: request AUTO mode via mux
        self.declare_parameter("auto_enable_topic", "/seano/auto_enable")
        self.declare_parameter("auto_enable_on_start", False)
        self.declare_parameter("auto_enable_keepalive_hz", 1.0)

        # Timing
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("thr_timeout_s", 0.8)
        self.declare_parameter("steer_timeout_s", 0.25)

        # Mix + clamps
        self.declare_parameter("diff_mix_gain", 0.7)
        self.declare_parameter("allow_reverse", False)
        self.declare_parameter("thr_max", 0.60)  # batas aman test
        self.declare_parameter("out_min", 0.0)
        self.declare_parameter("out_max", 1.0)

        self.declare_parameter("log_period_s", 1.5)

        # State
        self.thr = 0.0
        self.steer = 0.0
        self.t_thr = 0.0
        self.t_steer = 0.0
        self._last_log = time.time()

        # Pub/Sub
        self.pub_left = self.create_publisher(
            Float32, self.get_parameter("out_left_topic").value, 10
        )
        self.pub_right = self.create_publisher(
            Float32, self.get_parameter("out_right_topic").value, 10
        )
        self.pub_auto_enable = self.create_publisher(
            Bool, self.get_parameter("auto_enable_topic").value, 10
        )

        self.create_subscription(Float32, self.get_parameter("thr_topic").value, self._cb_thr, 10)
        self.create_subscription(
            Float32, self.get_parameter("steer_topic").value, self._cb_steer, 10
        )

        hz = float(self.get_parameter("rate_hz").value)
        if hz <= 0:
            hz = 20.0
        self.create_timer(1.0 / hz, self._tick)

        keep_hz = float(self.get_parameter("auto_enable_keepalive_hz").value)
        if keep_hz > 0:
            self.create_timer(1.0 / keep_hz, self._auto_enable_keepalive)

        if bool(self.get_parameter("auto_enable_on_start").value):
            self.pub_auto_enable.publish(Bool(data=True))

        self.get_logger().info("thr_steer -> AUTO left/right adapter ready.")

    def _cb_thr(self, msg: Float32):
        self.thr = float(msg.data)
        self.t_thr = time.time()

    def _cb_steer(self, msg: Float32):
        self.steer = float(msg.data)
        self.t_steer = time.time()

    def _auto_enable_keepalive(self):
        if bool(self.get_parameter("auto_enable_on_start").value):
            self.pub_auto_enable.publish(Bool(data=True))

    def _tick(self):
        now = time.time()

        thr_timeout = float(self.get_parameter("thr_timeout_s").value)
        steer_timeout = float(self.get_parameter("steer_timeout_s").value)

        gain = float(self.get_parameter("diff_mix_gain").value)
        allow_rev = bool(self.get_parameter("allow_reverse").value)
        thr_max = float(self.get_parameter("thr_max").value)
        out_min = float(self.get_parameter("out_min").value)
        out_max = float(self.get_parameter("out_max").value)

        thr = self.thr
        steer = self.steer

        # timeouts
        if (now - self.t_thr) > thr_timeout:
            thr = 0.0
        if (now - self.t_steer) > steer_timeout:
            steer = 0.0

        # clamp input
        if allow_rev:
            thr = clamp(thr, -1.0, 1.0)
        else:
            thr = clamp(thr, 0.0, 1.0)

        thr = clamp(thr, 0.0 if not allow_rev else -1.0, thr_max)
        steer = clamp(steer, -1.0, 1.0)

        # mix
        left = thr + gain * steer
        right = thr - gain * steer

        if allow_rev:
            left = clamp(left, -1.0, 1.0)
            right = clamp(right, -1.0, 1.0)
        else:
            left = clamp(left, 0.0, 1.0)
            right = clamp(right, 0.0, 1.0)

        left = clamp(left, out_min, out_max)
        right = clamp(right, out_min, out_max)

        self.pub_left.publish(Float32(data=float(left)))
        self.pub_right.publish(Float32(data=float(right)))

        # periodic log
        period = float(self.get_parameter("log_period_s").value)
        if period > 0 and (now - self._last_log) >= period:
            self._last_log = now
            self.get_logger().info(
                f"thr={thr:.2f} steer={steer:.2f} -> auto_left={left:.2f} auto_right={right:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ThrSteerToAutoLeftRight()
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
