#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO Command Mux (Left/Right interface)

Tujuan:
- Menerima 2 sumber perintah:
  MANUAL: /seano/manual/left_cmd  + /seano/manual/right_cmd
  AUTO  : /seano/auto/left_cmd    + /seano/auto/right_cmd
- Memilih salah satu (default MANUAL) berdasarkan /seano/auto_enable (Bool)
- Mengeluarkan perintah terpilih ke default output aktif:
  OUT   : /seano/selected/left_cmd  + /seano/selected/right_cmd

Catatan arsitektur:
- Node ini adalah pemilih sumber command, bukan final actuator output.
- Output node ini normalnya masuk dulu ke `actuator_safety_limiter_node`.
- Final command ke bridge MAVROS tetap berada pada `/seano/left_cmd` dan `/seano/right_cmd` setelah limiter.

Kenapa penting:
- Teleop, AI avoidance, atau controller lain cukup publish ke topik inputnya.
- Bridge MAVROS tetap menerima satu jalur command akhir yang stabil setelah melewati limiter.
"""

from dataclasses import dataclass
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class CmdPair:
    left: float = 0.0
    right: float = 0.0
    t_left: float = 0.0
    t_right: float = 0.0

    def age(self) -> float:
        # umur terburuk dari dua channel
        now = time.time()
        return max(now - self.t_left, now - self.t_right)


class CommandMuxNode(Node):
    def __init__(self):
        super().__init__("command_mux_node")

        # Topics
        self.declare_parameter("manual_left_topic", "/seano/manual/left_cmd")
        self.declare_parameter("manual_right_topic", "/seano/manual/right_cmd")
        self.declare_parameter("auto_left_topic", "/seano/auto/left_cmd")
        self.declare_parameter("auto_right_topic", "/seano/auto/right_cmd")

        self.declare_parameter("out_left_topic", "/seano/left_cmd")
        self.declare_parameter("out_right_topic", "/seano/right_cmd")

        self.declare_parameter("auto_enable_topic", "/seano/auto_enable")

        # Behavior
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("command_timeout_s", 0.6)  # stale -> failsafe
        self.declare_parameter("fallback_to_manual", True)  # kalau AUTO stale, pakai MANUAL
        self.declare_parameter("allow_reverse", False)  # default untuk USV test
        self.declare_parameter("output_min", 0.0)
        self.declare_parameter("output_max", 1.0)
        self.declare_parameter("log_period_s", 1.5)

        self.manual = CmdPair()
        self.auto = CmdPair()
        self.auto_enable = False

        # Publishers
        out_left = self.get_parameter("out_left_topic").value
        out_right = self.get_parameter("out_right_topic").value
        self.pub_left = self.create_publisher(Float32, out_left, 10)
        self.pub_right = self.create_publisher(Float32, out_right, 10)

        # Subscribers
        self.create_subscription(
            Float32, self.get_parameter("manual_left_topic").value, self._cb_manual_left, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("manual_right_topic").value, self._cb_manual_right, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("auto_left_topic").value, self._cb_auto_left, 10
        )
        self.create_subscription(
            Float32, self.get_parameter("auto_right_topic").value, self._cb_auto_right, 10
        )

        self.create_subscription(
            Bool, self.get_parameter("auto_enable_topic").value, self._cb_auto_enable, 10
        )

        hz = float(self.get_parameter("rate_hz").value)
        if hz <= 0:
            hz = 20.0
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self._tick)

        self._last_log = time.time()

        self.get_logger().info("Command mux ready (MANUAL/AUTO -> OUT left/right).")
        self.get_logger().info(
            "Default: auto_enable=false (MANUAL). Publish Bool true to /seano/auto_enable to switch AUTO."
        )

    def _cb_auto_enable(self, msg: Bool):
        self.auto_enable = bool(msg.data)

    def _cb_manual_left(self, msg: Float32):
        self.manual.left = float(msg.data)
        self.manual.t_left = time.time()

    def _cb_manual_right(self, msg: Float32):
        self.manual.right = float(msg.data)
        self.manual.t_right = time.time()

    def _cb_auto_left(self, msg: Float32):
        self.auto.left = float(msg.data)
        self.auto.t_left = time.time()

    def _cb_auto_right(self, msg: Float32):
        self.auto.right = float(msg.data)
        self.auto.t_right = time.time()

    def _tick(self):
        now = time.time()

        timeout = float(self.get_parameter("command_timeout_s").value)
        fallback = bool(self.get_parameter("fallback_to_manual").value)
        allow_rev = bool(self.get_parameter("allow_reverse").value)
        out_min = float(self.get_parameter("output_min").value)
        out_max = float(self.get_parameter("output_max").value)

        # pilih source
        use_auto = self.auto_enable

        chosen = self.auto if use_auto else self.manual
        chosen_name = "AUTO" if use_auto else "MANUAL"

        # stale handling
        chosen_age = chosen.age()
        manual_age = self.manual.age()

        if chosen_age > timeout:
            if use_auto and fallback and (manual_age <= timeout):
                chosen = self.manual
                chosen_name = "MANUAL_FALLBACK"
            else:
                # failsafe stop
                left = 0.0
                right = 0.0
                self._publish(left, right)
                self._log_periodic(
                    now, chosen_name, chosen_age, manual_age, left, right, failsafe=True
                )
                return

        left = chosen.left
        right = chosen.right

        # clamp output
        if allow_rev:
            left = clamp(left, -1.0, 1.0)
            right = clamp(right, -1.0, 1.0)
            left = clamp(left, out_min, out_max)
            right = clamp(right, out_min, out_max)
        else:
            left = clamp(left, 0.0, 1.0)
            right = clamp(right, 0.0, 1.0)
            left = clamp(left, out_min, out_max)
            right = clamp(right, out_min, out_max)

        self._publish(left, right)
        self._log_periodic(now, chosen_name, chosen_age, manual_age, left, right, failsafe=False)

    def _publish(self, left: float, right: float):
        self.pub_left.publish(Float32(data=float(left)))
        self.pub_right.publish(Float32(data=float(right)))

    def _log_periodic(
        self,
        now: float,
        mode: str,
        age_sel: float,
        age_manual: float,
        left: float,
        right: float,
        failsafe: bool,
    ):
        period = float(self.get_parameter("log_period_s").value)
        if period <= 0:
            return
        if (now - self._last_log) < period:
            return
        self._last_log = now
        tag = "FAILSAFE_STOP" if failsafe else "OK"
        self.get_logger().info(
            f"[{tag}] mode={mode} auto_enable={self.auto_enable} age_sel={age_sel:.2f}s age_manual={age_manual:.2f}s -> left={left:.2f} right={right:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CommandMuxNode()
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
