#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bridge output Collision Avoidance -> MAVROS RC override (ArduRover skid/differential thrust).

Input:
  - /seano/throttle_cmd : std_msgs/Float32 (0.0 .. 1.0)  -> forward throttle demand
  - /seano/rudder_cmd   : std_msgs/Float32 (-1.0 .. 1.0) -> steering/yaw demand (virtual steer)

Output:
  - /mavros/rc/override  : mavros_msgs/OverrideRCIn

Notes penting (biar tidak bingung):
- RC override biasanya efektif untuk menggerakkan rover saat mode MANUAL/STEERING/ACRO.
- Saat GUIDED, autopilot bisa “mengabaikan” RC override karena dia kontrol sendiri.
  Jadi untuk ngetes node ini: pakai mode MANUAL atau STEERING.
"""

from typing import List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32
from mavros_msgs.msg import OverrideRCIn


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class MavrosRcOverrideBridge(Node):
    def __init__(self):
        super().__init__("mavros_rc_override_bridge_node")

        # Topics
        self.declare_parameter("thr_topic", "/seano/throttle_cmd")
        self.declare_parameter("steer_topic", "/seano/rudder_cmd")
        self.declare_parameter("out_topic", "/mavros/rc/override")

        # RC channel mapping (1-based)
        self.declare_parameter("rc_steer_chan", 1)     # CH1 (steering input)
        self.declare_parameter("rc_throttle_chan", 3)  # CH3 (throttle input)

        # PWM calibration
        self.declare_parameter("pwm_neutral", 1500)
        self.declare_parameter("pwm_throttle_fwd_max", 1900)  # forward max
        self.declare_parameter("pwm_throttle_rev_min", 1100)  # reverse min (optional)
        self.declare_parameter("allow_reverse", False)

        self.declare_parameter("pwm_steer_left", 1100)
        self.declare_parameter("pwm_steer_right", 1900)

        # Scaling + deadband
        self.declare_parameter("thr_scale", 1.0)         # throttle scale
        self.declare_parameter("steer_scale", 1.0)       # steer scale
        self.declare_parameter("thr_deadband", 0.02)     # <= ini dianggap 0
        self.declare_parameter("steer_deadband", 0.05)

        # Safety behavior
        self.declare_parameter("enable", True)
        self.declare_parameter("command_timeout_s", 0.5)  # jika input stop > ini -> netral
        self.declare_parameter("pub_hz", 20.0)
        self.declare_parameter("log_period_s", 2.0)

        # Test mode (untuk validasi cepat)
        self.declare_parameter("test_enable", False)
        self.declare_parameter("test_throttle", 0.20)  # 0..1
        self.declare_parameter("test_steer", 0.0)      # -1..1

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        thr_topic = self.get_parameter("thr_topic").value
        steer_topic = self.get_parameter("steer_topic").value
        out_topic = self.get_parameter("out_topic").value

        self._thr_cmd = 0.0     # 0..1 (atau -1..1 jika reverse diizinkan)
        self._steer_cmd = 0.0   # -1..1
        self._last_cmd_time = self.get_clock().now()
        self._last_log_time = self.get_clock().now()

        self.create_subscription(Float32, thr_topic, self._on_thr, qos)
        self.create_subscription(Float32, steer_topic, self._on_steer, qos)
        self.pub = self.create_publisher(OverrideRCIn, out_topic, qos)

        hz = float(self.get_parameter("pub_hz").value)
        if hz <= 0.0:
            hz = 20.0
        self.create_timer(1.0 / hz, self._tick)

        self.get_logger().info(
            f"RC override bridge ready | in: {thr_topic}, {steer_topic} -> out: {out_topic}"
        )
        self.get_logger().info("Untuk ngetes gerak: set mode MANUAL/STEERING, ARM, lalu pakai test_enable.")

    def _on_thr(self, msg: Float32):
        if not bool(self.get_parameter("enable").value):
            return

        thr_scale = float(self.get_parameter("thr_scale").value)
        thr_dead = float(self.get_parameter("thr_deadband").value)
        allow_rev = bool(self.get_parameter("allow_reverse").value)

        thr = float(msg.data) * thr_scale

        if allow_rev:
            thr = clamp(thr, -1.0, 1.0)
        else:
            thr = clamp(thr, 0.0, 1.0)

        if abs(thr) < thr_dead:
            thr = 0.0

        self._thr_cmd = thr
        self._last_cmd_time = self.get_clock().now()

    def _on_steer(self, msg: Float32):
        if not bool(self.get_parameter("enable").value):
            return

        steer_scale = float(self.get_parameter("steer_scale").value)
        steer_dead = float(self.get_parameter("steer_deadband").value)

        steer = float(msg.data) * steer_scale
        steer = clamp(steer, -1.0, 1.0)

        if abs(steer) < steer_dead:
            steer = 0.0

        self._steer_cmd = steer
        self._last_cmd_time = self.get_clock().now()

    def _build_override(self, pwm_steer: int, pwm_thr: int) -> OverrideRCIn:
        msg = OverrideRCIn()
        channels: List[int] = [0] * 18

        ch_steer = int(self.get_parameter("rc_steer_chan").value) - 1
        ch_thr = int(self.get_parameter("rc_throttle_chan").value) - 1

        if 0 <= ch_steer < 18:
            channels[ch_steer] = int(pwm_steer)
        if 0 <= ch_thr < 18:
            channels[ch_thr] = int(pwm_thr)

        msg.channels = channels
        return msg

    def _tick(self):
        if not bool(self.get_parameter("enable").value):
            # kalau disable, kirim netral biar aman
            pwm_neu = int(self.get_parameter("pwm_neutral").value)
            self.pub.publish(self._build_override(pwm_neu, pwm_neu))
            return

        now = self.get_clock().now()

        # Test mode override
        test_enable = bool(self.get_parameter("test_enable").value)
        if test_enable:
            thr = float(self.get_parameter("test_throttle").value)
            steer = float(self.get_parameter("test_steer").value)
            thr = clamp(thr, 0.0, 1.0)
            steer = clamp(steer, -1.0, 1.0)
        else:
            thr = self._thr_cmd
            steer = self._steer_cmd

        # Timeout failsafe (kalau input mati)
        timeout_s = float(self.get_parameter("command_timeout_s").value)
        dt = (now - self._last_cmd_time).nanoseconds / 1e9
        if (not test_enable) and (dt > timeout_s):
            thr = 0.0
            steer = 0.0

        pwm_neu = int(self.get_parameter("pwm_neutral").value)

        pwm_fwd = int(self.get_parameter("pwm_throttle_fwd_max").value)
        pwm_rev = int(self.get_parameter("pwm_throttle_rev_min").value)
        allow_rev = bool(self.get_parameter("allow_reverse").value)

        pwm_left = int(self.get_parameter("pwm_steer_left").value)
        pwm_right = int(self.get_parameter("pwm_steer_right").value)

        # Throttle mapping
        if allow_rev and thr < 0.0:
            # -1..0 -> rev_min..neutral
            pwm_thr = int(pwm_neu + thr * (pwm_neu - pwm_rev))
            pwm_thr = clamp(pwm_thr, pwm_rev, pwm_neu)
        else:
            # 0..1 -> neutral..fwd_max
            thr = clamp(thr, 0.0, 1.0)
            pwm_thr = int(pwm_neu + thr * (pwm_fwd - pwm_neu))
            pwm_thr = clamp(pwm_thr, pwm_neu, pwm_fwd)

        # Steer mapping
        if steer >= 0.0:
            pwm_steer = int(pwm_neu + steer * (pwm_right - pwm_neu))
        else:
            pwm_steer = int(pwm_neu + (-steer) * (pwm_neu - pwm_left))
        pwm_steer = clamp(pwm_steer, pwm_left, pwm_right)

        self.pub.publish(self._build_override(int(pwm_steer), int(pwm_thr)))

        # periodic log
        log_period = float(self.get_parameter("log_period_s").value)
        if log_period > 0:
            since = (now - self._last_log_time).nanoseconds / 1e9
            if since >= log_period:
                self._last_log_time = now
                self.get_logger().info(
                    f"cmd thr={thr:.2f} steer={steer:.2f} -> PWM steer={int(pwm_steer)} thr={int(pwm_thr)}"
                )


def main(args=None):
    rclpy.init(args=args)
    node = MavrosRcOverrideBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
