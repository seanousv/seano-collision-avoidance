#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAVROS RC Override Bridge untuk SEANO (USV differential thruster) + SITL ArduRover rover-skid.

INPUT MODE (parameter: input_mode)
1) "thr_steer" (default)
   - /seano/throttle_cmd : std_msgs/Float32 (0..1 atau -1..1 jika allow_reverse true)
   - /seano/rudder_cmd   : std_msgs/Float32 (-1..1)
2) "left_right"
   - /seano/left_cmd  : std_msgs/Float32 (0..1 atau -1..1 jika allow_reverse true)
   - /seano/right_cmd : std_msgs/Float32 (0..1 atau -1..1 jika allow_reverse true)
3) "twist"
   - /cmd_vel : geometry_msgs/Twist
     * linear.x  -> forward velocity (dinormalisasi via twist_v_max)
     * angular.z -> yaw rate         (dinormalisasi via twist_yaw_max)

OUTPUT MODE (parameter: output_mode)
A) "rc_thr_steer" (default, REKOMENDASI untuk SITL rover-skid)
   - publish RC override ke channel steer + throttle (mis. CH1 steer, CH3 throttle)
   - ArduRover mixing sendiri untuk rover-skid
B) "rc_left_right"
   - publish RC override ke channel left + right (untuk hardware USV jika mapping langsung)

BARU (untuk uji autonomous + return-to-mission):
- /seano/rc_override_enable (std_msgs/Bool)
  * true  -> bridge mengirim PWM override seperti biasa
  * false -> bridge mengirim OverrideRCIn CHAN_RELEASE (0) untuk melepas override
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from geometry_msgs.msg import Twist
from mavros_msgs.msg import OverrideRCIn
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32


def clampf(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clampi(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def sign(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


class MavrosRcOverrideBridge(Node):
    def __init__(self) -> None:
        super().__init__("mavros_rc_override_bridge_node")

        # ========= Topics (commands) =========
        self.declare_parameter("thr_topic", "/seano/throttle_cmd")
        self.declare_parameter("steer_topic", "/seano/rudder_cmd")
        self.declare_parameter("left_topic", "/seano/left_cmd")
        self.declare_parameter("right_topic", "/seano/right_cmd")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("out_topic", "/mavros/rc/override")

        # ========= NEW: override enable =========
        self.declare_parameter("override_enable_topic", "/seano/rc_override_enable")
        self.declare_parameter("override_enabled_default", True)
        self.declare_parameter("publish_release_when_disabled", True)

        # ========= Mode =========
        self.declare_parameter("input_mode", "thr_steer")  # thr_steer | left_right | twist
        self.declare_parameter("output_mode", "rc_thr_steer")  # rc_thr_steer | rc_left_right

        # ========= RC channel mapping (1-based) =========
        # output_mode = rc_thr_steer:
        self.declare_parameter("rc_steer_chan", 1)  # CH1
        self.declare_parameter("rc_throttle_chan", 3)  # CH3
        # output_mode = rc_left_right:
        self.declare_parameter("rc_left_chan", 1)
        self.declare_parameter("rc_right_chan", 3)

        # ========= PWM calibration =========
        self.declare_parameter("pwm_neutral", 1500)
        self.declare_parameter("pwm_fwd_max", 1900)
        self.declare_parameter("pwm_rev_min", 1100)
        self.declare_parameter("allow_reverse", False)

        # steering endpoints (untuk rc_thr_steer)
        self.declare_parameter("pwm_steer_left", 1100)
        self.declare_parameter("pwm_steer_right", 1900)

        # global clamp safety
        self.declare_parameter("pwm_output_min", 1000)
        self.declare_parameter("pwm_output_max", 2000)

        # ========= Scaling + deadband =========
        self.declare_parameter("thr_scale", 1.0)
        self.declare_parameter("steer_scale", 1.0)
        self.declare_parameter("left_scale", 1.0)
        self.declare_parameter("right_scale", 1.0)
        self.declare_parameter("thr_deadband", 0.02)
        self.declare_parameter("steer_deadband", 0.05)
        self.declare_parameter("lr_deadband", 0.02)

        # ========= Differential mixer =========
        # Untuk output_mode=rc_left_right dan input_mode=thr_steer/twist:
        # left = thr + steer * diff_mix_gain
        # right = thr - steer * diff_mix_gain
        self.declare_parameter("diff_mix_gain", 1.0)

        # Untuk input_mode=left_right dan output_mode=rc_thr_steer (SITL rover-skid),
        # konversi balik:
        # thr = (L+R)/2
        # steer = (L-R)/2 * lr_to_steer_gain
        self.declare_parameter("lr_to_steer_gain", 1.0)

        # ========= Twist normalization =========
        self.declare_parameter("twist_v_max", 1.0)
        self.declare_parameter("twist_yaw_max", 1.0)

        # ========= Safety behavior =========
        self.declare_parameter("enable", True)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("pub_hz", 20.0)
        self.declare_parameter("log_period_s", 1.0)

        # slew-rate limiter (PWM microseconds per second). 0 = off
        self.declare_parameter("pwm_slew_rate_us_per_s", 0.0)

        # ========= Test mode =========
        self.declare_parameter("test_enable", False)
        self.declare_parameter("test_throttle", 0.20)  # 0..1
        self.declare_parameter("test_steer", 0.0)  # -1..1
        self.declare_parameter("test_left", 0.20)
        self.declare_parameter("test_right", 0.20)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        thr_topic = str(self.get_parameter("thr_topic").value)
        steer_topic = str(self.get_parameter("steer_topic").value)
        left_topic = str(self.get_parameter("left_topic").value)
        right_topic = str(self.get_parameter("right_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        out_topic = str(self.get_parameter("out_topic").value)

        self._thr_cmd: float = 0.0
        self._steer_cmd: float = 0.0
        self._left_cmd: float = 0.0
        self._right_cmd: float = 0.0
        self._twist_last: Optional[Twist] = None

        self._last_cmd_time = self.get_clock().now()
        self._last_log_time = self.get_clock().now()

        self._last_pwm_steer = int(self.get_parameter("pwm_neutral").value)
        self._last_pwm_thr = int(self.get_parameter("pwm_neutral").value)
        self._last_pwm_left = int(self.get_parameter("pwm_neutral").value)
        self._last_pwm_right = int(self.get_parameter("pwm_neutral").value)

        # NEW: override enable state
        self._override_enabled = bool(self.get_parameter("override_enabled_default").value)
        override_enable_topic = str(self.get_parameter("override_enable_topic").value)
        self.create_subscription(Bool, override_enable_topic, self._on_override_enable, qos)

        # Command subscriptions
        self.create_subscription(Float32, thr_topic, self._on_thr, qos)
        self.create_subscription(Float32, steer_topic, self._on_steer, qos)
        self.create_subscription(Float32, left_topic, self._on_left, qos)
        self.create_subscription(Float32, right_topic, self._on_right, qos)
        self.create_subscription(Twist, cmd_vel_topic, self._on_twist, qos)

        self.pub = self.create_publisher(OverrideRCIn, out_topic, qos)

        hz = float(self.get_parameter("pub_hz").value)
        if hz <= 0.0:
            hz = 20.0
        self._pub_hz = hz
        self.create_timer(1.0 / hz, self._tick)

        self.get_logger().info(
            f"RC override bridge ready | out: {out_topic} "
            f"| input_mode={self.get_parameter('input_mode').value} "
            f"| output_mode={self.get_parameter('output_mode').value}"
        )
        self.get_logger().info(
            "SITL rover-skid: rekomendasi output_mode=rc_thr_steer (CH1 steer, CH3 throttle)."
        )
        self.get_logger().info(
            f"Override enable topic: {override_enable_topic} " f"(default={self._override_enabled})"
        )

    # ===================== OVERRIDE ENABLE =====================
    def _on_override_enable(self, msg: Bool) -> None:
        self._override_enabled = bool(msg.data)

    # ===================== INPUT CALLBACKS =====================
    def _touch_cmd(self) -> None:
        self._last_cmd_time = self.get_clock().now()

    def _on_thr(self, msg: Float32) -> None:
        if not bool(self.get_parameter("enable").value):
            return
        thr_scale = float(self.get_parameter("thr_scale").value)
        thr_dead = float(self.get_parameter("thr_deadband").value)
        allow_rev = bool(self.get_parameter("allow_reverse").value)

        thr = float(msg.data) * thr_scale
        thr = clampf(thr, -1.0, 1.0) if allow_rev else clampf(thr, 0.0, 1.0)
        if abs(thr) < thr_dead:
            thr = 0.0
        self._thr_cmd = thr
        self._touch_cmd()

    def _on_steer(self, msg: Float32) -> None:
        if not bool(self.get_parameter("enable").value):
            return
        steer_scale = float(self.get_parameter("steer_scale").value)
        steer_dead = float(self.get_parameter("steer_deadband").value)

        steer = float(msg.data) * steer_scale
        steer = clampf(steer, -1.0, 1.0)
        if abs(steer) < steer_dead:
            steer = 0.0
        self._steer_cmd = steer
        self._touch_cmd()

    def _on_left(self, msg: Float32) -> None:
        if not bool(self.get_parameter("enable").value):
            return
        scale = float(self.get_parameter("left_scale").value)
        dead = float(self.get_parameter("lr_deadband").value)
        allow_rev = bool(self.get_parameter("allow_reverse").value)

        v = float(msg.data) * scale
        v = clampf(v, -1.0, 1.0) if allow_rev else clampf(v, 0.0, 1.0)
        if abs(v) < dead:
            v = 0.0
        self._left_cmd = v
        self._touch_cmd()

    def _on_right(self, msg: Float32) -> None:
        if not bool(self.get_parameter("enable").value):
            return
        scale = float(self.get_parameter("right_scale").value)
        dead = float(self.get_parameter("lr_deadband").value)
        allow_rev = bool(self.get_parameter("allow_reverse").value)

        v = float(msg.data) * scale
        v = clampf(v, -1.0, 1.0) if allow_rev else clampf(v, 0.0, 1.0)
        if abs(v) < dead:
            v = 0.0
        self._right_cmd = v
        self._touch_cmd()

    def _on_twist(self, msg: Twist) -> None:
        if not bool(self.get_parameter("enable").value):
            return
        self._twist_last = msg
        self._touch_cmd()

    # ===================== PWM HELPERS =====================
    def _norm_to_pwm(self, x: float) -> int:
        pwm_neu = int(self.get_parameter("pwm_neutral").value)
        pwm_fwd = int(self.get_parameter("pwm_fwd_max").value)
        pwm_rev = int(self.get_parameter("pwm_rev_min").value)
        allow_rev = bool(self.get_parameter("allow_reverse").value)

        if allow_rev:
            x = clampf(x, -1.0, 1.0)
            if x >= 0.0:
                pwm = int(pwm_neu + x * (pwm_fwd - pwm_neu))
            else:
                pwm = int(pwm_neu + x * (pwm_neu - pwm_rev))
        else:
            x = clampf(x, 0.0, 1.0)
            pwm = int(pwm_neu + x * (pwm_fwd - pwm_neu))
        return pwm

    def _steer_to_pwm(self, steer: float) -> int:
        pwm_neu = int(self.get_parameter("pwm_neutral").value)
        pwm_left = int(self.get_parameter("pwm_steer_left").value)
        pwm_right = int(self.get_parameter("pwm_steer_right").value)

        steer = clampf(steer, -1.0, 1.0)
        if steer >= 0.0:
            pwm = int(pwm_neu + steer * (pwm_right - pwm_neu))
        else:
            pwm = int(pwm_neu + (-steer) * (pwm_neu - pwm_left))
        return pwm

    def _apply_slew(self, target_pwm: int, last_pwm: int) -> int:
        rate = float(self.get_parameter("pwm_slew_rate_us_per_s").value)
        if rate <= 0.0:
            return target_pwm
        max_step = int(rate / self._pub_hz)
        if max_step <= 0:
            return target_pwm
        delta = target_pwm - last_pwm
        if abs(delta) <= max_step:
            return target_pwm
        return last_pwm + int(sign(delta) * max_step)

    def _global_pwm_clamp(self, pwm: int) -> int:
        lo = int(self.get_parameter("pwm_output_min").value)
        hi = int(self.get_parameter("pwm_output_max").value)
        return clampi(pwm, lo, hi)

    def _build_override(self, ch_pwm_pairs: List[Tuple[int, int]]) -> OverrideRCIn:
        msg = OverrideRCIn()
        channels: List[int] = [0] * 18  # CHAN_RELEASE
        for ch_1b, pwm in ch_pwm_pairs:
            idx = int(ch_1b) - 1
            if 0 <= idx < 18:
                channels[idx] = int(pwm)
        msg.channels = channels
        return msg

    def _publish_release(self) -> None:
        # Semua channel = 0 -> CHAN_RELEASE (melepas override)
        self.pub.publish(self._build_override([]))

    # ===================== MAIN TICK =====================
    def _tick(self) -> None:
        now = self.get_clock().now()

        publish_release_when_disabled = bool(
            self.get_parameter("publish_release_when_disabled").value
        )

        # Jika node di-disable, lebih aman kita release override (agar tidak nyangkut)
        if not bool(self.get_parameter("enable").value):
            if publish_release_when_disabled:
                self._publish_release()
            self._log_periodic(now, "DISABLED", "RC_OVERRIDE_RELEASE")
            return

        # Jika override diminta OFF, kirim release terus (hold) supaya FCU benar-benar balik ke mission/manual.
        if not self._override_enabled:
            if publish_release_when_disabled:
                self._publish_release()
            self._log_periodic(now, "OVERRIDE_OFF", "RC_OVERRIDE_RELEASE")
            return

        # ===== normal override path =====
        test_enable = bool(self.get_parameter("test_enable").value)
        timeout_s = float(self.get_parameter("command_timeout_s").value)
        dt = (now - self._last_cmd_time).nanoseconds / 1e9
        timed_out = (not test_enable) and (dt > timeout_s)

        in_mode = str(self.get_parameter("input_mode").value).strip()
        out_mode = str(self.get_parameter("output_mode").value).strip()
        allow_rev = bool(self.get_parameter("allow_reverse").value)

        # normalized commands
        thr = 0.0
        steer = 0.0
        left = 0.0
        right = 0.0

        if timed_out:
            thr = steer = left = right = 0.0
        elif test_enable:
            if in_mode == "left_right":
                left = float(self.get_parameter("test_left").value)
                right = float(self.get_parameter("test_right").value)
            else:
                thr = float(self.get_parameter("test_throttle").value)
                steer = float(self.get_parameter("test_steer").value)
        else:
            if in_mode == "left_right":
                left = self._left_cmd
                right = self._right_cmd
            elif in_mode == "twist":
                if self._twist_last is not None:
                    v_max = float(self.get_parameter("twist_v_max").value) or 1.0
                    yaw_max = float(self.get_parameter("twist_yaw_max").value) or 1.0
                    thr = clampf(self._twist_last.linear.x / v_max, -1.0, 1.0)
                    steer = clampf(self._twist_last.angular.z / yaw_max, -1.0, 1.0)
            else:
                thr = self._thr_cmd
                steer = self._steer_cmd

        # clamp
        if in_mode == "left_right":
            left = clampf(left, -1.0, 1.0) if allow_rev else clampf(left, 0.0, 1.0)
            right = clampf(right, -1.0, 1.0) if allow_rev else clampf(right, 0.0, 1.0)
        else:
            thr = clampf(thr, -1.0, 1.0) if allow_rev else clampf(thr, 0.0, 1.0)
            steer = clampf(steer, -1.0, 1.0)

        # output
        if out_mode == "rc_left_right":
            # mix jika input bukan left_right
            if in_mode != "left_right":
                gain = float(self.get_parameter("diff_mix_gain").value)
                left = thr + steer * gain
                right = thr - steer * gain
                left = clampf(left, -1.0, 1.0) if allow_rev else clampf(left, 0.0, 1.0)
                right = clampf(right, -1.0, 1.0) if allow_rev else clampf(right, 0.0, 1.0)

            pwm_left = self._global_pwm_clamp(self._norm_to_pwm(left))
            pwm_right = self._global_pwm_clamp(self._norm_to_pwm(right))
            pwm_left = self._apply_slew(pwm_left, self._last_pwm_left)
            pwm_right = self._apply_slew(pwm_right, self._last_pwm_right)
            self._last_pwm_left = pwm_left
            self._last_pwm_right = pwm_right

            ch_l = int(self.get_parameter("rc_left_chan").value)
            ch_r = int(self.get_parameter("rc_right_chan").value)
            self.pub.publish(self._build_override([(ch_l, pwm_left), (ch_r, pwm_right)]))

            extra = "TIMEOUT->neutral" if timed_out else "OK"
            self._log_periodic(
                now,
                f"{extra} L={left:.2f} R={right:.2f}",
                f"PWM L={pwm_left} R={pwm_right}",
            )
            return

        # rc_thr_steer
        if in_mode == "left_right":
            gain = float(self.get_parameter("lr_to_steer_gain").value)
            thr = 0.5 * (left + right)
            steer = 0.5 * (left - right) * gain
            thr = clampf(thr, -1.0, 1.0) if allow_rev else clampf(thr, 0.0, 1.0)
            steer = clampf(steer, -1.0, 1.0)

        pwm_thr = self._global_pwm_clamp(self._norm_to_pwm(thr))
        pwm_steer = self._global_pwm_clamp(self._steer_to_pwm(steer))
        pwm_thr = self._apply_slew(pwm_thr, self._last_pwm_thr)
        pwm_steer = self._apply_slew(pwm_steer, self._last_pwm_steer)
        self._last_pwm_thr = pwm_thr
        self._last_pwm_steer = pwm_steer

        ch_s = int(self.get_parameter("rc_steer_chan").value)
        ch_t = int(self.get_parameter("rc_throttle_chan").value)
        self.pub.publish(self._build_override([(ch_s, pwm_steer), (ch_t, pwm_thr)]))

        extra = "TIMEOUT->neutral" if timed_out else "OK"
        self._log_periodic(
            now,
            f"{extra} thr={thr:.2f} steer={steer:.2f}",
            f"PWM steer={pwm_steer} thr={pwm_thr}",
        )

    def _log_periodic(self, now, a: str, b: str) -> None:
        period = float(self.get_parameter("log_period_s").value)
        if period <= 0.0:
            return
        since = (now - self._last_log_time).nanoseconds / 1e9
        if since < period:
            return
        self._last_log_time = now
        self.get_logger().info(f"{a} | {b}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MavrosRcOverrideBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # on exit: release override (avoid stuck)
        try:
            if bool(node.get_parameter("publish_release_when_disabled").value):
                node._publish_release()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
