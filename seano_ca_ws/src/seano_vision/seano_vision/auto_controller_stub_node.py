#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTO Takeover Manager (entrypoint retained as auto_controller_stub_node for active baseline compatibility)

Tujuan (untuk uji autonomous minggu depan):
- Monitor /ca/command (dari risk_evaluator) dan /ca/failsafe_active (dari watchdog)
- Saat bahaya -> takeover: auto_enable=true + rc_override_enable=true
- Saat aman -> release RC override + auto_enable=false (autopilot lanjut mission)

Output ke stack aktuasi (sesuai arsitektur repo):
- /seano/auto/left_cmd, /seano/auto/right_cmd (Float32)
- /seano/auto_enable (Bool) -> command_mux pilih AUTO
- /seano/rc_override_enable (Bool) -> bridge apply/release RC override

Catatan penting:
- Filename `auto_controller_stub_node.py` masih dipertahankan agar kompatibel dengan launch aktif repo.
- Peran runtime node ini sekarang harus dibaca sebagai **auto takeover manager**, bukan stub dummy sederhana.
- Pada FAILSAFE (kamera/perception gagal): node memaksa STOP dengan takeover aktif
  (JANGAN release mission saat perception fail).
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class CmdStamp:
    value: str = ""
    t: float = 0.0

    def age(self) -> float:
        if self.t <= 0.0:
            return 1e9
        return time.time() - self.t


class AutoTakeoverManager(Node):
    def __init__(self) -> None:
        super().__init__("auto_controller_stub_node")  # keep same node name/entrypoint

        # ---------------- Topics ----------------
        self.declare_parameter("command_topic", "/ca/command")
        self.declare_parameter("failsafe_active_topic", "/ca/failsafe_active")

        self.declare_parameter("out_left_topic", "/seano/auto/left_cmd")
        self.declare_parameter("out_right_topic", "/seano/auto/right_cmd")
        self.declare_parameter("auto_enable_topic", "/seano/auto_enable")
        self.declare_parameter("rc_override_enable_topic", "/seano/rc_override_enable")

        # Master enable (biar bisa matikan autonomy tanpa kill node)
        self.declare_parameter("master_enable_topic", "/seano/auto_master_enable")
        self.declare_parameter("master_enable_on_start", False)

        # ---------------- Behavior ----------------
        self.declare_parameter("rate_hz", 20.0)

        # CA command handling
        self.declare_parameter("command_timeout_s", 1.0)  # jika /ca/command stale
        self.declare_parameter(
            "min_takeover_s", 1.0
        )  # minimal durasi takeover sebelum boleh release
        self.declare_parameter("clear_hold_s", 0.8)  # command harus "clear" stabil sebelum release

        # Perintah (harus match risk_evaluator defaults)
        self.declare_parameter("cmd_hold", "HOLD_COURSE")
        self.declare_parameter("cmd_slow", "SLOW_DOWN")
        self.declare_parameter("cmd_turn_left_slow", "TURN_LEFT_SLOW")
        self.declare_parameter("cmd_turn_right_slow", "TURN_RIGHT_SLOW")
        self.declare_parameter("cmd_turn_left", "TURN_LEFT")
        self.declare_parameter("cmd_turn_right", "TURN_RIGHT")
        self.declare_parameter("cmd_stop", "STOP")

        # Control outputs (normalized)
        self.declare_parameter("cruise_speed", 0.30)  # 0..1
        self.declare_parameter("slow_factor", 0.60)  # SLOW_DOWN speed = cruise*slow_factor
        self.declare_parameter(
            "turn_speed_factor", 0.85
        )  # saat TURN_* speed = cruise*turn_speed_factor
        self.declare_parameter("turn_cmd", 0.55)  # -1..1 (kiri negatif, kanan positif)

        # Mixer
        self.declare_parameter(
            "diff_mix_gain", 0.7
        )  # left = speed + gain*turn, right = speed - gain*turn
        self.declare_parameter("allow_reverse", False)  # default USV test 0..1
        self.declare_parameter("speed_max", 0.60)  # batas aman
        self.declare_parameter("turn_max", 1.00)

        # Logging
        self.declare_parameter("log_period_s", 1.0)

        # ---------------- State ----------------
        self.master_enabled = bool(self.get_parameter("master_enable_on_start").value)
        self.failsafe_active = False
        self.cmd = CmdStamp(value=str(self.get_parameter("cmd_hold").value), t=0.0)

        self.state = "PASSIVE"  # PASSIVE | TAKEOVER | FAILSAFE_STOP
        self.t_takeover = 0.0
        self.t_clear_since = 0.0
        self._last_log = 0.0

        # ---------------- Pub/Sub ----------------
        self.pub_left = self.create_publisher(
            Float32, str(self.get_parameter("out_left_topic").value), 10
        )
        self.pub_right = self.create_publisher(
            Float32, str(self.get_parameter("out_right_topic").value), 10
        )
        self.pub_auto_enable = self.create_publisher(
            Bool, str(self.get_parameter("auto_enable_topic").value), 10
        )
        self.pub_rc_override_enable = self.create_publisher(
            Bool, str(self.get_parameter("rc_override_enable_topic").value), 10
        )

        self.create_subscription(
            String, str(self.get_parameter("command_topic").value), self._cb_cmd, 10
        )
        self.create_subscription(
            Bool, str(self.get_parameter("failsafe_active_topic").value), self._cb_failsafe, 10
        )
        self.create_subscription(
            Bool, str(self.get_parameter("master_enable_topic").value), self._cb_master, 10
        )

        hz = float(self.get_parameter("rate_hz").value)
        if hz <= 0.0:
            hz = 20.0
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self._tick)

        self.get_logger().info("AutoTakeoverManager ready.")
        self.get_logger().info(
            f"master_enabled={self.master_enabled} | "
            f"cmd_topic={self.get_parameter('command_topic').value} "
            f"failsafe_topic={self.get_parameter('failsafe_active_topic').value}"
        )
        self.get_logger().info(
            "Publish: /seano/auto_enable + /seano/rc_override_enable + /seano/auto/left_cmd,right_cmd"
        )

        # Publish initial mode
        self._publish_passive()

    # ---------------- Callbacks ----------------
    def _cb_cmd(self, msg: String) -> None:
        self.cmd.value = str(msg.data).strip()
        self.cmd.t = time.time()

    def _cb_failsafe(self, msg: Bool) -> None:
        self.failsafe_active = bool(msg.data)

    def _cb_master(self, msg: Bool) -> None:
        self.master_enabled = bool(msg.data)

    # ---------------- Helpers ----------------
    def _is_cmd_clear(self, cmd: str) -> bool:
        # “clear” berarti aman / tidak perlu takeover
        # risk_evaluator default normalnya HOLD_COURSE
        hold = str(self.get_parameter("cmd_hold").value)
        return cmd == "" or cmd == hold

    def _is_cmd_hazard(self, cmd: str) -> bool:
        # hazard = butuh takeover
        stop = str(self.get_parameter("cmd_stop").value)
        slow = str(self.get_parameter("cmd_slow").value)
        tl = str(self.get_parameter("cmd_turn_left").value)
        tr = str(self.get_parameter("cmd_turn_right").value)
        tls = str(self.get_parameter("cmd_turn_left_slow").value)
        trs = str(self.get_parameter("cmd_turn_right_slow").value)
        return cmd in (stop, slow, tl, tr, tls, trs)

    def _mix_speed_turn_to_lr(self, speed: float, turn: float) -> tuple[float, float]:
        gain = float(self.get_parameter("diff_mix_gain").value)
        allow_reverse = bool(self.get_parameter("allow_reverse").value)

        # clamp input
        speed_max = float(self.get_parameter("speed_max").value)
        turn_max = float(self.get_parameter("turn_max").value)

        if allow_reverse:
            speed = clamp(speed, -1.0, 1.0)
        else:
            speed = clamp(speed, 0.0, 1.0)
        speed = clamp(speed, (-speed_max if allow_reverse else 0.0), speed_max)
        turn = clamp(turn, -turn_max, turn_max)

        left = speed + gain * turn
        right = speed - gain * turn

        if allow_reverse:
            left = clamp(left, -1.0, 1.0)
            right = clamp(right, -1.0, 1.0)
        else:
            left = clamp(left, 0.0, 1.0)
            right = clamp(right, 0.0, 1.0)
        return left, right

    def _publish(
        self, left: float, right: float, auto_enable: bool, rc_override_enable: bool
    ) -> None:
        self.pub_left.publish(Float32(data=float(left)))
        self.pub_right.publish(Float32(data=float(right)))
        self.pub_auto_enable.publish(Bool(data=bool(auto_enable)))
        self.pub_rc_override_enable.publish(Bool(data=bool(rc_override_enable)))

    def _publish_passive(self) -> None:
        # PASSIVE: lepaskan RC override + AUTO off
        self._publish(left=0.0, right=0.0, auto_enable=False, rc_override_enable=False)

    def _publish_stop_takeover(self) -> None:
        # FAILSAFE STOP: takeover ON tapi command 0
        self._publish(left=0.0, right=0.0, auto_enable=True, rc_override_enable=True)

    # ---------------- Main tick ----------------
    def _tick(self) -> None:
        now = time.time()

        # kalau master disable -> selalu PASSIVE (release)
        if not self.master_enabled:
            if self.state != "PASSIVE":
                self.get_logger().warn("MASTER DISABLED -> PASSIVE (RELEASE override)")
            self.state = "PASSIVE"
            self.t_takeover = 0.0
            self.t_clear_since = 0.0
            self._publish_passive()
            self._log(now, cmd=self.cmd.value, left=0.0, right=0.0)
            return

        # FAILSAFE: jangan release mission, paksa STOP takeover
        if self.failsafe_active:
            if self.state != "FAILSAFE_STOP":
                self.get_logger().error("FAILSAFE ACTIVE -> TAKEOVER STOP (do NOT release mission)")
                self.state = "FAILSAFE_STOP"
                self.t_takeover = now
                self.t_clear_since = 0.0
            self._publish_stop_takeover()
            self._log(now, cmd=self.cmd.value, left=0.0, right=0.0)
            return

        # command stale handling
        cmd_timeout = float(self.get_parameter("command_timeout_s").value)
        stale = self.cmd.age() > cmd_timeout
        cmd = self.cmd.value

        if stale:
            # Kalau command stale dan master enabled:
            # - jika sedang TAKEOVER -> STOP takeover (safety)
            # - jika PASSIVE -> tetap PASSIVE
            if self.state in ("TAKEOVER", "FAILSAFE_STOP"):
                self.get_logger().warn(f"CA command stale ({self.cmd.age():.2f}s) -> STOP takeover")
                self.state = "TAKEOVER"
                if self.t_takeover <= 0.0:
                    self.t_takeover = now
                self._publish_stop_takeover()
                self._log(now, cmd="(STALE)->STOP", left=0.0, right=0.0)
                return

            self.state = "PASSIVE"
            self._publish_passive()
            self._log(now, cmd="(STALE)->PASSIVE", left=0.0, right=0.0)
            return

        # Hazard?
        is_clear = self._is_cmd_clear(cmd)
        is_hazard = self._is_cmd_hazard(cmd)

        min_takeover = float(self.get_parameter("min_takeover_s").value)
        clear_hold = float(self.get_parameter("clear_hold_s").value)

        if is_hazard:
            # enter/keep takeover
            if self.state != "TAKEOVER":
                self.get_logger().info(f"TAKEOVER ON (cmd={cmd})")
                self.state = "TAKEOVER"
                self.t_takeover = now
                self.t_clear_since = 0.0

            cruise = float(self.get_parameter("cruise_speed").value)
            slow_factor = float(self.get_parameter("slow_factor").value)
            turn_speed_factor = float(self.get_parameter("turn_speed_factor").value)
            turn_cmd = float(self.get_parameter("turn_cmd").value)

            hold = str(self.get_parameter("cmd_hold").value)
            slow = str(self.get_parameter("cmd_slow").value)
            stop = str(self.get_parameter("cmd_stop").value)
            tl = str(self.get_parameter("cmd_turn_left").value)
            tr = str(self.get_parameter("cmd_turn_right").value)
            tls = str(self.get_parameter("cmd_turn_left_slow").value)
            trs = str(self.get_parameter("cmd_turn_right_slow").value)

            speed = 0.0
            turn = 0.0

            if cmd == stop:
                speed, turn = 0.0, 0.0
            elif cmd == slow or cmd == hold:
                speed, turn = cruise * slow_factor, 0.0
            elif cmd in (tl, tls):
                speed = cruise * (turn_speed_factor if cmd == tl else slow_factor)
                turn = -abs(turn_cmd)
            elif cmd in (tr, trs):
                speed = cruise * (turn_speed_factor if cmd == tr else slow_factor)
                turn = +abs(turn_cmd)
            else:
                # fallback aman
                speed, turn = 0.0, 0.0

            left, right = self._mix_speed_turn_to_lr(speed, turn)

            # takeover: AUTO on + RC override on
            self._publish(left, right, auto_enable=True, rc_override_enable=True)
            self._log(now, cmd=cmd, left=left, right=right)
            return

        # CLEAR path
        if is_clear:
            if self.state == "PASSIVE":
                self._publish_passive()
                self._log(now, cmd=cmd, left=0.0, right=0.0)
                return

            # sedang TAKEOVER, tunggu clear stabil + minimal takeover
            if self.t_clear_since <= 0.0:
                self.t_clear_since = now

            held_clear = (now - self.t_clear_since) >= clear_hold
            held_takeover = (now - self.t_takeover) >= min_takeover if self.t_takeover > 0 else True

            if held_clear and held_takeover:
                self.get_logger().info("TAKEOVER OFF -> RELEASE (return to mission)")
                self.state = "PASSIVE"
                self.t_takeover = 0.0
                self.t_clear_since = 0.0
                self._publish_passive()
                self._log(now, cmd=cmd, left=0.0, right=0.0)
                return

            # selama menunggu release, tetap STOP takeover (lebih aman daripada tetap jalan)
            self._publish_stop_takeover()
            self._log(now, cmd=f"{cmd} (waiting release)", left=0.0, right=0.0)
            return

        # command unknown -> aman: STOP takeover
        if self.state != "TAKEOVER":
            self.state = "TAKEOVER"
            self.t_takeover = now
            self.t_clear_since = 0.0
            self.get_logger().warn(f"Unknown cmd='{cmd}' -> STOP takeover")
        self._publish_stop_takeover()
        self._log(now, cmd=f"UNKNOWN:{cmd}", left=0.0, right=0.0)

    def _log(self, now: float, cmd: str, left: float, right: float) -> None:
        period = float(self.get_parameter("log_period_s").value)
        if period <= 0.0:
            return
        if (now - self._last_log) < period:
            return
        self._last_log = now
        self.get_logger().info(
            f"state={self.state} master={self.master_enabled} failsafe={self.failsafe_active} "
            f"cmd='{cmd}' -> L={left:.2f} R={right:.2f}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutoTakeoverManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            # release override + stop output on exit
            node.pub_left.publish(Float32(data=0.0))
            node.pub_right.publish(Float32(data=0.0))
            node.pub_auto_enable.publish(Bool(data=False))
            node.pub_rc_override_enable.publish(Bool(data=False))
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------------------
# _SEANO_INTERNAL_DELAYED_MASTER_PATCH_V1
# Internal startup guard for AutoTakeoverManager.
#
# Reason:
# - Camera and detector need warm-up time.
# - If master is enabled too early, startup STOP/lost-perception transient
#   forces AVOID before perception is stable.
# - If master stays false forever, /ca/mode_manager_state stays MISSION.
#
# Behavior:
# - Start with master disabled.
# - After master_auto_enable_delay_s, enable master inside this node.
# - No external ros2 param set process is needed.
# ---------------------------------------------------------------------------


def _seano_install_internal_delayed_master_patch():
    import time

    AutoClass = AutoTakeoverManager
    if getattr(AutoClass, "_seano_internal_delayed_master_installed", False):
        return

    _orig_init = AutoClass.__init__

    def _safe_declare(self, name, value):
        try:
            self.declare_parameter(name, value)
        except Exception:
            pass
        try:
            return self.get_parameter(name).value
        except Exception:
            return value

    def _set_master_flag(self, value: bool):
        # Most current code uses self.master_enabled.
        # Some older variants may use self.master_enable.
        try:
            self.master_enabled = bool(value)
        except Exception:
            pass
        try:
            self.master_enable = bool(value)
        except Exception:
            pass

    def _master_guard_tick(self):
        if getattr(self, "_seano_master_guard_done", False):
            return
        if not getattr(self, "_seano_master_guard_enable", True):
            return

        elapsed = time.monotonic() - getattr(self, "_seano_master_guard_t0", time.monotonic())
        delay_s = float(getattr(self, "_seano_master_guard_delay_s", 25.0))

        if elapsed < delay_s:
            return

        self._seano_set_master_flag(True)
        self._seano_master_guard_done = True

        try:
            self.get_logger().info(
                f"[SEANO] startup guard done: master_enabled=True after {elapsed:.1f}s"
            )
        except Exception:
            pass

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)

        self._seano_master_guard_enable = bool(
            _safe_declare(self, "master_auto_enable_after_startup", True)
        )
        self._seano_master_guard_delay_s = float(
            _safe_declare(self, "master_auto_enable_delay_s", 25.0)
        )
        self._seano_master_guard_t0 = time.monotonic()
        self._seano_master_guard_done = False

        if self._seano_master_guard_enable and self._seano_master_guard_delay_s > 0.0:
            self._seano_set_master_flag(False)
            self.create_timer(0.25, self._seano_master_guard_tick)
            try:
                self.get_logger().info(
                    f"[SEANO] startup guard active: master held OFF for "
                    f"{self._seano_master_guard_delay_s:.1f}s"
                )
            except Exception:
                pass

    AutoClass._seano_set_master_flag = _set_master_flag
    AutoClass._seano_master_guard_tick = _master_guard_tick
    AutoClass.__init__ = _patched_init
    AutoClass._seano_internal_delayed_master_installed = True


_seano_install_internal_delayed_master_patch()


if __name__ == "__main__":
    main()
