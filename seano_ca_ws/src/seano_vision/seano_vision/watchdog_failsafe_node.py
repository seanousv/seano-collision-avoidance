#!/usr/bin/env python3
"""
watchdog_failsafe_node.py (ROS 2 Humble) - SEANO Vision

FIX utama:
- Watchdog bisa memantau beberapa topic Image sekaligus (fallback), bukan cuma satu.
  Jadi kalau kamu pakai /camera/image_raw (bukan _reliable), atau pipeline cuma publish debug_image,
  watchdog tetap "lihat" image dan tidak false LOST.

UPDATE (sesuai request kamu):
- Default image_timeout_s dinaikkan jadi 4.5 detik (range aman 4–5 detik).
  Tujuannya biar jitter/lag RTSP WiFi yang kadang drop 2–3 detik tidak langsung bikin LOST -> STOP.

Parameter penting:
- image_topics (string array): daftar topic Image yang dianggap valid.
  Default: ["/camera/image_raw_reliable", "/camera/image_raw", "/ca/debug_image"]
- image_timeout_s (float): stale threshold image. Default: 4.5
"""

from __future__ import annotations

import json
import time
from typing import Optional, List, Tuple, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String, Bool


def _qos(depth: int = 1, reliability: str = "best_effort") -> QoSProfile:
    rel = str(reliability).strip().lower()
    rel_policy = ReliabilityPolicy.BEST_EFFORT
    if rel in ("reliable", "rel", "r"):
        rel_policy = ReliabilityPolicy.RELIABLE

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=rel_policy,
        durability=DurabilityPolicy.VOLATILE,
    )


def _now_s() -> float:
    return time.monotonic()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_mode(s: str) -> str:
    s2 = str(s or "").strip().upper()
    s2 = s2.replace("-", "_").replace(" ", "_")
    return s2


def _is_lost_mode(mode: str) -> bool:
    m = _norm_mode(mode)
    return ("LOST" in m)


def _is_caution_mode(mode: str) -> bool:
    m = _norm_mode(mode)
    return ("CAUTION" in m)


class WatchdogFailsafeNode(Node):
    def __init__(self) -> None:
        super().__init__("watchdog_failsafe_node")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("qos_depth", 1)
        self.declare_parameter("sub_reliability", "best_effort")

        # NEW: multi image topics
        self.declare_parameter(
            "image_topics",
            ["/camera/image_raw_reliable", "/camera/image_raw", "/ca/debug_image"],
        )

        # Other inputs
        self.declare_parameter("risk_topic", "/ca/risk")
        self.declare_parameter("mode_topic", "/ca/mode")
        self.declare_parameter("command_in_topic", "/ca/command")
        self.declare_parameter("vision_quality_topic", "/vision/quality")
        self.declare_parameter("freeze_topic", "/vision/freeze")
        self.declare_parameter("freeze_reason_topic", "/vision/freeze_reason")

        # Outputs
        self.declare_parameter("command_out_topic", "/ca/command_safe")
        self.declare_parameter("failsafe_active_topic", "/ca/failsafe_active")
        self.declare_parameter("failsafe_reason_topic", "/ca/failsafe_reason")
        self.declare_parameter("status_topic", "/ca/watchdog_status")

        # Timeouts
        # >>> REQUESTED CHANGE: 2.0 -> 4.5 seconds
        self.declare_parameter("image_timeout_s", 4.5)
        self.declare_parameter("risk_timeout_s", 2.0)
        self.declare_parameter("mode_timeout_s", 2.0)
        self.declare_parameter("command_timeout_s", 2.0)
        self.declare_parameter("vision_quality_timeout_s", 1.5)
        self.declare_parameter("freeze_timeout_s", 1.5)

        # Tick & startup grace
        self.declare_parameter("tick_hz", 10.0)
        self.declare_parameter("start_in_failsafe", True)
        self.declare_parameter("startup_grace_s", 2.0)

        # VQ caution hysteresis
        self.declare_parameter("vq_caution_enter", 0.35)
        self.declare_parameter("vq_caution_exit", 0.55)

        # Recovery rules from LOST
        self.declare_parameter("lost_min_hold_s", 1.0)
        self.declare_parameter("recover_ok_hold_s", 0.6)
        self.declare_parameter("recover_vq", 0.55)

        # Command vocabulary
        self.declare_parameter("cmd_hold", "HOLD_COURSE")
        self.declare_parameter("cmd_slow", "SLOW_DOWN")
        self.declare_parameter("cmd_turn_left_slow", "TURN_LEFT_SLOW")
        self.declare_parameter("cmd_turn_right_slow", "TURN_RIGHT_SLOW")
        self.declare_parameter("cmd_stop", "STOP")

        # LOST triggers
        self.declare_parameter("lost_if_image_stale", True)
        self.declare_parameter("lost_if_risk_stale", True)
        self.declare_parameter("lost_if_mode_stale", False)
        self.declare_parameter("lost_if_mode_lost", True)
        self.declare_parameter("lost_if_freeze_true", True)

        # Caution limiting
        self.declare_parameter("cap_turns_in_caution", True)

        # Logging
        self.declare_parameter("log_state_change", True)

        depth = int(self.get_parameter("qos_depth").value)
        rel = str(self.get_parameter("sub_reliability").value)
        qos = _qos(depth=depth, reliability=rel)

        # -------------------------
        # Internal state
        # -------------------------
        self.t0 = _now_s()

        # Multi image tracking
        self.image_topics: List[str] = self._get_image_topics()
        self.last_image_ts: Dict[str, Optional[float]] = {t: None for t in self.image_topics}
        self.last_image_topic: str = ""

        self.last_risk_t: Optional[float] = None
        self.risk: float = 0.0

        self.last_mode_t: Optional[float] = None
        self.mode: str = "UNKNOWN"

        self.last_cmd_t: Optional[float] = None
        self.cmd_in: str = "HOLD_COURSE"

        self.last_vq_t: Optional[float] = None
        self.vq: Optional[float] = None

        self.last_freeze_t: Optional[float] = None
        self.freeze: Optional[bool] = None
        self.freeze_reason: str = ""

        self.state: str = "LOST" if bool(self.get_parameter("start_in_failsafe").value) else "NORMAL"
        self.state_enter_t: float = _now_s()
        self.ok_since_t: Optional[float] = None

        # -------------------------
        # Subscribers
        # -------------------------
        self.image_subs = []
        for tpc in self.image_topics:
            sub = self.create_subscription(
                Image,
                tpc,
                lambda msg, topic=tpc: self._on_image(msg, topic),
                qos,
            )
            self.image_subs.append(sub)

        self.sub_risk = self.create_subscription(
            Float32,
            str(self.get_parameter("risk_topic").value),
            self._on_risk,
            qos,
        )
        self.sub_mode = self.create_subscription(
            String,
            str(self.get_parameter("mode_topic").value),
            self._on_mode,
            qos,
        )
        self.sub_cmd = self.create_subscription(
            String,
            str(self.get_parameter("command_in_topic").value),
            self._on_cmd,
            qos,
        )
        self.sub_vq = self.create_subscription(
            Float32,
            str(self.get_parameter("vision_quality_topic").value),
            self._on_vq,
            qos,
        )
        self.sub_freeze = self.create_subscription(
            Bool,
            str(self.get_parameter("freeze_topic").value),
            self._on_freeze,
            qos,
        )
        self.sub_freeze_reason = self.create_subscription(
            String,
            str(self.get_parameter("freeze_reason_topic").value),
            self._on_freeze_reason,
            qos,
        )

        # -------------------------
        # Publishers
        # -------------------------
        self.pub_cmd_safe = self.create_publisher(
            String,
            str(self.get_parameter("command_out_topic").value),
            _qos(depth=10, reliability="reliable"),
        )
        self.pub_failsafe = self.create_publisher(
            Bool,
            str(self.get_parameter("failsafe_active_topic").value),
            _qos(depth=10, reliability="reliable"),
        )
        self.pub_reason = self.create_publisher(
            String,
            str(self.get_parameter("failsafe_reason_topic").value),
            _qos(depth=10, reliability="reliable"),
        )
        self.pub_status = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            _qos(depth=10, reliability="reliable"),
        )

        tick_hz = float(self.get_parameter("tick_hz").value)
        tick_hz = _clamp(tick_hz, 1.0, 50.0)
        self.timer = self.create_timer(1.0 / tick_hz, self._on_tick)

        self.get_logger().info(
            f"watchdog_failsafe_node started: state={self.state}, image_topics={self.image_topics}, "
            f"image_timeout_s={float(self.get_parameter('image_timeout_s').value):.2f}"
        )

    # -------------------------
    # Parameter helpers
    # -------------------------
    def _get_image_topics(self) -> List[str]:
        v = self.get_parameter("image_topics").value
        topics: List[str] = []
        if isinstance(v, (list, tuple)):
            topics = [str(x) for x in v if str(x).strip()]
        else:
            s = str(v).strip()
            topics = [s] if s else []
        seen = set()
        out = []
        for t in topics:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out if out else ["/camera/image_raw_reliable"]

    # -------------------------
    # Callbacks
    # -------------------------
    def _on_image(self, _msg: Image, topic: str) -> None:
        t = _now_s()
        if topic in self.last_image_ts:
            self.last_image_ts[topic] = t
        self.last_image_topic = topic

    def _on_risk(self, msg: Float32) -> None:
        self.last_risk_t = _now_s()
        try:
            self.risk = float(msg.data)
        except Exception:
            self.risk = 0.0

    def _on_mode(self, msg: String) -> None:
        self.last_mode_t = _now_s()
        self.mode = str(msg.data)

    def _on_cmd(self, msg: String) -> None:
        self.last_cmd_t = _now_s()
        self.cmd_in = str(msg.data)

    def _on_vq(self, msg: Float32) -> None:
        self.last_vq_t = _now_s()
        try:
            self.vq = float(msg.data)
        except Exception:
            self.vq = None

    def _on_freeze(self, msg: Bool) -> None:
        self.last_freeze_t = _now_s()
        try:
            self.freeze = bool(msg.data)
        except Exception:
            self.freeze = None

    def _on_freeze_reason(self, msg: String) -> None:
        self.last_freeze_t = _now_s()
        self.freeze_reason = str(msg.data)

    # -------------------------
    # Helpers
    # -------------------------
    def _age(self, last_t: Optional[float]) -> float:
        if last_t is None:
            return 1e9
        return max(0.0, _now_s() - last_t)

    def _in_startup_grace(self) -> bool:
        grace = float(self.get_parameter("startup_grace_s").value)
        grace = max(0.0, grace)
        return (_now_s() - self.t0) < grace

    def _image_best_age(self) -> Tuple[float, str]:
        best_age = 1e9
        best_topic = ""
        for tpc, ts in self.last_image_ts.items():
            a = self._age(ts)
            if a < best_age:
                best_age = a
                best_topic = tpc
        return best_age, best_topic

    def _vq_valid(self) -> Tuple[bool, Optional[float]]:
        vq_to = float(self.get_parameter("vision_quality_timeout_s").value)
        if self.vq is None:
            return (False, None)
        if self._age(self.last_vq_t) > vq_to:
            return (False, None)
        return (True, float(self.vq))

    def _freeze_active(self) -> Tuple[bool, str]:
        if self.freeze is None:
            return (False, "")
        fr_to = float(self.get_parameter("freeze_timeout_s").value)
        if self._age(self.last_freeze_t) > fr_to:
            return (False, "")
        if bool(self.freeze):
            return (True, self.freeze_reason or "freeze_true")
        return (False, "")

    def _limit_cmd_for_caution(self, cmd: str) -> str:
        cmd = str(cmd or "").strip().upper()
        cmd_hold = str(self.get_parameter("cmd_hold").value)
        cmd_slow = str(self.get_parameter("cmd_slow").value)
        cmd_tls = str(self.get_parameter("cmd_turn_left_slow").value)
        cmd_trs = str(self.get_parameter("cmd_turn_right_slow").value)

        if not bool(self.get_parameter("cap_turns_in_caution").value):
            return cmd if cmd else cmd_hold

        if cmd in (cmd_hold, cmd_slow, cmd_tls, cmd_trs):
            return cmd

        if cmd == "TURN_LEFT":
            return cmd_tls
        if cmd == "TURN_RIGHT":
            return cmd_trs

        if cmd.startswith("TURN_") and (not cmd.endswith("_SLOW")):
            if "LEFT" in cmd:
                return cmd_tls
            if "RIGHT" in cmd:
                return cmd_trs
            return cmd_slow

        return cmd_slow if cmd else cmd_hold

    def _transition(self, new_state: str) -> None:
        new_state = str(new_state).strip().upper()
        if new_state == self.state:
            return
        self.state = new_state
        self.state_enter_t = _now_s()
        self.ok_since_t = None
        if bool(self.get_parameter("log_state_change").value):
            self.get_logger().warn(f"[WATCHDOG] STATE -> {self.state}")

    def _compute_reasons(self) -> Tuple[List[str], List[str]]:
        lost: List[str] = []
        caution: List[str] = []

        skip_stale = self._in_startup_grace()

        image_to = float(self.get_parameter("image_timeout_s").value)
        risk_to = float(self.get_parameter("risk_timeout_s").value)
        mode_to = float(self.get_parameter("mode_timeout_s").value)

        if not skip_stale:
            best_age, _best_topic = self._image_best_age()
            if bool(self.get_parameter("lost_if_image_stale").value) and best_age > image_to:
                lost.append(f"image_stale>{image_to:.1f}s")

            if bool(self.get_parameter("lost_if_risk_stale").value) and self._age(self.last_risk_t) > risk_to:
                lost.append(f"risk_stale>{risk_to:.1f}s")

            if bool(self.get_parameter("lost_if_mode_stale").value) and self._age(self.last_mode_t) > mode_to:
                lost.append(f"mode_stale>{mode_to:.1f}s")

        if bool(self.get_parameter("lost_if_mode_lost").value) and _is_lost_mode(self.mode):
            lost.append(f"mode={_norm_mode(self.mode)}")

        fr_active, fr_reason = self._freeze_active()
        if bool(self.get_parameter("lost_if_freeze_true").value) and fr_active:
            lost.append(f"freeze_true:{fr_reason}")

        if _is_caution_mode(self.mode):
            caution.append(f"mode={_norm_mode(self.mode)}")

        vq_ok, vq = self._vq_valid()
        if vq_ok and vq is not None:
            enter = float(self.get_parameter("vq_caution_enter").value)
            if vq < enter:
                caution.append(f"vq<{enter:.2f}")

        return (lost, caution)

    def _on_tick(self) -> None:
        t = _now_s()

        cmd_hold = str(self.get_parameter("cmd_hold").value)
        cmd_slow = str(self.get_parameter("cmd_slow").value)
        cmd_stop = str(self.get_parameter("cmd_stop").value)
        cmd_to = float(self.get_parameter("command_timeout_s").value)

        lost_reasons, caution_reasons = self._compute_reasons()
        wants_lost = (len(lost_reasons) > 0)

        vq_ok, vq = self._vq_valid()
        vq_enter = float(self.get_parameter("vq_caution_enter").value)
        vq_exit = float(self.get_parameter("vq_caution_exit").value)
        mode_caution = _is_caution_mode(self.mode)

        if self.state == "CAUTION":
            wants_caution = mode_caution
            if vq_ok and vq is not None:
                wants_caution = wants_caution or (vq < vq_exit)
        else:
            wants_caution = mode_caution
            if vq_ok and vq is not None:
                wants_caution = wants_caution or (vq < vq_enter)

        if self.state == "LOST":
            lost_min_hold = float(self.get_parameter("lost_min_hold_s").value)
            recover_hold = float(self.get_parameter("recover_ok_hold_s").value)
            recover_vq = float(self.get_parameter("recover_vq").value)

            held_long_enough = (t - self.state_enter_t) >= lost_min_hold
            ok_cond = not wants_lost
            if vq_ok and (vq is not None):
                ok_cond = ok_cond and (vq >= recover_vq)

            if held_long_enough and ok_cond:
                if self.ok_since_t is None:
                    self.ok_since_t = t
                if (t - self.ok_since_t) >= recover_hold:
                    self._transition("CAUTION" if wants_caution else "NORMAL")
            else:
                self.ok_since_t = None
        else:
            if wants_lost:
                self._transition("LOST")
            else:
                self._transition("CAUTION" if wants_caution else "NORMAL")

        cmd_in_ok = (self._age(self.last_cmd_t) <= cmd_to)
        cmd_in = (self.cmd_in if cmd_in_ok else "")

        if self.state == "LOST":
            cmd_safe = cmd_stop
            failsafe_active = True
            reason = ";".join(lost_reasons) if lost_reasons else "lost"
        elif self.state == "CAUTION":
            cmd_safe = self._limit_cmd_for_caution(cmd_in)
            if not cmd_safe:
                cmd_safe = cmd_slow
            failsafe_active = False
            reason = ";".join(caution_reasons) if caution_reasons else "caution"
        else:
            cmd_safe = cmd_in if cmd_in else cmd_hold
            failsafe_active = False
            reason = "ok"

        self.pub_cmd_safe.publish(String(data=str(cmd_safe)))
        self.pub_failsafe.publish(Bool(data=bool(failsafe_active)))
        self.pub_reason.publish(String(data=str(reason)))

        best_age, best_topic = self._image_best_age()
        per_topic_age = {tpc: round(self._age(ts), 3) for tpc, ts in self.last_image_ts.items()}

        status = {
            "t": round(t - self.t0, 3),
            "state": self.state,
            "failsafe_active": bool(failsafe_active),
            "reason": reason,
            "image": {
                "best_topic": best_topic,
                "best_age_s": round(best_age, 3),
                "ages_s": per_topic_age,
                "last_topic_seen": self.last_image_topic,
            },
            "in": {
                "mode": self.mode,
                "risk": round(float(self.risk), 3),
                "cmd": self.cmd_in,
                "vq": None if self.vq is None else round(float(self.vq), 3),
                "freeze": None if self.freeze is None else bool(self.freeze),
            },
            "ages_s": {
                "risk": round(self._age(self.last_risk_t), 3),
                "mode": round(self._age(self.last_mode_t), 3),
                "cmd": round(self._age(self.last_cmd_t), 3),
                "vq": round(self._age(self.last_vq_t), 3),
                "freeze": round(self._age(self.last_freeze_t), 3),
            },
            "startup_grace": bool(self._in_startup_grace()),
        }
        self.pub_status.publish(String(data=json.dumps(status, ensure_ascii=True)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WatchdogFailsafeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
