#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission / Mode Manager (ROS 2 Humble) — SEANO CA (ROBUST + REJOIN)

Tambahan utama dari versi sebelumnya:
- Menambahkan state REJOIN agar alur formal menjadi:
  FAILSAFE -> AVOID -> REJOIN -> MISSION
- Setelah takeover OFF, node tidak langsung menganggap MISSION selesai pulih.
  Node masuk ke REJOIN dulu, meminta mode mission target (default AUTO),
  lalu menunggu mode stabil selama beberapa saat sebelum publish MISSION.
- Menambahkan event:
  REJOIN_START, REJOIN_DONE, REJOIN_TIMEOUT, REJOIN_CANCELLED

State machine (FASE 5/6 mind map):
- FAILSAFE : /ca/failsafe_active true              -> target failsafe_mode (default MANUAL)
- AVOID    : /seano/rc_override_enable true        -> target avoid_mode (default MANUAL)
- REJOIN   : takeover OFF setelah AVOID/FAILSAFE   -> target mission restore mode (default AUTO)
- MISSION  : default setelah REJOIN selesai        -> target mission restore mode (default AUTO)

Input:
- /mavros/state (mavros_msgs/State)
- /seano/rc_override_enable (std_msgs/Bool)
- /ca/failsafe_active (std_msgs/Bool)

Output:
- /ca/mode_manager_state (String): MISSION/AVOID/REJOIN/FAILSAFE
- /ca/mode_manager_event (String): JSON event log

Action:
- service /mavros/set_mode (mavros_msgs/srv/SetMode)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Optional

from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


def _qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _now_s() -> float:
    return time.monotonic()


def _norm_mode(s: str) -> str:
    return str(s or "").strip().upper().replace("-", "_").replace(" ", "_")


@dataclass
class _MgrState:
    last_override: Optional[bool] = None
    last_failsafe: Optional[bool] = None

    mavros_connected: bool = False
    mavros_mode: str = "UNKNOWN"

    pending_mode: Optional[str] = None
    pending_since: float = 0.0

    last_mode_req_t: float = 0.0
    last_enforce_t: float = 0.0

    restore_mode_after_avoid: Optional[str] = None
    restore_mode_after_failsafe: Optional[str] = None

    rejoin_active: bool = False
    rejoin_since: float = 0.0
    rejoin_target_mode: Optional[str] = None
    rejoin_mode_match_since: float = 0.0
    avoid_active_decision: bool = False
    confirmed_avoid_session: bool = False


class MissionModeManager(Node):
    def __init__(self) -> None:
        super().__init__("mission_mode_manager_node")

        # -------- Parameters --------
        self.declare_parameter("mavros_state_topic", "/mavros/state")
        self.declare_parameter("rc_override_enable_topic", "/seano/rc_override_enable")
        self.declare_parameter("failsafe_active_topic", "/ca/failsafe_active")
        self.declare_parameter("set_mode_service", "/mavros/set_mode")

        # Mode policy
        self.declare_parameter("avoid_mode", "MANUAL")
        self.declare_parameter("mission_mode_default", "AUTO")
        self.declare_parameter("failsafe_mode", "MANUAL")

        # Behavior toggles
        self.declare_parameter("switch_to_avoid_on_takeover", True)
        self.declare_parameter("restore_mode_on_release", True)
        self.declare_parameter("switch_to_failsafe_on_failsafe", True)
        self.declare_parameter("restore_after_failsafe_if_clear", True)
        self.declare_parameter("enable_rejoin_state", True)

        # Robustness
        self.declare_parameter("enforce_mode", True)
        self.declare_parameter("enforce_period_s", 1.5)
        self.declare_parameter("min_mode_switch_interval_s", 0.8)
        self.declare_parameter("pending_timeout_s", 3.0)

        # Rejoin policy
        self.declare_parameter("rejoin_stable_time_s", 2.0)
        self.declare_parameter("rejoin_timeout_s", 12.0)

        # Outputs
        self.declare_parameter("state_out_topic", "/ca/mode_manager_state")
        self.declare_parameter("event_out_topic", "/ca/mode_manager_event")
        self.declare_parameter("avoid_active_topic", "/ca/avoid_active")
        self.declare_parameter("rejoin_required_avoid_s", 0.50)
        self.declare_parameter("avoid_exit_hold_s", 0.75)
        self.declare_parameter("avoid_enter_min_s", 0.05)
        self.declare_parameter("valid_avoid_min_s", 0.80)

        # Tick
        self.declare_parameter("tick_hz", 5.0)

        self.st = _MgrState()
        # SEANO_STATE_MACHINE_GOVERNOR_V5
        self.st.avoid_active_decision = False
        self.st.avoid_true_since = 0.0
        self.st.avoid_false_since = 0.0
        self.st.avoid_session_start = 0.0
        self.st.confirmed_avoid_session = False
        # SEANO confirmed avoidance session latch
        self.st.avoid_active_decision = False
        self.st.confirmed_avoid_session = False
        self.st.avoid_true_since = 0.0
        self.st.avoid_session_start = 0.0

        self.pub_state = self.create_publisher(
            String, str(self.get_parameter("state_out_topic").value), _qos(10)
        )
        self.pub_event = self.create_publisher(
            String, str(self.get_parameter("event_out_topic").value), _qos(10)
        )
        self.sub_avoid_active = self.create_subscription(
            Bool,
            str(self.get_parameter("avoid_active_topic").value),
            self._cb_avoid_active,
            _qos(10),
        )

        self.create_subscription(
            State,
            str(self.get_parameter("mavros_state_topic").value),
            self._cb_mavros_state,
            _qos(10),
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("rc_override_enable_topic").value),
            self._cb_override,
            _qos(10),
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("failsafe_active_topic").value),
            self._cb_failsafe,
            _qos(10),
        )

        self.cli_set_mode = self.create_client(
            SetMode, str(self.get_parameter("set_mode_service").value)
        )

        hz = float(self.get_parameter("tick_hz").value)
        if hz <= 0.0:
            hz = 5.0
        self.create_timer(1.0 / hz, self._tick)

        self._emit_event(
            "START",
            {
                "avoid_mode": str(self.get_parameter("avoid_mode").value),
                "mission_mode_default": str(self.get_parameter("mission_mode_default").value),
                "failsafe_mode": str(self.get_parameter("failsafe_mode").value),
                "enable_rejoin_state": bool(self.get_parameter("enable_rejoin_state").value),
                "rejoin_stable_time_s": float(self.get_parameter("rejoin_stable_time_s").value),
                "rejoin_timeout_s": float(self.get_parameter("rejoin_timeout_s").value),
            },
        )

    # -------- Callbacks --------

    def _cb_avoid_active(self, msg: Bool) -> None:
        now = float(self.get_clock().now().nanoseconds) * 1.0e-9
        cur = bool(msg.data)
        prev = bool(getattr(self.st, "avoid_active_decision", False))

        self.st.avoid_active_decision = cur

        if cur:
            if not prev:
                self.st.avoid_true_since = now
                self.st.avoid_session_start = now
                self.st.avoid_false_since = 0.0
                self._emit_event("AVOID_ACTIVE", {"active": True})

            enter_min = float(self.get_parameter("avoid_enter_min_s").value)
            true_for = now - float(getattr(self.st, "avoid_true_since", now))

            if true_for >= enter_min and not bool(
                getattr(self.st, "confirmed_avoid_session", False)
            ):
                self.st.confirmed_avoid_session = True
                self._emit_event("AVOID_CONFIRMED", {"duration_s": round(true_for, 3)})

        else:
            if prev:
                self.st.avoid_false_since = now
                start = float(getattr(self.st, "avoid_true_since", 0.0))
                dur = now - start if start > 0.0 else 0.0

                if not bool(getattr(self.st, "confirmed_avoid_session", False)):
                    self._emit_event(
                        "AVOID_BLIP_REJECTED",
                        {"duration_s": round(dur, 3), "reason": "below_avoid_enter_min_s"},
                    )

                self._emit_event("AVOID_ACTIVE", {"active": False})

            self.st.avoid_true_since = 0.0

    def _cb_mavros_state(self, msg: State) -> None:
        self.st.mavros_connected = bool(msg.connected)
        self.st.mavros_mode = str(msg.mode or "UNKNOWN")

    def _cb_override(self, msg: Bool) -> None:
        cur = bool(msg.data)
        prev = self.st.last_override
        self.st.last_override = cur

        if prev is None:
            return

        # rising: takeover ON
        if (not prev) and cur:
            self.st.restore_mode_after_avoid = self._current_mission_restore_target()
            self._cancel_rejoin("takeover_on")
            self._emit_event(
                "TAKEOVER_ON",
                {
                    "restore_mode": self.st.restore_mode_after_avoid,
                    "avoid_mode": str(self.get_parameter("avoid_mode").value),
                },
            )
            if bool(self.get_parameter("switch_to_avoid_on_takeover").value):
                self._request_mode(str(self.get_parameter("avoid_mode").value), cause="takeover_on")

        # falling: takeover OFF
        if prev and (not cur):
            restore = self.st.restore_mode_after_avoid or str(
                self.get_parameter("mission_mode_default").value
            )
            self._emit_event("TAKEOVER_OFF", {"restore_mode": restore})

            if bool(self.get_parameter("enable_rejoin_state").value):
                self._start_rejoin(restore, reason="takeover_off")
            elif bool(self.get_parameter("restore_mode_on_release").value):
                self._request_mode(restore, cause="takeover_off_restore")

            self.st.restore_mode_after_avoid = None

    def _cb_failsafe(self, msg: Bool) -> None:
        cur = bool(msg.data)
        prev = self.st.last_failsafe
        self.st.last_failsafe = cur

        if prev is None:
            return

        # rising: failsafe ON
        if (not prev) and cur:
            self.st.restore_mode_after_failsafe = self._current_mission_restore_target()
            self._cancel_rejoin("failsafe_on")
            self._emit_event(
                "FAILSAFE_ON",
                {
                    "restore_mode": self.st.restore_mode_after_failsafe,
                    "failsafe_mode": str(self.get_parameter("failsafe_mode").value),
                },
            )
            if bool(self.get_parameter("switch_to_failsafe_on_failsafe").value):
                self._request_mode(
                    str(self.get_parameter("failsafe_mode").value), cause="failsafe_on"
                )

        # falling: failsafe OFF
        if prev and (not cur):
            self._emit_event("FAILSAFE_OFF", {})

            if bool(self.get_parameter("restore_after_failsafe_if_clear").value):
                if bool(self.st.last_override):
                    self._emit_event("RESTORE_SKIP", {"reason": "takeover_still_on"})
                else:
                    restore = self.st.restore_mode_after_failsafe or str(
                        self.get_parameter("mission_mode_default").value
                    )
                    if bool(self.get_parameter("enable_rejoin_state").value):
                        self._start_rejoin(restore, reason="failsafe_off")
                    else:
                        self._request_mode(restore, cause="failsafe_off_restore")

            self.st.restore_mode_after_failsafe = None

    # -------- Core tick / enforcement --------
    def _tick(self) -> None:
        override_on = bool(self.st.last_override) if self.st.last_override is not None else False
        failsafe_on = bool(self.st.last_failsafe) if self.st.last_failsafe is not None else False

        mgr_state = self._compute_mgr_state(override_on=override_on, failsafe_on=failsafe_on)
        self.pub_state.publish(String(data=mgr_state))

        if mgr_state == "REJOIN":
            self._tick_rejoin()

        if not bool(self.get_parameter("enforce_mode").value):
            return

        now = _now_s()
        enforce_period = float(self.get_parameter("enforce_period_s").value)
        if enforce_period <= 0.0:
            enforce_period = 1.5

        if (now - self.st.last_enforce_t) < enforce_period:
            return
        self.st.last_enforce_t = now

        target = self._desired_mode(mgr_state)
        cur_mode = _norm_mode(self.st.mavros_mode)

        pending_timeout = float(self.get_parameter("pending_timeout_s").value)
        if self.st.pending_mode is not None and pending_timeout > 0.0:
            if (now - self.st.pending_since) > pending_timeout:
                self._emit_event("PENDING_TIMEOUT", {"pending_mode": self.st.pending_mode})
                self.st.pending_mode = None

        if cur_mode == _norm_mode(target):
            return

        self._emit_event(
            "ENFORCE",
            {"mgr_state": mgr_state, "target": target, "current": cur_mode},
        )
        self._request_mode(target, cause=f"enforce_{mgr_state.lower()}")

    def _compute_mgr_state(self, override_on: bool, failsafe_on: bool) -> str:
        if failsafe_on:
            return "FAILSAFE"

        now = float(self.get_clock().now().nanoseconds) * 1.0e-9

        avoid_active = bool(getattr(self.st, "avoid_active_decision", False))
        confirmed = bool(getattr(self.st, "confirmed_avoid_session", False))

        enter_min = float(self.get_parameter("avoid_enter_min_s").value)
        exit_hold = float(self.get_parameter("avoid_exit_hold_s").value)

        true_since = float(getattr(self.st, "avoid_true_since", 0.0))
        false_since = float(getattr(self.st, "avoid_false_since", 0.0))

        if avoid_active:
            if true_since > 0.0 and (now - true_since) >= enter_min:
                self.st.confirmed_avoid_session = True
                return "AVOID"

            # Provisional hazard, not a confirmed avoid state yet.
            return "MISSION"

        if confirmed:
            if false_since > 0.0 and (now - false_since) <= exit_hold:
                return "AVOID"

            if bool(self.get_parameter("enable_rejoin_state").value) and bool(
                getattr(self.st, "rejoin_active", False)
            ):
                return "REJOIN"

        return "MISSION"

    def _tick_rejoin(self) -> None:
        target = _norm_mode(self.st.rejoin_target_mode or self._current_mission_restore_target())
        cur = _norm_mode(self.st.mavros_mode)
        now = _now_s()

        stable_required = max(0.0, float(self.get_parameter("rejoin_stable_time_s").value))
        timeout_s = max(0.0, float(self.get_parameter("rejoin_timeout_s").value))

        if cur == target and target:
            if self.st.rejoin_mode_match_since <= 0.0:
                self.st.rejoin_mode_match_since = now
                self._emit_event(
                    "REJOIN_MODE_MATCH",
                    {"target_mode": target, "stable_required_s": stable_required},
                )
            elif (now - self.st.rejoin_mode_match_since) >= stable_required:
                elapsed = max(0.0, now - self.st.rejoin_since)
                self._emit_event(
                    "REJOIN_DONE",
                    {"target_mode": target, "elapsed_s": round(elapsed, 3)},
                )
                self._clear_rejoin()
        else:
            if self.st.rejoin_mode_match_since > 0.0:
                self._emit_event(
                    "REJOIN_MODE_UNSTABLE",
                    {"target_mode": target, "current_mode": cur},
                )
            self.st.rejoin_mode_match_since = 0.0

        if self.st.rejoin_active and timeout_s > 0.0:
            elapsed = now - self.st.rejoin_since
            if elapsed >= timeout_s:
                self._emit_event(
                    "REJOIN_TIMEOUT",
                    {
                        "target_mode": target,
                        "current_mode": cur,
                        "elapsed_s": round(elapsed, 3),
                    },
                )
                self._clear_rejoin()

    def _start_rejoin(self, restore_mode: str, reason: str) -> None:
        target = _norm_mode(restore_mode) or _norm_mode(
            str(self.get_parameter("mission_mode_default").value)
        )
        now = _now_s()

        if not bool(getattr(self.st, "confirmed_avoid_session", False)):
            self._emit_event("REJOIN_SKIPPED", {"reason": "no_confirmed_avoid_session"})
            return
        # SEANO_REJOIN_REQUIRES_CONFIRMED_AVOID_V3
        if not bool(getattr(self.st, "confirmed_avoid_session", False)):
            self._emit_event("REJOIN_SKIPPED", {"reason": "no_confirmed_avoid_session"})
            return
        # SEANO_REJOIN_REQUIRES_CONFIRMED_SESSION_V5
        _now = float(self.get_clock().now().nanoseconds) * 1.0e-9
        _start = float(getattr(self.st, "avoid_session_start", 0.0))
        _dur = (_now - _start) if _start > 0.0 else 0.0
        _min = float(self.get_parameter("rejoin_required_avoid_s").value)
        if (not bool(getattr(self.st, "confirmed_avoid_session", False))) or (_dur < _min):
            self.st.rejoin_active = False
            self.st.confirmed_avoid_session = False
            self._emit_event(
                "REJOIN_SKIPPED",
                {"reason": "no_valid_avoid_session", "avoid_duration_s": round(_dur, 3)},
            )
            return
        self.st.rejoin_active = True
        self.st.rejoin_since = now
        self.st.rejoin_target_mode = target
        self.st.rejoin_mode_match_since = 0.0

        self._emit_event("REJOIN_START", {"target_mode": target, "reason": reason})

        if reason == "takeover_off":
            if bool(self.get_parameter("restore_mode_on_release").value):
                self._request_mode(target, cause="rejoin_start_restore")
        elif reason == "failsafe_off":
            if bool(self.get_parameter("restore_after_failsafe_if_clear").value):
                self._request_mode(target, cause="rejoin_start_restore")

    def _clear_rejoin(self) -> None:
        self.st.rejoin_active = False
        self.st.rejoin_since = 0.0
        self.st.rejoin_target_mode = None
        self.st.rejoin_mode_match_since = 0.0

    def _cancel_rejoin(self, reason: str) -> None:
        if not self.st.rejoin_active:
            return
        elapsed = max(0.0, _now_s() - self.st.rejoin_since)
        self._emit_event("REJOIN_CANCELLED", {"reason": reason, "elapsed_s": round(elapsed, 3)})
        self._clear_rejoin()

    def _desired_mode(self, mgr_state: str) -> str:
        if mgr_state == "FAILSAFE":
            return str(self.get_parameter("failsafe_mode").value)
        if mgr_state == "AVOID":
            return str(self.get_parameter("avoid_mode").value)
        if mgr_state == "REJOIN":
            return str(self.st.rejoin_target_mode or self._current_mission_restore_target())
        return self._current_mission_restore_target()

    def _current_mission_restore_target(self) -> str:
        cur = _norm_mode(self.st.mavros_mode)
        mission_default = str(self.get_parameter("mission_mode_default").value)

        if cur and cur not in ("MANUAL", "STABILIZE"):
            return cur
        return mission_default

    def _request_mode(self, mode: str, cause: str) -> None:
        mode = _norm_mode(mode)
        # SEANO_AVOID_ACTIVE_MODE_REQUEST_GATE
        cause_l = str(cause).lower()
        avoid_target = _norm_mode(str(self.get_parameter("avoid_mode").value))
        avoid_related = ("avoid" in cause_l) or ("takeover" in cause_l)
        if (
            mode == avoid_target
            and avoid_related
            and not bool(getattr(self.st, "avoid_active_decision", False))
        ):
            self._emit_event(
                "MODE_REQ_SKIPPED",
                {"mode": mode, "cause": cause, "reason": "avoid_active_false"},
            )
            return
        if not mode:
            return

        if not self.st.mavros_connected:
            self._emit_event(
                "MODE_REQ_SKIPPED",
                {"mode": mode, "cause": cause, "reason": "mavros_not_connected"},
            )
            return

        if not self.cli_set_mode.service_is_ready():
            self._emit_event(
                "MODE_REQ_SKIPPED",
                {"mode": mode, "cause": cause, "reason": "set_mode_srv_not_ready"},
            )
            return

        min_dt = float(self.get_parameter("min_mode_switch_interval_s").value)
        now = _now_s()
        if (now - self.st.last_mode_req_t) < max(0.0, min_dt):
            self._emit_event(
                "MODE_REQ_SKIPPED",
                {"mode": mode, "cause": cause, "reason": "rate_limited"},
            )
            return

        if self.st.pending_mode is not None:
            self._emit_event(
                "MODE_REQ_SKIPPED",
                {"mode": mode, "cause": cause, "reason": "pending_exists"},
            )
            return

        if _norm_mode(self.st.mavros_mode) == mode:
            self._emit_event("MODE_ALREADY", {"mode": mode, "cause": cause})
            return

        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = mode

        self.st.last_mode_req_t = now
        self.st.pending_mode = mode
        self.st.pending_since = now
        self._emit_event("MODE_REQ_SENT", {"mode": mode, "cause": cause})

        fut = self.cli_set_mode.call_async(req)

        def _done_cb(fut_obj) -> None:
            ok = False
            detail = ""
            try:
                resp = fut_obj.result()
                ok = bool(resp.mode_sent)
                detail = "mode_sent=true" if ok else "mode_sent=false"
            except Exception as exc:
                ok = False
                detail = f"exception:{type(exc).__name__}"
            self._emit_event("MODE_REQ_DONE", {"mode": mode, "ok": ok, "detail": detail})
            self.st.pending_mode = None

        fut.add_done_callback(_done_cb)

    def _emit_event(self, name: str, payload: dict) -> None:
        # SEANO_REJOIN_EVENT_SUPPRESSOR_V5
        try:
            _ev = str(name)
            if (
                _ev.startswith("REJOIN")
                and _ev != "REJOIN_SKIPPED"
                and not bool(getattr(self.st, "confirmed_avoid_session", False))
            ):
                return
        except Exception:
            pass

        evt = {
            "t": round(_now_s(), 3),
            "event": str(name),
            "mavros": {"connected": self.st.mavros_connected, "mode": self.st.mavros_mode},
            "payload": payload,
        }
        self.pub_event.publish(String(data=json.dumps(evt, ensure_ascii=True)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionModeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
