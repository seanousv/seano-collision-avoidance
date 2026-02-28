#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission / Mode Manager (ROS 2 Humble) — SEANO CA (ROBUST)

Fix utama dari versi sebelumnya:
- Tidak hanya edge-based. Ada "enforce loop" (periodik) agar kalau restore mode ke-skip,
  node akan mencoba lagi sampai mode benar (tanpa spam).
- Menghindari kasus: state=MISSION tetapi autopilot masih MANUAL.

State machine (FASE 5 mind map):
- FAILSAFE  : /ca/failsafe_active true  -> target failsafe_mode (default MANUAL)
- AVOID     : /seano/rc_override_enable true -> target avoid_mode (default MANUAL)
- MISSION   : default -> target mission_mode_default (default AUTO)

Input:
- /mavros/state (mavros_msgs/State)
- /seano/rc_override_enable (std_msgs/Bool)
- /ca/failsafe_active (std_msgs/Bool)

Output:
- /ca/mode_manager_state (String): MISSION/AVOID/FAILSAFE
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

    # last known mavros mode
    mavros_connected: bool = False
    mavros_mode: str = "UNKNOWN"

    # pending mode request tracking
    pending_mode: Optional[str] = None
    pending_since: float = 0.0

    # rate limit / enforcement
    last_mode_req_t: float = 0.0
    last_enforce_t: float = 0.0

    # restore targets captured at edge
    restore_mode_after_avoid: Optional[str] = None
    restore_mode_after_failsafe: Optional[str] = None


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

        # Robustness
        self.declare_parameter("enforce_mode", True)
        self.declare_parameter("enforce_period_s", 1.5)  # coba set_mode ulang tiap ini
        self.declare_parameter("min_mode_switch_interval_s", 0.8)  # anti-spam
        self.declare_parameter("pending_timeout_s", 3.0)  # kalau pending terlalu lama, boleh retry

        # Outputs
        self.declare_parameter("state_out_topic", "/ca/mode_manager_state")
        self.declare_parameter("event_out_topic", "/ca/mode_manager_event")

        # Tick
        self.declare_parameter("tick_hz", 5.0)

        self.st = _MgrState()

        # -------- ROS IO --------
        self.pub_state = self.create_publisher(
            String, str(self.get_parameter("state_out_topic").value), _qos(10)
        )
        self.pub_event = self.create_publisher(
            String, str(self.get_parameter("event_out_topic").value), _qos(10)
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
        if hz <= 0:
            hz = 5.0
        self.create_timer(1.0 / hz, self._tick)

        self._emit_event(
            "START",
            {
                "avoid_mode": str(self.get_parameter("avoid_mode").value),
                "mission_mode_default": str(self.get_parameter("mission_mode_default").value),
                "failsafe_mode": str(self.get_parameter("failsafe_mode").value),
            },
        )

    # -------- Callbacks --------
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
            # simpan mode sebelum avoid untuk restore nanti
            self.st.restore_mode_after_avoid = self._current_mission_restore_target()
            self._emit_event(
                "TAKEOVER_ON",
                {
                    "restore_mode": self.st.restore_mode_after_avoid,
                    "avoid_mode": self.get_parameter("avoid_mode").value,
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
            if bool(self.get_parameter("restore_mode_on_release").value):
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
            self._emit_event(
                "FAILSAFE_ON",
                {
                    "restore_mode": self.st.restore_mode_after_failsafe,
                    "failsafe_mode": self.get_parameter("failsafe_mode").value,
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
                # kalau takeover masih ON, jangan restore mission dulu
                if bool(self.st.last_override):
                    self._emit_event("RESTORE_SKIP", {"reason": "takeover_still_on"})
                else:
                    restore = self.st.restore_mode_after_failsafe or str(
                        self.get_parameter("mission_mode_default").value
                    )
                    self._request_mode(restore, cause="failsafe_off_restore")
            self.st.restore_mode_after_failsafe = None

    # -------- Core tick / enforcement --------
    def _tick(self) -> None:
        override_on = bool(self.st.last_override) if (self.st.last_override is not None) else False
        failsafe_on = bool(self.st.last_failsafe) if (self.st.last_failsafe is not None) else False

        # publish high-level state
        if failsafe_on:
            mgr_state = "FAILSAFE"
        elif override_on:
            mgr_state = "AVOID"
        else:
            mgr_state = "MISSION"
        self.pub_state.publish(String(data=mgr_state))

        # enforcement (robust)
        if not bool(self.get_parameter("enforce_mode").value):
            return

        now = _now_s()
        enforce_period = float(self.get_parameter("enforce_period_s").value)
        if enforce_period <= 0:
            enforce_period = 1.5

        if (now - self.st.last_enforce_t) < enforce_period:
            return
        self.st.last_enforce_t = now

        target = self._desired_mode(mgr_state)
        cur_mode = _norm_mode(self.st.mavros_mode)

        # pending timeout handling
        pending_timeout = float(self.get_parameter("pending_timeout_s").value)
        if self.st.pending_mode is not None and pending_timeout > 0:
            if (now - self.st.pending_since) > pending_timeout:
                self._emit_event("PENDING_TIMEOUT", {"pending_mode": self.st.pending_mode})
                self.st.pending_mode = None  # allow retry

        # if already match, nothing
        if cur_mode == _norm_mode(target):
            return

        # enforce by retrying set_mode
        self._emit_event("ENFORCE", {"mgr_state": mgr_state, "target": target, "current": cur_mode})
        self._request_mode(target, cause=f"enforce_{mgr_state.lower()}")

    def _desired_mode(self, mgr_state: str) -> str:
        if mgr_state == "FAILSAFE":
            return str(self.get_parameter("failsafe_mode").value)
        if mgr_state == "AVOID":
            return str(self.get_parameter("avoid_mode").value)
        # MISSION
        # Prefer last saved restore target if available; else default
        return self._current_mission_restore_target()

    def _current_mission_restore_target(self) -> str:
        # kalau mode sekarang sudah AUTO/GUIDED/MISSION type lain, kita simpan itu sebagai restore.
        # kalau mode sekarang MANUAL dan kita butuh restore mission, pakai mission_mode_default.
        cur = _norm_mode(self.st.mavros_mode)
        mission_default = str(self.get_parameter("mission_mode_default").value)

        if cur and cur not in ("MANUAL", "STABILIZE"):
            return cur
        return mission_default

    def _request_mode(self, mode: str, cause: str) -> None:
        mode = _norm_mode(mode)
        if not mode:
            return

        # connected check
        if not self.st.mavros_connected:
            self._emit_event(
                "MODE_REQ_SKIPPED", {"mode": mode, "cause": cause, "reason": "mavros_not_connected"}
            )
            return

        # service ready?
        if not self.cli_set_mode.service_is_ready():
            self._emit_event(
                "MODE_REQ_SKIPPED",
                {"mode": mode, "cause": cause, "reason": "set_mode_srv_not_ready"},
            )
            return

        # rate limit (anti spam)
        min_dt = float(self.get_parameter("min_mode_switch_interval_s").value)
        now = _now_s()
        if (now - self.st.last_mode_req_t) < max(0.0, min_dt):
            self._emit_event(
                "MODE_REQ_SKIPPED", {"mode": mode, "cause": cause, "reason": "rate_limited"}
            )
            return

        # avoid duplicate pending
        if self.st.pending_mode is not None:
            self._emit_event(
                "MODE_REQ_SKIPPED", {"mode": mode, "cause": cause, "reason": "pending_exists"}
            )
            return

        # already in mode
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

        def _done_cb(f) -> None:
            ok = False
            detail = ""
            try:
                resp = f.result()
                ok = bool(resp.mode_sent)
                detail = "mode_sent=true" if ok else "mode_sent=false"
            except Exception as e:
                ok = False
                detail = f"exception:{type(e).__name__}"
            self._emit_event("MODE_REQ_DONE", {"mode": mode, "ok": ok, "detail": detail})
            self.st.pending_mode = None

        fut.add_done_callback(_done_cb)

    def _emit_event(self, name: str, payload: dict) -> None:
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
