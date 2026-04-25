#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEANO Collision Avoidance Evaluation Logger

Runtime evidence logger for Phase 5 and Phase 7.
It saves HUD frames from /ca/debug_image only when:
- /ca/mode_manager_state changes
- /ca/command_safe changes

It also writes evaluation metrics:
- events.csv / events.jsonl
- avoidance_cycles.csv
- metrics_summary.csv / metrics_summary.json
- time_series.csv
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String

try:
    from mavros_msgs.msg import OverrideRCIn, State as MavrosState

    HAS_MAVROS = True
except Exception:
    OverrideRCIn = None
    MavrosState = None
    HAS_MAVROS = False

EVENT_FIELDS = [
    "seq",
    "event_id",
    "wall_time_iso",
    "ros_time_sec",
    "trigger",
    "value",
    "avoid_state",
    "command_safe",
    "risk",
    "mode_event",
    "mavros_connected",
    "mavros_mode",
    "auto_enable",
    "rc_override_enable",
    "left_cmd",
    "right_cmd",
    "selected_left_cmd",
    "selected_right_cmd",
    "auto_left_cmd",
    "auto_right_cmd",
    "image_topic",
    "image_saved",
    "image_path",
    "image_age_s",
    "image_stamp_sec",
    "notes",
]

CYCLE_FIELDS = [
    "episode_id",
    "start_wall_time_iso",
    "end_wall_time_iso",
    "start_ros_time_sec",
    "end_ros_time_sec",
    "start_reason",
    "completion_reason",
    "completed",
    "success",
    "total_cycle_duration_s",
    "first_risk_time_sec",
    "first_hazard_command_time_sec",
    "avoid_start_time_sec",
    "rejoin_start_time_sec",
    "mission_return_time_sec",
    "reaction_risk_to_command_s",
    "reaction_risk_to_avoid_s",
    "command_to_avoid_s",
    "avoid_to_rejoin_s",
    "rejoin_to_mission_s",
    "rc_override_duration_s",
    "max_risk",
    "mean_risk",
    "risk_auc",
    "risk_time_high_s",
    "risk_sample_count",
    "command_counts_json",
    "command_switches",
    "max_abs_left_cmd",
    "max_abs_right_cmd",
    "max_abs_diff_cmd",
    "frame_events_saved",
    "frame_events_failed",
    "notes",
]

TS_FIELDS = [
    "wall_time_iso",
    "ros_time_sec",
    "avoid_state",
    "command_safe",
    "risk",
    "mode_event",
    "mavros_connected",
    "mavros_mode",
    "auto_enable",
    "rc_override_enable",
    "left_cmd",
    "right_cmd",
    "selected_left_cmd",
    "selected_right_cmd",
    "auto_left_cmd",
    "auto_right_cmd",
]


@dataclass
class LatestImage:
    msg: Optional[Image] = None
    recv_wall_time: float = 0.0
    stamp_sec: float = 0.0


@dataclass
class PendingEvent:
    seq: int
    event_id: str
    trigger: str
    value: str
    created_wall_time: float
    due_wall_time: float
    ros_time_sec: float
    snapshot: Dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def stamp_to_sec(stamp: Any) -> float:
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    except Exception:
        return 0.0


def safe_name(text: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip()).strip("._-")
    return (s if s else "UNKNOWN")[:max_len]


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        if value is None:
            return ""
        x = float(value)
        if not math.isfinite(x):
            return ""
        return f"{x:.{digits}f}"
    except Exception:
        return ""


def fmt_bool(value: Any) -> str:
    if value is None:
        return ""
    return "true" if bool(value) else "false"


def duration(start: Any, end: Any) -> str:
    try:
        if start is None or end is None:
            return ""
        return f"{max(0.0, float(end) - float(start)):.4f}"
    except Exception:
        return ""


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


class EventLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("event_logger_node")
        self._declare_params()

        self.log_root = os.path.expanduser(str(self.get_parameter("log_root").value))
        run_id = str(self.get_parameter("run_id").value).strip() or datetime.now().strftime(
            "seano_ca_eval_%Y%m%d_%H%M%S"
        )
        self.run_id = safe_name(run_id, 96)
        self.run_dir = os.path.join(self.log_root, self.run_id)
        self.frame_dir = os.path.join(self.run_dir, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)

        self.events_csv = os.path.join(self.run_dir, "events.csv")
        self.events_jsonl = os.path.join(self.run_dir, "events.jsonl")
        self.cycles_csv = os.path.join(self.run_dir, "avoidance_cycles.csv")
        self.summary_csv = os.path.join(self.run_dir, "metrics_summary.csv")
        self.summary_json = os.path.join(self.run_dir, "metrics_summary.json")
        self.timeseries_csv = os.path.join(self.run_dir, "time_series.csv")

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.frame_max_age_s = float(self.get_parameter("frame_max_age_s").value)
        self.capture_delay_s = float(self.get_parameter("capture_delay_s").value)
        self.jpeg_quality = max(1, min(100, int(self.get_parameter("jpeg_quality").value)))
        self.risk_enter_threshold = float(self.get_parameter("risk_enter_threshold").value)
        self.risk_clear_threshold = float(self.get_parameter("risk_clear_threshold").value)
        self.timeseries_period_s = max(0.1, float(self.get_parameter("timeseries_period_s").value))
        self.idle_close_s = max(0.5, float(self.get_parameter("idle_close_s").value))
        self.hazard_commands = {
            x.strip().upper()
            for x in str(self.get_parameter("hazard_commands").value).split(",")
            if x.strip()
        }

        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.latest_image = LatestImage()
        self.pending_events: List[PendingEvent] = []

        self.seq = 0
        self.episode_seq = 0
        self.run_start_wall = time.time()
        self.run_start_ros = self.ros_time_sec()
        self.last_activity_ros = self.run_start_ros

        self.state: Dict[str, Any] = {
            "avoid_state": "UNKNOWN",
            "command_safe": "UNKNOWN",
            "risk": None,
            "mode_event": "",
            "mavros_connected": None,
            "mavros_mode": "",
            "auto_enable": None,
            "rc_override_enable": None,
            "left_cmd": None,
            "right_cmd": None,
            "selected_left_cmd": None,
            "selected_right_cmd": None,
            "auto_left_cmd": None,
            "auto_right_cmd": None,
        }

        self.last_logged_avoid_state: Optional[str] = None
        self.last_logged_command_safe: Optional[str] = None
        self.last_command_safe: Optional[str] = None
        self.last_risk_value: Optional[float] = None
        self.last_risk_time: Optional[float] = None
        self.event_counts: Dict[str, int] = {}
        self.command_counts_total: Dict[str, int] = {}
        self.completed_cycles: List[Dict[str, Any]] = []
        self.current_cycle: Optional[Dict[str, Any]] = None

        self.init_outputs()
        self.setup_subscriptions()

        self.create_timer(0.02, self.process_pending_events)
        self.create_timer(self.timeseries_period_s, self.write_timeseries_row)
        self.create_timer(1.0, self.periodic_housekeeping)

        self.get_logger().info(
            "CA evaluation logger ready | "
            f"run_dir={self.run_dir} image_topic={self.image_topic} "
            f"triggers=avoid_state,command_safe risk_enter={self.risk_enter_threshold:.3f}"
        )

    def _declare_params(self) -> None:
        self.declare_parameter("log_root", "~/seano_event_logs")
        self.declare_parameter("run_id", "")
        self.declare_parameter("image_topic", "/ca/debug_image")
        self.declare_parameter("avoid_state_topic", "/ca/mode_manager_state")
        self.declare_parameter("command_safe_topic", "/ca/command_safe")
        self.declare_parameter("risk_topic", "/ca/risk")
        self.declare_parameter("mode_event_topic", "/ca/mode_manager_event")
        self.declare_parameter("auto_enable_topic", "/seano/auto_enable")
        self.declare_parameter("rc_override_enable_topic", "/seano/rc_override_enable")
        self.declare_parameter("left_cmd_topic", "/seano/left_cmd")
        self.declare_parameter("right_cmd_topic", "/seano/right_cmd")
        self.declare_parameter("selected_left_cmd_topic", "/seano/selected/left_cmd")
        self.declare_parameter("selected_right_cmd_topic", "/seano/selected/right_cmd")
        self.declare_parameter("auto_left_cmd_topic", "/seano/auto/left_cmd")
        self.declare_parameter("auto_right_cmd_topic", "/seano/auto/right_cmd")
        self.declare_parameter("mavros_state_topic", "/mavros/state")
        self.declare_parameter("mavros_rc_override_topic", "/mavros/rc/override")
        self.declare_parameter("frame_max_age_s", 10.0)
        self.declare_parameter("capture_delay_s", 0.35)
        self.declare_parameter("jpeg_quality", 92)
        self.declare_parameter("risk_enter_threshold", 0.20)
        self.declare_parameter("risk_clear_threshold", 0.10)
        self.declare_parameter("hazard_commands", "STOP,TURN_LEFT,TURN_RIGHT")
        self.declare_parameter("timeseries_period_s", 0.5)
        self.declare_parameter("idle_close_s", 2.0)

    def reliable_qos(self) -> QoSProfile:
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

    def image_qos(self) -> QoSProfile:
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

    def init_outputs(self) -> None:
        with open(self.events_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EVENT_FIELDS).writeheader()
        with open(self.cycles_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CYCLE_FIELDS).writeheader()
        with open(self.timeseries_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TS_FIELDS).writeheader()
        with open(self.events_jsonl, "w", encoding="utf-8") as f:
            f.write("")
        self.write_summary_files()

    def setup_subscriptions(self) -> None:
        q = self.reliable_qos()
        self.create_subscription(Image, self.image_topic, self.on_image, self.image_qos())
        self.create_subscription(
            String, str(self.get_parameter("avoid_state_topic").value), self.on_avoid_state, q
        )
        self.create_subscription(
            String, str(self.get_parameter("command_safe_topic").value), self.on_command_safe, q
        )
        self.create_subscription(
            Float32, str(self.get_parameter("risk_topic").value), self.on_risk, q
        )
        self.create_subscription(
            String, str(self.get_parameter("mode_event_topic").value), self.on_mode_event, q
        )
        self.create_subscription(
            Bool, str(self.get_parameter("auto_enable_topic").value), self.on_auto_enable, q
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("rc_override_enable_topic").value),
            self.on_rc_override_enable,
            q,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("left_cmd_topic").value),
            lambda m: self.update_float("left_cmd", m.data),
            q,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("right_cmd_topic").value),
            lambda m: self.update_float("right_cmd", m.data),
            q,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("selected_left_cmd_topic").value),
            lambda m: self.update_float("selected_left_cmd", m.data),
            q,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("selected_right_cmd_topic").value),
            lambda m: self.update_float("selected_right_cmd", m.data),
            q,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("auto_left_cmd_topic").value),
            lambda m: self.update_float("auto_left_cmd", m.data),
            q,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("auto_right_cmd_topic").value),
            lambda m: self.update_float("auto_right_cmd", m.data),
            q,
        )
        if HAS_MAVROS and MavrosState is not None:
            self.create_subscription(
                MavrosState,
                str(self.get_parameter("mavros_state_topic").value),
                self.on_mavros_state,
                q,
            )
        if HAS_MAVROS and OverrideRCIn is not None:
            self.create_subscription(
                OverrideRCIn,
                str(self.get_parameter("mavros_rc_override_topic").value),
                self.on_mavros_rc_override,
                q,
            )

    def ros_time_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1.0e-9

    def snapshot_state(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def on_image(self, msg: Image) -> None:
        with self.lock:
            self.latest_image = LatestImage(
                msg=msg, recv_wall_time=time.time(), stamp_sec=stamp_to_sec(msg.header.stamp)
            )

    def on_mode_event(self, msg: String) -> None:
        with self.lock:
            self.state["mode_event"] = str(msg.data)

    def on_auto_enable(self, msg: Bool) -> None:
        with self.lock:
            self.state["auto_enable"] = bool(msg.data)

    def on_mavros_state(self, msg: Any) -> None:
        with self.lock:
            self.state["mavros_connected"] = bool(getattr(msg, "connected", False))
            self.state["mavros_mode"] = str(getattr(msg, "mode", ""))

    def on_mavros_rc_override(self, msg: Any) -> None:
        try:
            active = any(int(v) != 0 for v in list(msg.channels))
        except Exception:
            active = False
        self.update_rc_override_activity(active, self.ros_time_sec())

    def on_rc_override_enable(self, msg: Bool) -> None:
        active = bool(msg.data)
        with self.lock:
            self.state["rc_override_enable"] = active
        self.update_rc_override_activity(active, self.ros_time_sec())

    def update_float(self, name: str, value: Any) -> None:
        try:
            x = float(value)
            if not math.isfinite(x):
                return
        except Exception:
            return
        with self.lock:
            self.state[name] = x
            c = self.current_cycle
            if c is not None:
                left = self.state.get("left_cmd")
                right = self.state.get("right_cmd")
                if left is not None:
                    c["max_abs_left_cmd"] = max(c["max_abs_left_cmd"], abs(float(left)))
                if right is not None:
                    c["max_abs_right_cmd"] = max(c["max_abs_right_cmd"], abs(float(right)))
                if left is not None and right is not None:
                    c["max_abs_diff_cmd"] = max(
                        c["max_abs_diff_cmd"], abs(float(right) - float(left))
                    )

    def on_risk(self, msg: Float32) -> None:
        now = self.ros_time_sec()
        risk = float(msg.data)
        if not math.isfinite(risk):
            return
        with self.lock:
            self.state["risk"] = risk
        if risk >= self.risk_enter_threshold:
            self.ensure_cycle("risk_threshold", now)
        with self.lock:
            c = self.current_cycle
            if c is not None:
                if c["first_risk_time_sec"] is None and risk >= self.risk_enter_threshold:
                    c["first_risk_time_sec"] = now
                if self.last_risk_value is not None and self.last_risk_time is not None:
                    dt = now - self.last_risk_time
                    if 0.0 <= dt <= 5.0:
                        c["risk_auc"] += float(self.last_risk_value) * dt
                        if float(self.last_risk_value) >= self.risk_enter_threshold:
                            c["risk_time_high_s"] += dt
                c["max_risk"] = max(c["max_risk"], risk)
                c["risk_sum"] += risk
                c["risk_sample_count"] += 1
            self.last_risk_value = risk
            self.last_risk_time = now

    def on_avoid_state(self, msg: String) -> None:
        now = self.ros_time_sec()
        value = str(msg.data).strip() or "UNKNOWN"
        do_log = False
        with self.lock:
            self.state["avoid_state"] = value
            if self.last_logged_avoid_state != value:
                self.last_logged_avoid_state = value
                do_log = True
        upper = value.upper()
        if upper == "AVOID":
            self.ensure_cycle("avoid_state_AVOID", now)
            with self.lock:
                if self.current_cycle is not None:
                    self.current_cycle["avoid_start_time_sec"] = (
                        self.current_cycle["avoid_start_time_sec"] or now
                    )
                    self.current_cycle["entered_avoid"] = True
                    self.last_activity_ros = now
        elif upper == "REJOIN":
            self.ensure_cycle("avoid_state_REJOIN", now)
            with self.lock:
                if self.current_cycle is not None:
                    self.current_cycle["rejoin_start_time_sec"] = (
                        self.current_cycle["rejoin_start_time_sec"] or now
                    )
                    self.current_cycle["entered_rejoin"] = True
                    self.last_activity_ros = now
        elif upper == "MISSION":
            with self.lock:
                if self.current_cycle is not None and (
                    self.current_cycle["entered_avoid"] or self.current_cycle["entered_rejoin"]
                ):
                    self.current_cycle["mission_return_time_sec"] = (
                        self.current_cycle["mission_return_time_sec"] or now
                    )
                    self.current_cycle["entered_mission_return"] = True
                    self.last_activity_ros = now
            self.close_cycle_if_success("mission_return")
        if do_log:
            self.queue_event("avoid_state", value)

    def on_command_safe(self, msg: String) -> None:
        now = self.ros_time_sec()
        value = str(msg.data).strip() or "UNKNOWN"
        upper = value.upper()
        do_log = False
        with self.lock:
            self.state["command_safe"] = value
            self.command_counts_total[value] = self.command_counts_total.get(value, 0) + 1
            if self.last_logged_command_safe != value:
                self.last_logged_command_safe = value
                do_log = True
        if upper in self.hazard_commands:
            self.ensure_cycle(f"command_{upper}", now)
            with self.lock:
                if self.current_cycle is not None:
                    if self.current_cycle["first_hazard_command_time_sec"] is None:
                        self.current_cycle["first_hazard_command_time_sec"] = now
                    self.current_cycle["last_hazard_command_time_sec"] = now
                    self.last_activity_ros = now
        with self.lock:
            c = self.current_cycle
            if c is not None:
                c["command_counts"][value] = c["command_counts"].get(value, 0) + 1
                if self.last_command_safe is not None and self.last_command_safe != value:
                    c["command_switches"] += 1
            self.last_command_safe = value
        if do_log:
            self.queue_event("command_safe", value)

    def new_cycle(self, reason: str, now: float) -> Dict[str, Any]:
        self.episode_seq += 1
        return {
            "episode_id": self.episode_seq,
            "start_wall_time_iso": now_iso(),
            "end_wall_time_iso": "",
            "start_ros_time_sec": now,
            "end_ros_time_sec": None,
            "start_reason": reason,
            "completion_reason": "",
            "completed": False,
            "success": False,
            "first_risk_time_sec": None,
            "first_hazard_command_time_sec": None,
            "last_hazard_command_time_sec": None,
            "avoid_start_time_sec": None,
            "rejoin_start_time_sec": None,
            "mission_return_time_sec": None,
            "entered_avoid": False,
            "entered_rejoin": False,
            "entered_mission_return": False,
            "max_risk": 0.0,
            "risk_sum": 0.0,
            "risk_auc": 0.0,
            "risk_time_high_s": 0.0,
            "risk_sample_count": 0,
            "command_counts": {},
            "command_switches": 0,
            "max_abs_left_cmd": 0.0,
            "max_abs_right_cmd": 0.0,
            "max_abs_diff_cmd": 0.0,
            "rc_override_active": False,
            "rc_override_on_time_sec": None,
            "rc_override_duration_s": 0.0,
            "frame_events_saved": 0,
            "frame_events_failed": 0,
            "notes": "",
        }

    def ensure_cycle(self, reason: str, now: float) -> None:
        with self.lock:
            if self.current_cycle is None:
                self.current_cycle = self.new_cycle(reason, now)
            self.last_activity_ros = now

    def update_rc_override_activity(self, active: bool, now: float) -> None:
        with self.lock:
            c = self.current_cycle
            if c is None:
                return
            old = bool(c["rc_override_active"])
            if active and not old:
                c["rc_override_active"] = True
                c["rc_override_on_time_sec"] = now
            elif old and not active:
                start = c.get("rc_override_on_time_sec")
                if start is not None:
                    c["rc_override_duration_s"] += max(0.0, now - float(start))
                c["rc_override_active"] = False
                c["rc_override_on_time_sec"] = None

    def close_cycle_if_success(self, reason: str) -> None:
        with self.lock:
            c = self.current_cycle
            if c is None:
                return
            success = c["entered_avoid"] and c["entered_rejoin"] and c["entered_mission_return"]
        if success:
            self.close_cycle(reason, completed=True)

    def maybe_close_idle_cycle(self) -> None:
        now = self.ros_time_sec()
        with self.lock:
            c = self.current_cycle
            if c is None or c["entered_avoid"]:
                return
            idle_s = now - self.last_activity_ros
            risk = self.state.get("risk")
            cmd = str(self.state.get("command_safe", "UNKNOWN")).upper()
            state = str(self.state.get("avoid_state", "UNKNOWN")).upper()
        risk_clear = risk is None or float(risk) <= self.risk_clear_threshold
        if (
            idle_s >= self.idle_close_s
            and risk_clear
            and cmd in {"HOLD_COURSE", "UNKNOWN"}
            and state in {"MISSION", "UNKNOWN"}
        ):
            self.close_cycle("idle_clear_no_avoid", completed=True)

    def close_cycle(self, reason: str, completed: bool) -> None:
        now = self.ros_time_sec()
        with self.lock:
            c = self.current_cycle
            if c is None:
                return
            if c["rc_override_active"] and c["rc_override_on_time_sec"] is not None:
                c["rc_override_duration_s"] += max(0.0, now - float(c["rc_override_on_time_sec"]))
            c["end_wall_time_iso"] = now_iso()
            c["end_ros_time_sec"] = now
            c["completion_reason"] = reason
            c["completed"] = bool(completed)
            c["success"] = bool(
                c["entered_avoid"] and c["entered_rejoin"] and c["entered_mission_return"]
            )
            row = self.cycle_to_row(c)
            self.completed_cycles.append(dict(c))
            self.current_cycle = None
        with open(self.cycles_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CYCLE_FIELDS).writerow(row)
        self.write_summary_files()
        self.get_logger().info(
            f"cycle | id={row['episode_id']} success={row['success']} "
            f"duration={row['total_cycle_duration_s']} max_risk={row['max_risk']} reason={reason}"
        )

    def cycle_to_row(self, c: Dict[str, Any]) -> Dict[str, str]:
        n = int(c.get("risk_sample_count", 0))
        mean_risk = ""
        if n > 0:
            mean_risk = f"{float(c.get('risk_sum', 0.0)) / n:.4f}"
        return {
            "episode_id": str(c.get("episode_id", "")),
            "start_wall_time_iso": str(c.get("start_wall_time_iso", "")),
            "end_wall_time_iso": str(c.get("end_wall_time_iso", "")),
            "start_ros_time_sec": fmt_float(c.get("start_ros_time_sec"), 6),
            "end_ros_time_sec": fmt_float(c.get("end_ros_time_sec"), 6),
            "start_reason": str(c.get("start_reason", "")),
            "completion_reason": str(c.get("completion_reason", "")),
            "completed": fmt_bool(c.get("completed")),
            "success": fmt_bool(c.get("success")),
            "total_cycle_duration_s": duration(
                c.get("start_ros_time_sec"), c.get("end_ros_time_sec")
            ),
            "first_risk_time_sec": fmt_float(c.get("first_risk_time_sec"), 6),
            "first_hazard_command_time_sec": fmt_float(c.get("first_hazard_command_time_sec"), 6),
            "avoid_start_time_sec": fmt_float(c.get("avoid_start_time_sec"), 6),
            "rejoin_start_time_sec": fmt_float(c.get("rejoin_start_time_sec"), 6),
            "mission_return_time_sec": fmt_float(c.get("mission_return_time_sec"), 6),
            "reaction_risk_to_command_s": duration(
                c.get("first_risk_time_sec"), c.get("first_hazard_command_time_sec")
            ),
            "reaction_risk_to_avoid_s": duration(
                c.get("first_risk_time_sec"), c.get("avoid_start_time_sec")
            ),
            "command_to_avoid_s": duration(
                c.get("first_hazard_command_time_sec"), c.get("avoid_start_time_sec")
            ),
            "avoid_to_rejoin_s": duration(
                c.get("avoid_start_time_sec"), c.get("rejoin_start_time_sec")
            ),
            "rejoin_to_mission_s": duration(
                c.get("rejoin_start_time_sec"), c.get("mission_return_time_sec")
            ),
            "rc_override_duration_s": fmt_float(c.get("rc_override_duration_s"), 4),
            "max_risk": fmt_float(c.get("max_risk"), 4),
            "mean_risk": mean_risk,
            "risk_auc": fmt_float(c.get("risk_auc"), 4),
            "risk_time_high_s": fmt_float(c.get("risk_time_high_s"), 4),
            "risk_sample_count": str(c.get("risk_sample_count", 0)),
            "command_counts_json": json.dumps(
                c.get("command_counts", {}), ensure_ascii=False, sort_keys=True
            ),
            "command_switches": str(c.get("command_switches", 0)),
            "max_abs_left_cmd": fmt_float(c.get("max_abs_left_cmd"), 4),
            "max_abs_right_cmd": fmt_float(c.get("max_abs_right_cmd"), 4),
            "max_abs_diff_cmd": fmt_float(c.get("max_abs_diff_cmd"), 4),
            "frame_events_saved": str(c.get("frame_events_saved", 0)),
            "frame_events_failed": str(c.get("frame_events_failed", 0)),
            "notes": str(c.get("notes", "")),
        }

    def queue_event(self, trigger: str, value: str) -> None:
        with self.lock:
            self.seq += 1
            seq = self.seq
            self.event_counts[trigger] = self.event_counts.get(trigger, 0) + 1
        event = PendingEvent(
            seq=seq,
            event_id=f"event_{seq:06d}_{safe_name(trigger, 32)}_{safe_name(value, 48)}",
            trigger=trigger,
            value=value,
            created_wall_time=time.time(),
            due_wall_time=time.time() + max(0.0, self.capture_delay_s),
            ros_time_sec=self.ros_time_sec(),
            snapshot=self.snapshot_state(),
        )
        with self.lock:
            self.pending_events.append(event)

    def process_pending_events(self) -> None:
        now = time.time()
        ready: List[PendingEvent] = []
        with self.lock:
            keep: List[PendingEvent] = []
            for ev in self.pending_events:
                if ev.due_wall_time <= now:
                    ready.append(ev)
                else:
                    keep.append(ev)
            self.pending_events = keep
        for ev in ready:
            self.write_event(ev)

    def snapshot_image(self) -> LatestImage:
        with self.lock:
            return LatestImage(
                self.latest_image.msg, self.latest_image.recv_wall_time, self.latest_image.stamp_sec
            )

    def save_hud_frame(self, ev: PendingEvent) -> Dict[str, str]:
        img = self.snapshot_image()
        if img.msg is None:
            return {
                "image_saved": "false",
                "image_path": "",
                "image_age_s": "",
                "image_stamp_sec": "",
                "notes": "no_debug_image_received",
            }
        age = time.time() - img.recv_wall_time
        if self.frame_max_age_s > 0.0 and age > self.frame_max_age_s:
            return {
                "image_saved": "false",
                "image_path": "",
                "image_age_s": f"{age:.4f}",
                "image_stamp_sec": f"{img.stamp_sec:.6f}",
                "notes": f"debug_image_stale>{self.frame_max_age_s:.3f}s",
            }
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img.msg, desired_encoding="bgr8")
            filename = f"{ev.event_id}.jpg"
            abs_path = os.path.join(self.frame_dir, filename)
            ok = cv2.imwrite(abs_path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise RuntimeError("cv2.imwrite returned false")
            return {
                "image_saved": "true",
                "image_path": os.path.relpath(abs_path, self.run_dir),
                "image_age_s": f"{age:.4f}",
                "image_stamp_sec": f"{img.stamp_sec:.6f}",
                "notes": "",
            }
        except Exception as exc:
            return {
                "image_saved": "false",
                "image_path": "",
                "image_age_s": f"{age:.4f}",
                "image_stamp_sec": f"{img.stamp_sec:.6f}",
                "notes": f"debug_image_save_failed:{exc}",
            }

    def write_event(self, ev: PendingEvent) -> None:
        info = self.save_hud_frame(ev)
        s = ev.snapshot
        row = {
            "seq": str(ev.seq),
            "event_id": ev.event_id,
            "wall_time_iso": now_iso(),
            "ros_time_sec": f"{ev.ros_time_sec:.6f}",
            "trigger": ev.trigger,
            "value": ev.value,
            "avoid_state": str(s.get("avoid_state", "")),
            "command_safe": str(s.get("command_safe", "")),
            "risk": fmt_float(s.get("risk"), 4),
            "mode_event": str(s.get("mode_event", "")),
            "mavros_connected": fmt_bool(s.get("mavros_connected")),
            "mavros_mode": str(s.get("mavros_mode", "")),
            "auto_enable": fmt_bool(s.get("auto_enable")),
            "rc_override_enable": fmt_bool(s.get("rc_override_enable")),
            "left_cmd": fmt_float(s.get("left_cmd"), 4),
            "right_cmd": fmt_float(s.get("right_cmd"), 4),
            "selected_left_cmd": fmt_float(s.get("selected_left_cmd"), 4),
            "selected_right_cmd": fmt_float(s.get("selected_right_cmd"), 4),
            "auto_left_cmd": fmt_float(s.get("auto_left_cmd"), 4),
            "auto_right_cmd": fmt_float(s.get("auto_right_cmd"), 4),
            "image_topic": self.image_topic,
            **info,
        }
        with open(self.events_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EVENT_FIELDS).writerow(row)
        with open(self.events_jsonl, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"row": row, "event": jsonable(asdict(ev))},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        with self.lock:
            if self.current_cycle is not None:
                if info["image_saved"] == "true":
                    self.current_cycle["frame_events_saved"] += 1
                else:
                    self.current_cycle["frame_events_failed"] += 1
        self.get_logger().info(
            f"event | {ev.event_id} | {ev.trigger}={ev.value} state={row['avoid_state']} "
            f"cmd={row['command_safe']} risk={row['risk']} frame={row['image_saved']}"
        )

    def write_timeseries_row(self) -> None:
        s = self.snapshot_state()
        row = {
            "wall_time_iso": now_iso(),
            "ros_time_sec": f"{self.ros_time_sec():.6f}",
            "avoid_state": str(s.get("avoid_state", "")),
            "command_safe": str(s.get("command_safe", "")),
            "risk": fmt_float(s.get("risk"), 4),
            "mode_event": str(s.get("mode_event", "")),
            "mavros_connected": fmt_bool(s.get("mavros_connected")),
            "mavros_mode": str(s.get("mavros_mode", "")),
            "auto_enable": fmt_bool(s.get("auto_enable")),
            "rc_override_enable": fmt_bool(s.get("rc_override_enable")),
            "left_cmd": fmt_float(s.get("left_cmd"), 4),
            "right_cmd": fmt_float(s.get("right_cmd"), 4),
            "selected_left_cmd": fmt_float(s.get("selected_left_cmd"), 4),
            "selected_right_cmd": fmt_float(s.get("selected_right_cmd"), 4),
            "auto_left_cmd": fmt_float(s.get("auto_left_cmd"), 4),
            "auto_right_cmd": fmt_float(s.get("auto_right_cmd"), 4),
        }
        with open(self.timeseries_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=TS_FIELDS).writerow(row)

    def periodic_housekeeping(self) -> None:
        self.maybe_close_idle_cycle()
        self.write_summary_files()

    def summary_dict(self) -> Dict[str, Any]:
        completed = list(self.completed_cycles)
        total = len(completed)
        success = sum(1 for c in completed if c.get("success"))

        def collect(field: str) -> List[float]:
            vals = []
            for c in completed:
                raw = self.cycle_to_row(c).get(field, "")
                try:
                    if raw != "":
                        vals.append(float(raw))
                except Exception:
                    pass
            return vals

        def stats(field: str) -> Dict[str, Any]:
            vals = collect(field)
            if not vals:
                return {"count": 0, "mean": None, "min": None, "max": None}
            return {
                "count": len(vals),
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }

        metrics = {}
        for field in [
            "reaction_risk_to_command_s",
            "reaction_risk_to_avoid_s",
            "command_to_avoid_s",
            "avoid_to_rejoin_s",
            "rejoin_to_mission_s",
            "total_cycle_duration_s",
            "rc_override_duration_s",
            "max_risk",
            "mean_risk",
            "risk_auc",
            "risk_time_high_s",
            "max_abs_left_cmd",
            "max_abs_right_cmd",
            "max_abs_diff_cmd",
        ]:
            metrics[field] = stats(field)
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "current_wall_iso": now_iso(),
            "run_duration_s": max(0.0, self.ros_time_sec() - self.run_start_ros),
            "total_events": self.seq,
            "event_counts": dict(self.event_counts),
            "command_counts_total": dict(self.command_counts_total),
            "total_completed_cycles": total,
            "successful_cycles": success,
            "failed_cycles": total - success,
            "success_rate": (success / total) if total > 0 else None,
            "active_cycle": jsonable(self.current_cycle),
            "metrics": metrics,
        }

    def write_summary_files(self) -> None:
        summary = self.summary_dict()
        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(jsonable(summary), f, ensure_ascii=False, indent=2)
        rows = []

        def add(metric: str, value: Any) -> None:
            if isinstance(value, float):
                value = f"{value:.6f}"
            elif value is None:
                value = ""
            rows.append({"metric": metric, "value": str(value)})

        for k in [
            "run_id",
            "run_duration_s",
            "total_events",
            "total_completed_cycles",
            "successful_cycles",
            "failed_cycles",
            "success_rate",
        ]:
            add(k, summary.get(k))
        for name, st in summary["metrics"].items():
            for sk, sv in st.items():
                add(f"{name}_{sk}", sv)
        with open(self.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)

    def finalize(self) -> None:
        while True:
            with self.lock:
                if not self.pending_events:
                    break
                ready = list(self.pending_events)
                self.pending_events = []
            for ev in ready:
                self.write_event(ev)
        with self.lock:
            active = self.current_cycle is not None
        if active:
            self.close_cycle("logger_shutdown", completed=False)
        self.write_summary_files()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EventLoggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.finalize()
        except Exception as exc:
            node.get_logger().error(f"failed to finalize logger outputs: {exc}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
