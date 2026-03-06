#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 Metrics Extractor (ROS 2 Humble) — SEANO CA

Mengambil metrik dari rosbag2 (sqlite3) menggunakan rosbag2_py:
- Reaction time  : command_safe hazard -> rc_override_enable true
- Release time   : command_safe clear   -> rc_override_enable false
- Durasi takeover (rc_override_enable true segment)
- Jumlah trigger failsafe (rising edge /ca/failsafe_active)
- Ringkasan mismatch: saat mode_manager_state=MISSION tapi mavros mode != AUTO
- REJOIN metrics:
  - REJOIN_START -> REJOIN_DONE
  - jumlah rejoin done / cancelled / timeout
  - fallback ke state transition REJOIN -> MISSION bila event topic tidak ada

Cara pakai:
  source /opt/ros/humble/setup.bash
  source ~/seano-collision-avoidance/seano_ca_ws/install/setup.bash
  python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_metrics_from_bag.py \
      --bag ~/bags/<bag_folder>

Butuh topic minimal dalam bag:
- /ca/command_safe         (std_msgs/String)
- /seano/rc_override_enable (std_msgs/Bool)
- /ca/failsafe_active      (std_msgs/Bool)
- /mavros/state            (mavros_msgs/State)

Opsional tapi sangat disarankan:
- /ca/mode_manager_state   (std_msgs/String)
- /ca/mode_manager_event   (std_msgs/String, JSON event)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def norm_cmd(s: str) -> str:
    return str(s or "").strip().upper()


def norm_mode(s: str) -> str:
    return str(s or "").strip().upper().replace("-", "_").replace(" ", "_")


def is_hazard_cmd(cmd: str) -> bool:
    c = norm_cmd(cmd)
    if not c:
        return False
    return c not in ("HOLD_COURSE", "HOLD", "OK")


def ns_to_s(ns: int) -> float:
    return float(ns) * 1e-9


@dataclass
class EdgeCount:
    rises: int = 0
    falls: int = 0


@dataclass
class Segment:
    t_on: float
    t_off: float


@dataclass
class EventSample:
    t: float
    event: str
    payload: Dict[str, Any]


def load_bag(bag_dir: str) -> Tuple[SequentialReader, Dict[str, str]]:
    if not os.path.isdir(bag_dir):
        raise FileNotFoundError(f"Bag dir not found: {bag_dir}")

    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="",
        output_serialization_format="",
    )
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    type_map: Dict[str, str] = {t.name: t.type for t in topics}
    return reader, type_map


def stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": float(len(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "mean": float(sum(xs) / len(xs)),
    }


def extract_rejoin_from_events(event_samples: List[EventSample]) -> Dict[str, Any]:
    pending_start: Optional[float] = None
    durations: List[float] = []
    done_count = 0
    cancelled_count = 0
    timeout_count = 0

    for ev in event_samples:
        name = norm_cmd(ev.event)

        if name == "REJOIN_START":
            pending_start = ev.t
            continue

        if name == "REJOIN_DONE":
            if pending_start is not None and ev.t >= pending_start:
                durations.append(ev.t - pending_start)
                done_count += 1
            pending_start = None
            continue

        if name == "REJOIN_CANCELLED":
            cancelled_count += 1
            pending_start = None
            continue

        if name == "REJOIN_TIMEOUT":
            timeout_count += 1
            pending_start = None
            continue

    return {
        "source": "mode_manager_event",
        "segments": int(done_count),
        "done": int(done_count),
        "cancelled": int(cancelled_count),
        "timeouts": int(timeout_count),
        "duration_s": stats(durations),
    }


def extract_rejoin_from_state(mgr_state_samples: List[Tuple[float, str]]) -> Dict[str, Any]:
    pending_start: Optional[float] = None
    durations: List[float] = []
    done_count = 0
    aborted_count = 0

    prev_state: Optional[str] = None
    for t, st in mgr_state_samples:
        stn = norm_cmd(st)

        if prev_state is None:
            prev_state = stn
            if stn == "REJOIN":
                pending_start = t
            continue

        if stn == "REJOIN" and prev_state != "REJOIN":
            pending_start = t

        elif prev_state == "REJOIN" and stn != "REJOIN":
            if pending_start is not None:
                if stn == "MISSION":
                    durations.append(t - pending_start)
                    done_count += 1
                else:
                    aborted_count += 1
            pending_start = None

        prev_state = stn

    return {
        "source": "mode_manager_state_fallback",
        "segments": int(done_count),
        "done": int(done_count),
        "cancelled": int(aborted_count),
        "timeouts": 0,
        "duration_s": stats(durations),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path folder rosbag2 (directory, not .db3)")
    ap.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: <bag>/phase6_metrics.json)",
    )
    ap.add_argument("--max_messages", type=int, default=0, help="Limit messages (0=all)")
    args = ap.parse_args()

    bag_dir = os.path.expanduser(args.bag)
    out_path = (
        os.path.expanduser(args.out) if args.out else os.path.join(bag_dir, "phase6_metrics.json")
    )

    rclpy.init(args=None)
    reader, type_map = load_bag(bag_dir)

    TOP_CMD = "/ca/command_safe"
    TOP_OVR = "/seano/rc_override_enable"
    TOP_FAIL = "/ca/failsafe_active"
    TOP_MAV = "/mavros/state"
    TOP_MGR = "/ca/mode_manager_state"
    TOP_EVT = "/ca/mode_manager_event"

    needed = [TOP_CMD, TOP_OVR, TOP_FAIL, TOP_MAV]
    missing = [t for t in needed if t not in type_map]
    if missing:
        print(f"[WARN] Bag missing required topics: {missing}")

    msg_cls: Dict[str, Any] = {}
    for t in [TOP_CMD, TOP_OVR, TOP_FAIL, TOP_MAV, TOP_MGR, TOP_EVT]:
        if t in type_map:
            msg_cls[t] = get_message(type_map[t])

    cmd_samples: List[Tuple[float, str]] = []
    ovr_samples: List[Tuple[float, bool]] = []
    fs_samples: List[Tuple[float, bool]] = []
    mav_mode_samples: List[Tuple[float, str]] = []
    mgr_state_samples: List[Tuple[float, str]] = []
    mgr_event_samples: List[EventSample] = []

    count = 0
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        count += 1
        if args.max_messages and count > args.max_messages:
            break
        if topic not in msg_cls:
            continue

        t = ns_to_s(int(t_ns))
        msg = deserialize_message(data, msg_cls[topic])

        if topic == TOP_CMD:
            cmd_samples.append((t, str(msg.data)))

        elif topic == TOP_OVR:
            ovr_samples.append((t, bool(msg.data)))

        elif topic == TOP_FAIL:
            fs_samples.append((t, bool(msg.data)))

        elif topic == TOP_MAV:
            mav_mode_samples.append((t, str(msg.mode)))

        elif topic == TOP_MGR:
            mgr_state_samples.append((t, str(msg.data)))

        elif topic == TOP_EVT:
            try:
                evt = json.loads(str(msg.data))
                mgr_event_samples.append(
                    EventSample(
                        t=t,
                        event=str(evt.get("event", "")),
                        payload=(
                            dict(evt.get("payload", {}))
                            if isinstance(evt.get("payload", {}), dict)
                            else {}
                        ),
                    )
                )
            except Exception:
                # event rusak tidak membatalkan ekstraksi metrik lain
                pass

    # ------------------------------------------------------------------
    # Override segments
    # ------------------------------------------------------------------
    takeover_segments: List[Segment] = []
    ovr_edges = EdgeCount()
    prev_ovr: Optional[bool] = None
    t_on: Optional[float] = None

    for t, v in ovr_samples:
        if prev_ovr is None:
            prev_ovr = v
            if v:
                t_on = t
            continue

        if (not prev_ovr) and v:
            ovr_edges.rises += 1
            t_on = t

        if prev_ovr and (not v):
            ovr_edges.falls += 1
            if t_on is not None:
                takeover_segments.append(Segment(t_on=t_on, t_off=t))
            t_on = None

        prev_ovr = v

    if t_on is not None and ovr_samples:
        takeover_segments.append(Segment(t_on=t_on, t_off=ovr_samples[-1][0]))

    takeover_durations = [seg.t_off - seg.t_on for seg in takeover_segments]

    # ------------------------------------------------------------------
    # Failsafe edges
    # ------------------------------------------------------------------
    fs_edges = EdgeCount()
    prev_fs: Optional[bool] = None
    for _, v in fs_samples:
        if prev_fs is None:
            prev_fs = v
            continue
        if (not prev_fs) and v:
            fs_edges.rises += 1
        if prev_fs and (not v):
            fs_edges.falls += 1
        prev_fs = v

    # ------------------------------------------------------------------
    # Reaction / release time
    # ------------------------------------------------------------------
    ovr_on_times = [seg.t_on for seg in takeover_segments]
    ovr_off_times = [seg.t_off for seg in takeover_segments]

    reaction_times: List[float] = []
    release_times: List[float] = []

    prev_cmd: Optional[str] = None
    for t, cmd in cmd_samples:
        c = norm_cmd(cmd)
        if prev_cmd is None:
            prev_cmd = c
            continue

        if c == prev_cmd:
            continue

        if is_hazard_cmd(c) and (not is_hazard_cmd(prev_cmd)):
            dt = None
            for ton in ovr_on_times:
                if ton >= t:
                    dt = ton - t
                    break
            if dt is not None:
                reaction_times.append(dt)

        if (not is_hazard_cmd(c)) and is_hazard_cmd(prev_cmd):
            dt = None
            for toff in ovr_off_times:
                if toff >= t:
                    dt = toff - t
                    break
            if dt is not None:
                release_times.append(dt)

        prev_cmd = c

    # ------------------------------------------------------------------
    # MISSION vs MAVROS mode mismatch
    # ------------------------------------------------------------------
    mismatch = 0
    mission_samples = 0
    if mgr_state_samples and mav_mode_samples:
        i = 0
        cur_mode = norm_mode(mav_mode_samples[0][1]) if mav_mode_samples else "UNKNOWN"

        for t_mgr, st in mgr_state_samples:
            stn = norm_cmd(st)
            while i + 1 < len(mav_mode_samples) and mav_mode_samples[i + 1][0] <= t_mgr:
                i += 1
                cur_mode = norm_mode(mav_mode_samples[i][1])

            if stn == "MISSION":
                mission_samples += 1
                if cur_mode != "AUTO":
                    mismatch += 1

    # ------------------------------------------------------------------
    # REJOIN metrics
    # Prioritas:
    # 1) /ca/mode_manager_event
    # 2) fallback /ca/mode_manager_state
    # ------------------------------------------------------------------
    if mgr_event_samples:
        rejoin = extract_rejoin_from_events(mgr_event_samples)
    elif mgr_state_samples:
        rejoin = extract_rejoin_from_state(mgr_state_samples)
    else:
        rejoin = {
            "source": "unavailable",
            "segments": 0,
            "done": 0,
            "cancelled": 0,
            "timeouts": 0,
            "duration_s": stats([]),
        }

    out: Dict[str, Any] = {
        "bag": bag_dir,
        "counts": {
            "cmd_samples": len(cmd_samples),
            "override_samples": len(ovr_samples),
            "failsafe_samples": len(fs_samples),
            "mavros_state_samples": len(mav_mode_samples),
            "mode_manager_samples": len(mgr_state_samples),
            "mode_manager_event_samples": len(mgr_event_samples),
        },
        "override": {
            "takeover_segments": len(takeover_segments),
            "rises": ovr_edges.rises,
            "falls": ovr_edges.falls,
            "duration_s": stats(takeover_durations),
        },
        "failsafe": {
            "rises": fs_edges.rises,
            "falls": fs_edges.falls,
        },
        "reaction_time_s": stats(reaction_times),
        "release_time_s": stats(release_times),
        "rejoin": rejoin,
        "rejoin_time_s": rejoin["duration_s"],
        "mode_mismatch": {
            "mission_samples": mission_samples,
            "mismatch_samples": mismatch,
            "mismatch_ratio": (float(mismatch) / mission_samples) if mission_samples > 0 else 0.0,
        },
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Phase 6 Metrics Summary ===")
    print(f"bag: {bag_dir}")
    print(
        f"takeover segments: {out['override']['takeover_segments']} "
        f"dur_mean={out['override']['duration_s']['mean']:.3f}s"
    )
    print(
        f"reaction mean: {out['reaction_time_s']['mean']:.3f}s "
        f"(n={int(out['reaction_time_s']['n'])})"
    )
    print(
        f"release mean: {out['release_time_s']['mean']:.3f}s "
        f"(n={int(out['release_time_s']['n'])})"
    )
    print(
        f"rejoin mean: {out['rejoin_time_s']['mean']:.3f}s "
        f"(n={int(out['rejoin_time_s']['n'])}, source={out['rejoin']['source']})"
    )
    print(
        f"rejoin detail: done={out['rejoin']['done']} "
        f"cancelled={out['rejoin']['cancelled']} "
        f"timeouts={out['rejoin']['timeouts']}"
    )
    print(f"failsafe rises: {out['failsafe']['rises']}")
    print(
        f"mission-mode mismatch ratio: {out['mode_mismatch']['mismatch_ratio']:.3f} "
        f"(mismatch {mismatch}/{mission_samples})"
    )
    print(f"saved: {out_path}\n")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
