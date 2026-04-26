#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

RISK_THRESHOLD = 0.25
DIFF_THRESHOLD = 0.02

TURN_CMDS = {"TURN_RIGHT_SLOW", "TURN_LEFT_SLOW", "TURN_RIGHT", "TURN_LEFT"}
AVOID_CMDS = {
    "STOP",
    "SLOW_DOWN",
    "TURN_RIGHT_SLOW",
    "TURN_LEFT_SLOW",
    "TURN_RIGHT",
    "TURN_LEFT",
    "DODGE_RIGHT",
    "DODGE_LEFT",
}


def parse_time(s):
    if not s:
        return None
    return datetime.fromisoformat(str(s))


def fnum(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def cmd(row):
    return str(row.get("command_safe", "")).upper()


def state(row):
    return str(row.get("avoid_state", "")).upper()


def rel_time(row, t0):
    t = parse_time(row.get("wall_time_iso"))
    if t is None or t0 is None:
        return None
    return (t - t0).total_seconds()


def first_row(rows, predicate):
    for r in rows:
        if predicate(r):
            return r
    return None


def fmt(x, nd=3):
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def selected_lr(row):
    left = row.get("selected_left_cmd")
    right = row.get("selected_right_cmd")
    if left in (None, ""):
        left = row.get("left_cmd")
    if right in (None, ""):
        right = row.get("right_cmd")
    return fnum(left), fnum(right)


def analyze(run_id):
    log_dir = Path.home() / "seano_event_logs" / run_id
    event_file = log_dir / "events.jsonl"

    if not event_file.exists():
        raise FileNotFoundError(f"events.jsonl not found: {event_file}")

    rows = []
    with event_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                rows.append(obj.get("row", obj))

    if not rows:
        raise RuntimeError(f"No rows in {event_file}")

    t0 = None
    for r in rows:
        t0 = parse_time(r.get("wall_time_iso"))
        if t0:
            break

    risk_rows = [r for r in rows if fnum(r.get("risk")) >= RISK_THRESHOLD]
    avoid_rows = [r for r in rows if state(r) == "AVOID"]
    stop_rows = [r for r in rows if cmd(r) == "STOP"]
    turn_rows = [r for r in rows if cmd(r) in TURN_CMDS]

    first_risk = risk_rows[0] if risk_rows else None
    first_avoid = first_row(rows, lambda r: state(r) == "AVOID")
    first_turn = first_row(rows, lambda r: cmd(r) in TURN_CMDS)

    # More defensible observation-to-maneuver time:
    # use the last STOP immediately before the first turn, not the first STOP in the whole trial.
    last_stop_before_turn = None
    if first_turn is not None:
        turn_seq = int(first_turn.get("seq", -1) or -1)
        candidates = [
            r for r in rows if cmd(r) == "STOP" and int(r.get("seq", -1) or -1) < turn_seq
        ]
        if candidates:
            last_stop_before_turn = candidates[-1]

    recovery = None
    if first_turn is not None:
        turn_seq = int(first_turn.get("seq", -1) or -1)
        recovery = first_row(
            rows,
            lambda r: int(r.get("seq", -1) or -1) > turn_seq and state(r) in ("REJOIN", "MISSION"),
        )
    elif first_avoid is not None:
        avoid_seq = int(first_avoid.get("seq", -1) or -1)
        recovery = first_row(
            rows,
            lambda r: int(r.get("seq", -1) or -1) > avoid_seq and state(r) in ("REJOIN", "MISSION"),
        )

    max_risk = max((fnum(r.get("risk")) for r in rows), default=0.0)

    max_delta = 0.0
    max_left = 0.0
    max_right = 0.0
    max_delta_seq = ""
    for r in rows:
        l, rr = selected_lr(r)
        d = abs(l - rr)
        if d > max_delta:
            max_delta = d
            max_left = l
            max_right = rr
            max_delta_seq = str(r.get("seq", ""))

    t_risk = rel_time(first_risk, t0) if first_risk else None
    t_avoid = rel_time(first_avoid, t0) if first_avoid else None
    t_turn = rel_time(first_turn, t0) if first_turn else None
    t_stop_local = rel_time(last_stop_before_turn, t0) if last_stop_before_turn else None
    t_recovery = rel_time(recovery, t0) if recovery else None

    avoidance_activation_delay = None
    if t_risk is not None and t_avoid is not None:
        avoidance_activation_delay = max(0.0, t_avoid - t_risk)

    observation_to_maneuver = None
    if t_stop_local is not None and t_turn is not None:
        observation_to_maneuver = max(0.0, t_turn - t_stop_local)

    avoidance_duration = None
    if t_avoid is not None and t_recovery is not None:
        avoidance_duration = max(0.0, t_recovery - t_avoid)

    has_risk = bool(risk_rows)
    has_avoid = bool(avoid_rows)
    has_stop = bool(stop_rows)
    has_turn = bool(turn_rows)
    has_recovery = recovery is not None
    has_diff = max_delta > DIFF_THRESHOLD

    if has_risk and has_avoid and has_stop and has_turn and has_recovery and has_diff:
        outcome = "Success"
    elif has_risk and has_avoid and (has_stop or has_turn):
        outcome = "Partial"
    else:
        outcome = "Fail"

    result = {
        "Trial": run_id.split("_")[-1],
        "Run ID": run_id,
        "Max Risk Score": fmt(max_risk, 4),
        "Avoidance Activation Delay (s)": fmt(avoidance_activation_delay),
        "Observation-to-Maneuver Time (s)": fmt(observation_to_maneuver),
        "Avoidance Duration (s)": fmt(avoidance_duration),
        "Max Thruster Difference": fmt(max_delta, 4),
        "Mission Recovery": "Yes" if has_recovery else "No",
        "Outcome": outcome,
        "Turn Commands Observed": ",".join(sorted(set(cmd(r) for r in turn_rows))),
        "Total Events": str(len(rows)),
        "AVOID Events": str(sum(1 for r in rows if state(r) == "AVOID")),
        "REJOIN Events": str(sum(1 for r in rows if state(r) == "REJOIN")),
        "MISSION Events": str(sum(1 for r in rows if state(r) == "MISSION")),
        "STOP Events": str(sum(1 for r in rows if cmd(r) == "STOP")),
        "TURN Events": str(len(turn_rows)),
        "Last STOP Before First Turn Seq": (
            str(last_stop_before_turn.get("seq", "")) if last_stop_before_turn else ""
        ),
        "First Turn Seq": str(first_turn.get("seq", "")) if first_turn else "",
        "Max Delta Seq": max_delta_seq,
        "Max Delta Left Cmd": fmt(max_left, 4),
        "Max Delta Right Cmd": fmt(max_right, 4),
    }

    return result


def upsert_master(row):
    out_dir = Path.home() / "seano_journal_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dynamic_obstacle_results.csv"

    fieldnames = list(row.keys())
    rows = []

    if out_file.exists():
        with out_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("Run ID") != row["Run ID"]:
                    rows.append(r)

    rows.append(row)
    rows.sort(key=lambda r: r.get("Run ID", ""))

    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_file


def main():
    if len(sys.argv) != 2:
        print("Usage: analyze_dynamic_trial.py <run_id>")
        sys.exit(2)

    run_id = sys.argv[1]
    row = analyze(run_id)
    out_file = upsert_master(row)

    print("==== TRIAL SUMMARY ====")
    for k, v in row.items():
        print(f"{k}: {v}")

    print()
    print(f"[OK] updated master CSV: {out_file}")


if __name__ == "__main__":
    main()
