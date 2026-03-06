#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _infer_scenario(name: str) -> str:
    n = name.lower()
    if "failsafe" in n:
        return "failsafe"
    if "watchdog" in n:
        return "watchdog"
    if "rejoin" in n:
        return "hazard_rejoin"
    return "other"


def _infer_profile(name: str) -> str:
    n = name.lower()
    if "failsafe" in n or "watchdog" in n:
        return "synthetic_watchdog"
    if "rejoin" in n:
        return "synthetic_light"
    return "unknown"


def _pass_fail(row: Dict[str, Any]) -> str:
    scenario = row["scenario"]

    if scenario == "hazard_rejoin":
        if (
            row["takeover_segments"] >= 1
            and row["reaction_n"] >= 1
            and row["release_n"] >= 1
            and row["rejoin_n"] >= 1
            and row["rejoin_done"] >= 1
            and row["rejoin_cancelled"] == 0
            and row["rejoin_timeouts"] == 0
        ):
            return "PASS"
        return "FAIL"

    if scenario in ("failsafe", "watchdog"):
        if row["failsafe_rises"] >= 1:
            return "PASS"
        return "FAIL"

    return "CHECK"


def _load_one_metrics(json_path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] skip invalid json: {json_path} ({type(exc).__name__})")
        return None

    bag_path = str(data.get("bag", ""))
    bag_name = Path(bag_path).name if bag_path else json_path.parent.name

    row = {
        "bag_name": bag_name,
        "bag_path": bag_path or str(json_path.parent),
        "metrics_json": str(json_path),
        "scenario": _infer_scenario(bag_name),
        "runtime_profile": _infer_profile(bag_name),
        "takeover_segments": int(_safe_get(data, "override", "takeover_segments", default=0)),
        "takeover_duration_mean_s": _to_float(
            _safe_get(data, "override", "duration_s", "mean", default=0.0)
        ),
        "reaction_mean_s": _to_float(_safe_get(data, "reaction_time_s", "mean", default=0.0)),
        "reaction_n": int(_to_float(_safe_get(data, "reaction_time_s", "n", default=0.0))),
        "release_mean_s": _to_float(_safe_get(data, "release_time_s", "mean", default=0.0)),
        "release_n": int(_to_float(_safe_get(data, "release_time_s", "n", default=0.0))),
        "rejoin_mean_s": _to_float(_safe_get(data, "rejoin_time_s", "mean", default=0.0)),
        "rejoin_n": int(_to_float(_safe_get(data, "rejoin_time_s", "n", default=0.0))),
        "rejoin_done": int(_safe_get(data, "rejoin", "done", default=0)),
        "rejoin_cancelled": int(_safe_get(data, "rejoin", "cancelled", default=0)),
        "rejoin_timeouts": int(_safe_get(data, "rejoin", "timeouts", default=0)),
        "failsafe_rises": int(_safe_get(data, "failsafe", "rises", default=0)),
        "mismatch_ratio": _to_float(
            _safe_get(data, "mode_mismatch", "mismatch_ratio", default=0.0)
        ),
    }
    row["pass_fail"] = _pass_fail(row)
    return row


def _find_metrics_files(root: Path) -> List[Path]:
    return sorted(root.rglob("phase6_metrics.json"))


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "bag_name",
        "scenario",
        "runtime_profile",
        "takeover_segments",
        "takeover_duration_mean_s",
        "reaction_mean_s",
        "reaction_n",
        "release_mean_s",
        "release_n",
        "rejoin_mean_s",
        "rejoin_n",
        "rejoin_done",
        "rejoin_cancelled",
        "rejoin_timeouts",
        "failsafe_rises",
        "mismatch_ratio",
        "pass_fail",
        "bag_path",
        "metrics_json",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(v: Any, ndigits: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def _write_markdown(rows: List[Dict[str, Any]], out_md: Path) -> None:
    lines: List[str] = []
    lines.append("# Phase 6 Results Summary")
    lines.append("")
    lines.append(
        "| Bag | Scenario | Profile | Takeover | React Mean (n) | Release Mean (n) | Rejoin Mean (n) | Rejoin Done | Cancel | Timeout | Failsafe | Mismatch | Pass/Fail |"
    )
    lines.append("|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["bag_name"],
                    r["scenario"],
                    r["runtime_profile"],
                    str(r["takeover_segments"]),
                    f'{_fmt(r["reaction_mean_s"])} ({r["reaction_n"]})',
                    f'{_fmt(r["release_mean_s"])} ({r["release_n"]})',
                    f'{_fmt(r["rejoin_mean_s"])} ({r["rejoin_n"]})',
                    str(r["rejoin_done"]),
                    str(r["rejoin_cancelled"]),
                    str(r["rejoin_timeouts"]),
                    str(r["failsafe_rises"]),
                    _fmt(r["mismatch_ratio"]),
                    r["pass_fail"],
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `hazard_rejoin` dinilai PASS bila takeover, release, dan rejoin terisi serta tidak ada cancel/timeout."
    )
    lines.append("- `failsafe/watchdog` dinilai PASS bila `failsafe_rises >= 1`.")
    lines.append("- Hasil ini berasal dari file `phase6_metrics.json` per bag.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="~/bags",
        help="Folder root yang berisi banyak bag run",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Folder output summary (default: <root>/phase6_summary)",
    )
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root)).resolve()
    out_dir = (
        Path(os.path.expanduser(args.out_dir)).resolve()
        if args.out_dir
        else root / "phase6_summary"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = _find_metrics_files(root)
    if not metrics_files:
        print(f"[WARN] no phase6_metrics.json found under: {root}")
        return 1

    rows: List[Dict[str, Any]] = []
    for jf in metrics_files:
        row = _load_one_metrics(jf)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: r["bag_name"])

    out_csv = out_dir / "phase6_results.csv"
    out_md = out_dir / "phase6_results.md"
    _write_csv(rows, out_csv)
    _write_markdown(rows, out_md)

    print("\n=== Phase 6 Aggregated Results ===")
    print(f"runs found: {len(rows)}")
    for r in rows:
        print(
            f"- {r['bag_name']}: "
            f"scenario={r['scenario']} "
            f"takeover={r['takeover_segments']} "
            f"reaction={r['reaction_mean_s']:.3f}s(n={r['reaction_n']}) "
            f"release={r['release_mean_s']:.3f}s(n={r['release_n']}) "
            f"rejoin={r['rejoin_mean_s']:.3f}s(n={r['rejoin_n']}) "
            f"failsafe={r['failsafe_rises']} "
            f"mismatch={r['mismatch_ratio']:.3f} "
            f"{r['pass_fail']}"
        )

    print(f"\nsaved csv: {out_csv}")
    print(f"saved md : {out_md}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
