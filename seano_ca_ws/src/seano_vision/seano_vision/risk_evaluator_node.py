#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO Collision Avoidance - Risk Evaluator Node (ROS2 Humble)

Subscribe:
  - /camera/detections        (vision_msgs/Detection2DArray)
  - /camera/image_raw_reliable (sensor_msgs/Image)

Publish:
  - /ca/risk            (std_msgs/Float32)
  - /ca/command         (std_msgs/String)
  - /ca/metrics         (std_msgs/String JSON)
  - /ca/vision_quality  (std_msgs/Float32)
  - /ca/debug_image     (sensor_msgs/Image)

Notes:
- ASCII-only overlay text (avoid unicode rendering issues in cv2).
- Maritime-style HUD theme (navy/teal), compact & readable.
- Bearing ruler overlay (camera-relative) for navigational feel.
- Image buffer by stamp to better align debug overlay with detections.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from cv_bridge import CvBridge  # type: ignore
    _HAS_CV = True
except Exception:
    cv2 = None
    np = None
    CvBridge = None
    _HAS_CV = False


# --------------------------
# Utils
# --------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def clampi(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(x))))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax - aw / 2.0, ay - ah / 2.0
    ax2, ay2 = ax + aw / 2.0, ay + ah / 2.0
    bx1, by1 = bx - bw / 2.0, by - bh / 2.0
    bx2, by2 = bx + bw / 2.0, by + bh / 2.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def _clean_str_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(v).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _rect_intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return int(iw * ih)


def _stamp_to_sec(stamp) -> float:
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return 0.0


def _ascii_safe(s: str) -> str:
    try:
        return s.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return str(s)


# --------------------------
# Data structures
# --------------------------
@dataclass
class Det:
    class_id: str
    score: float
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Track:
    tid: int
    class_id: str
    score: float
    cx: float
    cy: float
    w: float
    h: float
    last_t: float

    bearing_deg: float = 0.0
    bearing_rate_dps: float = 0.0
    log_area: float = 0.0
    dlog_area_dt: float = 0.0
    risk_ema: float = 0.0


# --------------------------
# Node
# --------------------------
class RiskEvaluatorNode(Node):
    def __init__(self) -> None:
        super().__init__("risk_evaluator_node")

        # QoS (low-latency default)
        self.declare_parameter("qos_depth", 1)
        depth = max(1, int(self.get_parameter("qos_depth").value))
        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Topics
        self.declare_parameter("detections_topic", "/camera/detections")
        self.declare_parameter("image_topic", "/camera/image_raw_reliable")

        self.declare_parameter("risk_topic", "/ca/risk")
        self.declare_parameter("command_topic", "/ca/command")
        self.declare_parameter("metrics_topic", "/ca/metrics")
        self.declare_parameter("vision_quality_topic", "/ca/vision_quality")
        self.declare_parameter("debug_image_topic", "/ca/debug_image")

        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("min_det_score", 0.35)

        self.declare_parameter(
            "allow_class_ids",
            [""],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="Whitelist class_id. Empty => allow all"
            ),
        )
        self.declare_parameter(
            "deny_class_ids",
            [""],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="Blacklist class_id"
            ),
        )

        # Tracking
        self.declare_parameter("enable_tracking", True)
        self.declare_parameter("iou_match_thresh", 0.25)
        self.declare_parameter("track_timeout_s", 1.0)
        self.declare_parameter("max_tracks", 30)

        # Camera proxy
        self.declare_parameter("camera_hfov_deg", 70.0)

        # Domain
        self.declare_parameter("center_band_ratio", 0.30)
        self.declare_parameter("bottom_danger_ratio", 0.65)
        self.declare_parameter("near_area_ratio", 0.08)

        # Risk weights
        self.declare_parameter("w_proximity", 0.50)
        self.declare_parameter("w_center", 0.20)
        self.declare_parameter("w_approach", 0.20)
        self.declare_parameter("w_bearing_const", 0.10)
        self.declare_parameter("bearing_rate_bad_dps", 12.0)

        # TTC proxy
        self.declare_parameter("ttc_area_threshold", 0.10)
        self.declare_parameter("ttc_max_s", 60.0)

        # Avoid hysteresis
        self.declare_parameter("enter_avoid_risk", 0.55)
        self.declare_parameter("exit_avoid_risk", 0.35)
        self.declare_parameter("min_cmd_hold_s", 0.6)

        # Commands
        self.declare_parameter("cmd_hold", "HOLD_COURSE")
        self.declare_parameter("cmd_slow", "SLOW_DOWN")
        self.declare_parameter("cmd_turn_left_slow", "TURN_LEFT_SLOW")
        self.declare_parameter("cmd_turn_right_slow", "TURN_RIGHT_SLOW")
        self.declare_parameter("cmd_turn_left", "TURN_LEFT")
        self.declare_parameter("cmd_turn_right", "TURN_RIGHT")
        self.declare_parameter("cmd_stop", "STOP")

        self.declare_parameter("prefer_starboard", True)
        self.declare_parameter("emergency_turn_away", True)

        # Vision quality
        self.declare_parameter("use_vision_quality", True)
        self.declare_parameter("vq_min", 0.25)
        self.declare_parameter("vq_check_every_n_frames", 6)

        # Image buffer sync
        self.declare_parameter("image_buffer_size", 12)
        self.declare_parameter("max_image_age_s", 0.40)

        # --------------------------
        # Overlay (maritime HUD)
        # --------------------------
        self.declare_parameter("overlay_enabled", True)
        self.declare_parameter("overlay_anchor", "auto")  # auto/left/right
        self.declare_parameter("overlay_max_width_ratio", 0.42)
        self.declare_parameter("overlay_margin_px", 12)
        self.declare_parameter("overlay_padding_px", 10)
        self.declare_parameter("overlay_alpha_bg", 0.80)
        self.declare_parameter("overlay_border_thickness", 1)

        self.declare_parameter("overlay_font_face", "simplex")  # simplex/plain
        self.declare_parameter("overlay_scale_head", 0.56)
        self.declare_parameter("overlay_scale_body", 0.44)
        self.declare_parameter("overlay_thickness", 1)
        self.declare_parameter("overlay_text_shadow", True)

        self.declare_parameter("overlay_draw_grid", True)
        self.declare_parameter("overlay_draw_corridor", True)
        self.declare_parameter("overlay_line_alpha", 0.20)

        self.declare_parameter("overlay_riskbar_h_px", 9)
        self.declare_parameter("overlay_bbox_chip_alpha", 0.36)

        # Theme colors (BGR)
        self.declare_parameter("overlay_bg_bgr", [16, 24, 40])        # deep navy
        self.declare_parameter("overlay_panel_bgr", [12, 18, 32])     # panel navy
        self.declare_parameter("overlay_teal_bgr", [180, 200, 0])     # teal accent
        self.declare_parameter("overlay_grid_bgr", [110, 140, 170])   # steel
        self.declare_parameter("overlay_text_bgr", [238, 238, 238])   # near white
        self.declare_parameter("overlay_muted_bgr", [170, 170, 170])  # muted

        # Bearing ruler
        self.declare_parameter("overlay_draw_bearing_ruler", True)
        self.declare_parameter("overlay_ruler_h_px", 36)
        self.declare_parameter("overlay_ruler_alpha", 0.28)
        self.declare_parameter("overlay_ruler_tick_deg", 10)

        # HUD content
        self.declare_parameter("overlay_show_topk", 3)

        # --------------------------
        # State
        # --------------------------
        self.bridge = CvBridge() if (_HAS_CV and CvBridge is not None) else None

        self.image_w: Optional[int] = None
        self.image_h: Optional[int] = None

        self.frame_count = 0
        self.vision_quality = 1.0

        self.tracks: Dict[int, Track] = {}
        self.next_tid = 1

        self.avoid_mode = False
        self.last_cmd = str(self.get_parameter("cmd_hold").value)
        self.last_cmd_time = 0.0

        self.allow_ids: set[str] = set()
        self.deny_ids: set[str] = set()
        self._refresh_filters()

        self.last_det_t: Optional[float] = None

        self.image_buf: Deque[Image] = deque(maxlen=max(4, int(self.get_parameter("image_buffer_size").value)))

        self.add_on_set_parameters_callback(self._on_params)

        # pubs
        self.pub_risk = self.create_publisher(Float32, str(self.get_parameter("risk_topic").value), self.qos)
        self.pub_cmd = self.create_publisher(String, str(self.get_parameter("command_topic").value), self.qos)
        self.pub_metrics = self.create_publisher(String, str(self.get_parameter("metrics_topic").value), self.qos)
        self.pub_vq = self.create_publisher(Float32, str(self.get_parameter("vision_quality_topic").value), self.qos)
        self.pub_dbg = self.create_publisher(Image, str(self.get_parameter("debug_image_topic").value), self.qos)

        # subs
        self.sub_det = self.create_subscription(
            Detection2DArray, str(self.get_parameter("detections_topic").value), self.on_detections, self.qos
        )
        self.sub_img = self.create_subscription(
            Image, str(self.get_parameter("image_topic").value), self.on_raw_image, self.qos
        )

        self.get_logger().info(
            f"risk_evaluator_node started | cv={_HAS_CV} depth={self.qos.depth} "
            f"det={self.get_parameter('detections_topic').value} img={self.get_parameter('image_topic').value}"
        )

    # --------------------------
    # Params
    # --------------------------
    def _on_params(self, params: List[Parameter]) -> SetParametersResult:
        res = SetParametersResult()
        res.successful = True
        res.reason = "ok"
        try:
            names = {p.name for p in params}
            if "allow_class_ids" in names or "deny_class_ids" in names:
                self._refresh_filters()
            if "image_buffer_size" in names:
                n = max(4, int(self.get_parameter("image_buffer_size").value))
                self.image_buf = deque(self.image_buf, maxlen=n)
        except Exception as e:
            self.get_logger().warn(f"param refresh warning: {e}")
        return res

    def _refresh_filters(self) -> None:
        allow_v = self.get_parameter("allow_class_ids").value
        deny_v = self.get_parameter("deny_class_ids").value
        self.allow_ids = set(_clean_str_list(allow_v))
        self.deny_ids = set(_clean_str_list(deny_v))
        self.allow_ids.discard("")
        self.deny_ids.discard("")

    # --------------------------
    # Image callback (buffer)
    # --------------------------
    def on_raw_image(self, msg: Image) -> None:
        self.image_buf.append(msg)

        if msg.width > 0 and msg.height > 0:
            self.image_w = int(msg.width)
            self.image_h = int(msg.height)

        if not bool(self.get_parameter("use_vision_quality").value):
            self.vision_quality = 1.0
            return
        if self.bridge is None or not _HAS_CV:
            self.vision_quality = 1.0
            return

        self.frame_count += 1
        n = int(self.get_parameter("vq_check_every_n_frames").value)
        if n < 1:
            n = 1
        if (self.frame_count % n) != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.vision_quality = float(self._compute_vision_quality(frame))
            self.pub_vq.publish(Float32(data=float(self.vision_quality)))
        except Exception:
            self.vision_quality = 0.0
            self.pub_vq.publish(Float32(data=float(self.vision_quality)))

    def _compute_vision_quality(self, bgr) -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_var = float(lap.var())
        blur_score = smoothstep(30.0, 180.0, blur_var)

        mean_b = float(gray.mean())
        bright_score = 1.0 - abs(mean_b - 120.0) / 120.0
        bright_score = clamp(bright_score, 0.0, 1.0)

        glare_ratio = float((gray > 245).mean())
        glare_score = 1.0 - clamp(glare_ratio / 0.08, 0.0, 1.0)

        std_b = float(gray.std())
        contrast_score = clamp(std_b / 60.0, 0.0, 1.0)

        score = (0.35 * blur_score) + (0.30 * bright_score) + (0.20 * contrast_score) + (0.15 * glare_score)
        return clamp(score, 0.0, 1.0)

    def _pick_image_for_stamp(self, target_stamp_sec: float) -> Optional[Image]:
        if not self.image_buf:
            return None
        max_age = float(self.get_parameter("max_image_age_s").value)

        best = None
        best_dt = 1e9
        for m in self.image_buf:
            s = _stamp_to_sec(m.header.stamp)
            if s <= 0:
                continue
            dt = abs(s - target_stamp_sec)
            if dt < best_dt:
                best_dt = dt
                best = m

        if best is not None and max_age > 0:
            if best_dt > max_age:
                return None
        return best

    # --------------------------
    # Detections callback
    # --------------------------
    def on_detections(self, msg: Detection2DArray) -> None:
        t0 = time.time()
        t = t0

        det_dt_ms = None
        if self.last_det_t is not None:
            det_dt_ms = float((t - self.last_det_t) * 1000.0)
        self.last_det_t = t

        dets = self._parse_detections(msg)
        dets = self._apply_filters(dets)

        if bool(self.get_parameter("enable_tracking").value):
            self._update_tracks(dets, t)
        else:
            self._tracks_from_dets(dets, t)

        overall_risk, top, metrics = self._evaluate(t, det_dt_ms)
        cmd = self._decide_command(t, overall_risk, top, metrics)

        proc_ms = (time.time() - t0) * 1000.0
        metrics["proc_ms"] = float(proc_ms)
        if det_dt_ms and det_dt_ms > 1e-6:
            metrics["fps"] = float(1000.0 / det_dt_ms)
        else:
            metrics["fps"] = None

        self.pub_risk.publish(Float32(data=float(overall_risk)))
        self.pub_cmd.publish(String(data=str(cmd)))

        try:
            self.pub_metrics.publish(String(data=json.dumps(metrics)))
        except Exception:
            pass

        if bool(self.get_parameter("publish_debug_image").value):
            self._publish_debug_overlay(msg, metrics, top)

    # --------------------------
    # Parse & filter
    # --------------------------
    def _parse_detections(self, msg: Detection2DArray) -> List[Det]:
        out: List[Det] = []
        min_score = float(self.get_parameter("min_det_score").value)

        for d in msg.detections:
            if len(d.results) == 0:
                continue

            best_score = -1.0
            best_cid = None
            for r in d.results:
                try:
                    sc = float(r.hypothesis.score)
                    cid = str(r.hypothesis.class_id)
                except Exception:
                    continue
                if sc > best_score:
                    best_score = sc
                    best_cid = cid

            if best_cid is None or best_score < min_score:
                continue

            cx = float(d.bbox.center.position.x)
            cy = float(d.bbox.center.position.y)
            w = float(d.bbox.size_x)
            h = float(d.bbox.size_y)

            out.append(Det(class_id=str(best_cid), score=float(best_score), cx=cx, cy=cy, w=w, h=h))

        return out

    def _apply_filters(self, dets: List[Det]) -> List[Det]:
        self._refresh_filters()
        if self.deny_ids:
            dets = [d for d in dets if d.class_id not in self.deny_ids]
        if self.allow_ids:
            dets = [d for d in dets if d.class_id in self.allow_ids]
        return dets

    # --------------------------
    # Tracking
    # --------------------------
    def _tracks_from_dets(self, dets: List[Det], t: float) -> None:
        self.tracks.clear()
        self.next_tid = 1
        for d in dets[: int(self.get_parameter("max_tracks").value)]:
            tr = self._make_track(self.next_tid, d, t, prev=None)
            self.tracks[self.next_tid] = tr
            self.next_tid += 1

    def _update_tracks(self, dets: List[Det], t: float) -> None:
        timeout_s = float(self.get_parameter("track_timeout_s").value)
        dead = [tid for tid, tr in self.tracks.items() if (t - tr.last_t) > timeout_s]
        for tid in dead:
            self.tracks.pop(tid, None)

        iou_thr = float(self.get_parameter("iou_match_thresh").value)

        candidates: List[Tuple[float, int, int]] = []
        det_bbs = [(d.cx, d.cy, d.w, d.h) for d in dets]

        for tid, tr in self.tracks.items():
            tr_bb = (tr.cx, tr.cy, tr.w, tr.h)
            for j, db in enumerate(det_bbs):
                candidates.append((iou_xywh(tr_bb, db), tid, j))

        candidates.sort(reverse=True, key=lambda x: x[0])

        used_tids = set()
        used_det = set()

        for iou_val, tid, j in candidates:
            if iou_val < iou_thr:
                break
            if tid in used_tids or j in used_det:
                continue
            prev = self.tracks.get(tid)
            if prev is None:
                continue
            self.tracks[tid] = self._make_track(tid, dets[j], t, prev=prev)
            used_tids.add(tid)
            used_det.add(j)

        max_tracks = int(self.get_parameter("max_tracks").value)
        for j, d in enumerate(dets):
            if j in used_det:
                continue
            if len(self.tracks) >= max_tracks:
                break
            tid = self.next_tid
            self.next_tid += 1
            self.tracks[tid] = self._make_track(tid, d, t, prev=None)

    def _make_track(self, tid: int, d: Det, t: float, prev: Optional[Track]) -> Track:
        W = float(self.image_w or 1)
        x_ratio = d.cx / max(W, 1e-9)
        hfov = float(self.get_parameter("camera_hfov_deg").value)
        bearing_deg = (x_ratio - 0.5) * hfov

        area = max(1.0, d.w * d.h)
        log_area = math.log(area)

        bearing_rate = 0.0
        dlog_dt = 0.0
        risk_ema = 0.0

        if prev is not None:
            dt = max(1e-3, t - prev.last_t)
            bearing_rate = (bearing_deg - prev.bearing_deg) / dt
            dlog_dt = (log_area - prev.log_area) / dt
            risk_ema = prev.risk_ema

        return Track(
            tid=tid,
            class_id=d.class_id,
            score=d.score,
            cx=d.cx,
            cy=d.cy,
            w=d.w,
            h=d.h,
            last_t=t,
            bearing_deg=bearing_deg,
            bearing_rate_dps=bearing_rate,
            log_area=log_area,
            dlog_area_dt=dlog_dt,
            risk_ema=risk_ema,
        )

    # --------------------------
    # Risk evaluate
    # --------------------------
    def _evaluate(self, t: float, det_dt_ms: Optional[float]) -> Tuple[float, Optional[Track], dict]:
        W = float(self.image_w or 1)
        H = float(self.image_h or 1)
        img_area = max(W * H, 1e-9)

        center_band = float(self.get_parameter("center_band_ratio").value)
        bottom_danger = float(self.get_parameter("bottom_danger_ratio").value)
        near_area_ratio = float(self.get_parameter("near_area_ratio").value)
        bearing_rate_bad = float(self.get_parameter("bearing_rate_bad_dps").value)

        w_prox = float(self.get_parameter("w_proximity").value)
        w_center = float(self.get_parameter("w_center").value)
        w_app = float(self.get_parameter("w_approach").value)
        w_bconst = float(self.get_parameter("w_bearing_const").value)

        use_vq = bool(self.get_parameter("use_vision_quality").value)
        vq = float(self.vision_quality) if use_vq else 1.0
        self.pub_vq.publish(Float32(data=float(vq)))

        metrics: dict = {
            "ts": t,
            "status": "no_detections" if not self.tracks else "ok",
            "risk": 0.0,
            "cmd": str(self.last_cmd),
            "vision_quality": float(vq),
            "num_tracks": int(len(self.tracks)),
            "det_dt_ms": det_dt_ms,
            "avoid_mode": bool(self.avoid_mode),
        }

        if not self.tracks:
            return 0.0, None, metrics

        top: Optional[Track] = None
        best = 0.0
        top_comp = {}
        top_feat = {}

        ranked: List[Tuple[float, Track, dict, dict]] = []

        for tr in self.tracks.values():
            x_ratio = tr.cx / max(W, 1e-9)
            bottom_y_ratio = (tr.cy + tr.h / 2.0) / max(H, 1e-9)
            area_ratio = (tr.w * tr.h) / img_area

            in_corridor = abs(x_ratio - 0.5) <= (center_band / 2.0)
            bottomness = smoothstep(bottom_danger, 1.0, bottom_y_ratio)

            proximity = smoothstep(near_area_ratio * 0.25, near_area_ratio, area_ratio)
            centrality = 1.0 - clamp(abs(x_ratio - 0.5) / max(center_band / 2.0, 1e-6), 0.0, 1.0)
            approach = smoothstep(0.00, 0.55, tr.dlog_area_dt)
            bearing_const = 1.0 - clamp(abs(tr.bearing_rate_dps) / max(bearing_rate_bad, 1e-6), 0.0, 1.0)

            prox_combo = clamp(0.60 * proximity + 0.40 * bottomness, 0.0, 1.0)
            conf = clamp(tr.score, 0.0, 1.0)

            raw = (
                w_prox * prox_combo +
                w_center * centrality +
                w_app * approach +
                w_bconst * bearing_const
            )
            raw = clamp(raw * (0.55 + 0.45 * conf), 0.0, 1.0)
            raw = clamp(raw * (0.5 + 0.5 * vq), 0.0, 1.0)

            alpha = 0.35
            tr.risk_ema = alpha * raw + (1.0 - alpha) * tr.risk_ema

            comp = {
                "prox": float(prox_combo),
                "center": float(centrality),
                "approach": float(approach),
                "bconst": float(bearing_const),
                "conf": float(conf),
                "raw": float(raw),
                "ema": float(tr.risk_ema),
            }
            feat = {
                "x_ratio": float(x_ratio),
                "bottom_y_ratio": float(bottom_y_ratio),
                "area_ratio": float(area_ratio),
                "in_corridor": bool(in_corridor),
            }

            ranked.append((float(tr.risk_ema), tr, comp, feat))

            if tr.risk_ema > best:
                best = tr.risk_ema
                top = tr
                top_comp = comp
                top_feat = feat

        if top is None:
            return 0.0, None, metrics

        # TTC proxy
        a_th = float(self.get_parameter("ttc_area_threshold").value)
        ttc_max = float(self.get_parameter("ttc_max_s").value)
        area_ratio = float(top_feat.get("area_ratio", 0.0))

        ttc_proxy = None
        ttc_reason = "unknown"
        if a_th <= 1e-9:
            ttc_reason = "disabled"
        else:
            if area_ratio >= a_th:
                ttc_proxy = 0.0
                ttc_reason = "at_threshold"
            else:
                if top.dlog_area_dt > 1e-6:
                    ttc = (math.log(max(a_th, 1e-9)) - math.log(max(area_ratio, 1e-9))) / top.dlog_area_dt
                    ttc = clamp(ttc, 0.0, ttc_max)
                    ttc_proxy = float(ttc)
                    ttc_reason = "ok"
                else:
                    ttc_reason = "no_approach"

        situation = self._classify_situation(
            bearing_deg=float(top.bearing_deg),
            bearing_rate_dps=float(top.bearing_rate_dps),
            in_corridor=bool(top_feat.get("in_corridor", False)),
        )

        ranked.sort(key=lambda x: x[0], reverse=True)
        topk_n = max(0, int(self.get_parameter("overlay_show_topk").value))
        topk_list = []
        for i, (r, tr, _c, _f) in enumerate(ranked[:topk_n]):
            topk_list.append({
                "rank": i + 1,
                "track_id": int(tr.tid),
                "class_id": str(tr.class_id),
                "score": float(tr.score),
                "risk": float(clamp(r, 0.0, 1.0)),
                "bearing_deg": float(tr.bearing_deg),
            })

        metrics.update({
            "risk": float(clamp(best, 0.0, 1.0)),
            "target": {
                "track_id": int(top.tid),
                "class_id": str(top.class_id),
                "score": float(top.score),
                "x_ratio": float(top_feat["x_ratio"]),
                "bottom_y_ratio": float(top_feat["bottom_y_ratio"]),
                "area_ratio": float(top_feat["area_ratio"]),
                "in_corridor": bool(top_feat["in_corridor"]),
                "bearing_deg": float(top.bearing_deg),
                "bearing_rate_dps": float(top.bearing_rate_dps),
                "dlog_area_dt": float(top.dlog_area_dt),
                "ttc_proxy_s": ttc_proxy,
                "ttc_reason": ttc_reason,
            },
            "components": top_comp,
            "situation": situation,
            "topk": topk_list,
        })

        return float(clamp(best, 0.0, 1.0)), top, metrics

    def _classify_situation(self, bearing_deg: float, bearing_rate_dps: float, in_corridor: bool) -> str:
        br = abs(bearing_rate_dps)
        b = bearing_deg
        if in_corridor and abs(b) < 8.0 and br < 2.0:
            return "HEAD_ON"
        if br < 2.0:
            if b >= 8.0:
                return "CROSSING_RIGHT"
            if b <= -8.0:
                return "CROSSING_LEFT"
        if br >= 6.0:
            return "DIVERGING"
        return "UNKNOWN"

    # --------------------------
    # Command + COLREG hint
    # --------------------------
    def _colregs_hint(self, situation: str) -> str:
        if situation == "HEAD_ON":
            return "COLREG: HEAD-ON -> TURN STARBOARD"
        if situation == "CROSSING_RIGHT":
            return "COLREG: GIVE-WAY (TARGET STARBOARD)"
        if situation == "CROSSING_LEFT":
            return "COLREG: STAND-ON (TARGET PORT)"
        if situation == "DIVERGING":
            return "COLREG: DIVERGING"
        return "COLREG: UNKNOWN"

    def _decide_command(self, t: float, risk: float, top: Optional[Track], metrics: dict) -> str:
        cmd_hold = str(self.get_parameter("cmd_hold").value)
        cmd_slow = str(self.get_parameter("cmd_slow").value)
        cmd_tls = str(self.get_parameter("cmd_turn_left_slow").value)
        cmd_trs = str(self.get_parameter("cmd_turn_right_slow").value)
        cmd_tl = str(self.get_parameter("cmd_turn_left").value)
        cmd_tr = str(self.get_parameter("cmd_turn_right").value)
        cmd_stop = str(self.get_parameter("cmd_stop").value)

        enter_r = float(self.get_parameter("enter_avoid_risk").value)
        exit_r = float(self.get_parameter("exit_avoid_risk").value)
        hold_s = float(self.get_parameter("min_cmd_hold_s").value)

        prefer_starboard = bool(self.get_parameter("prefer_starboard").value)
        emergency_turn_away = bool(self.get_parameter("emergency_turn_away").value)

        vq_min = float(self.get_parameter("vq_min").value)
        vq = float(self.vision_quality)

        if not self.avoid_mode and risk >= enter_r:
            self.avoid_mode = True
        elif self.avoid_mode and risk <= exit_r:
            self.avoid_mode = False

        situation = str(metrics.get("situation", "UNKNOWN"))
        metrics["colregs"] = self._colregs_hint(situation)

        if bool(self.get_parameter("use_vision_quality").value) and (vq < vq_min):
            metrics["vision_mode"] = "CAUTION"
            desired = cmd_slow if risk > 0.25 else cmd_hold
            self._maybe_update_cmd(t, desired, hold_s)
            metrics["cmd"] = self.last_cmd
            return self.last_cmd

        metrics["vision_mode"] = "NORMAL"

        if top is None:
            self._maybe_update_cmd(t, cmd_hold, hold_s)
            metrics["cmd"] = self.last_cmd
            return self.last_cmd

        if risk >= 0.92:
            self._maybe_update_cmd(t, cmd_stop, hold_s)
            metrics["cmd"] = self.last_cmd
            return self.last_cmd

        if (not self.avoid_mode) and risk < 0.45:
            self._maybe_update_cmd(t, cmd_hold, hold_s)
            metrics["cmd"] = self.last_cmd
            return self.last_cmd

        W = float(self.image_w or 1)
        x_ratio = top.cx / max(W, 1e-9)
        in_corridor = bool(metrics.get("target", {}).get("in_corridor", False))

        extreme = 0.82
        area_ratio = float(metrics.get("target", {}).get("area_ratio", 0.0))
        near_area_ratio = float(self.get_parameter("near_area_ratio").value)
        very_close = area_ratio >= (near_area_ratio * 1.2)

        direction = "RIGHT" if prefer_starboard else ("LEFT" if x_ratio > 0.5 else "RIGHT")

        if emergency_turn_away and very_close:
            if x_ratio > extreme:
                direction = "LEFT"
            elif x_ratio < (1.0 - extreme):
                direction = "RIGHT"

        if risk < 0.55:
            desired = cmd_slow
        elif risk < 0.75:
            desired = cmd_trs if direction == "RIGHT" else cmd_tls
        else:
            desired = cmd_tr if direction == "RIGHT" else cmd_tl

        if (not in_corridor) and risk < 0.70:
            desired = cmd_slow

        self._maybe_update_cmd(t, desired, hold_s)
        metrics["cmd"] = self.last_cmd
        metrics["decision"] = {"direction": direction, "very_close": bool(very_close)}
        return self.last_cmd

    def _maybe_update_cmd(self, t: float, desired: str, hold_s: float) -> None:
        if desired == self.last_cmd:
            return
        if (t - self.last_cmd_time) < hold_s:
            return
        self.last_cmd = desired
        self.last_cmd_time = t

    # --------------------------
    # Overlay helpers
    # --------------------------
    def _font(self) -> int:
        face = str(self.get_parameter("overlay_font_face").value).lower()
        if face in ("plain", "hershey_plain"):
            return cv2.FONT_HERSHEY_PLAIN
        return cv2.FONT_HERSHEY_SIMPLEX

    def _pcolor(self, name: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        try:
            v = self.get_parameter(name).value
            if isinstance(v, (list, tuple)) and len(v) == 3:
                b = clampi(v[0], 0, 255)
                g = clampi(v[1], 0, 255)
                r = clampi(v[2], 0, 255)
                return (b, g, r)
        except Exception:
            pass
        return fallback

    def _alpha_rect(self, img, x1, y1, x2, y2, bgr, alpha: float) -> None:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, -1)
        img[:] = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    def _alpha_line(self, img, p1, p2, bgr, thickness: int, alpha: float) -> None:
        overlay = img.copy()
        cv2.line(overlay, p1, p2, bgr, thickness, cv2.LINE_AA)
        img[:] = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    def _dashed_line(self, img, p1, p2, bgr, thickness: int, alpha: float, dash: int = 12, gap: int = 10) -> None:
        x1, y1 = p1
        x2, y2 = p2
        length = int(math.hypot(x2 - x1, y2 - y1))
        if length <= 0:
            return
        vx = (x2 - x1) / float(length)
        vy = (y2 - y1) / float(length)
        cur = 0
        while cur < length:
            s = cur
            e = min(length, cur + dash)
            sx = int(x1 + vx * s)
            sy = int(y1 + vy * s)
            ex = int(x1 + vx * e)
            ey = int(y1 + vy * e)
            self._alpha_line(img, (sx, sy), (ex, ey), bgr, thickness, alpha)
            cur += dash + gap

    def _put_text(self, img, text: str, x: int, y: int, scale: float, thickness: int,
                  color=(255, 255, 255), shadow: bool = True) -> Tuple[int, int]:
        font = self._font()
        th = max(1, thickness)
        text = _ascii_safe(text)
        if shadow and bool(self.get_parameter("overlay_text_shadow").value):
            cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)
        (tw, thh), _ = cv2.getTextSize(text, font, scale, th)
        return int(tw), int(thh)

    def _fit_text(self, text: str, max_px: int, scale: float, thickness: int) -> str:
        """
        Ensure text width <= max_px by trimming and adding '...'.
        """
        font = self._font()
        th = max(1, thickness)
        s = _ascii_safe(text)

        (tw, _), _ = cv2.getTextSize(s, font, scale, th)
        if tw <= max_px:
            return s

        ell = "..."
        (ew, _), _ = cv2.getTextSize(ell, font, scale, th)
        if ew >= max_px:
            return ""

        # binary search length
        lo, hi = 0, len(s)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = s[:mid].rstrip() + ell
            (cw, _), _ = cv2.getTextSize(cand, font, scale, th)
            if cw <= max_px:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _risk_color(self, risk: float) -> Tuple[int, int, int]:
        if risk < 0.35:
            return (0, 200, 80)
        if risk < 0.70:
            return (0, 200, 255)
        return (0, 0, 255)

    def _draw_corridor(self, img) -> None:
        H, W = img.shape[:2]
        a = float(self.get_parameter("overlay_line_alpha").value)
        center_band = float(self.get_parameter("center_band_ratio").value)
        bottom_danger = float(self.get_parameter("bottom_danger_ratio").value)

        teal = self._pcolor("overlay_teal_bgr", (180, 200, 0))

        x0 = int((0.5 - center_band / 2.0) * W)
        x1 = int((0.5 + center_band / 2.0) * W)
        yb = int(bottom_danger * H)

        self._alpha_line(img, (x0, 0), (x0, H), teal, 1, a)
        self._alpha_line(img, (x1, 0), (x1, H), teal, 1, a)
        self._dashed_line(img, (0, yb), (W, yb), teal, 1, a, dash=18, gap=14)

    def _draw_grid(self, img) -> None:
        H, W = img.shape[:2]
        a = float(self.get_parameter("overlay_line_alpha").value)
        grid = self._pcolor("overlay_grid_bgr", (110, 140, 170))
        self._dashed_line(img, (W // 2, 0), (W // 2, H), grid, 1, a, dash=16, gap=14)
        self._dashed_line(img, (0, H // 2), (W, H // 2), grid, 1, a, dash=16, gap=14)

    def _draw_corner_marks(self, img, x1, y1, x2, y2, color, th: int = 2, L: int = 20) -> None:
        cv2.line(img, (x1, y1), (x1 + L, y1), color, th, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + L), color, th, cv2.LINE_AA)

        cv2.line(img, (x2, y1), (x2 - L, y1), color, th, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + L), color, th, cv2.LINE_AA)

        cv2.line(img, (x1, y2), (x1 + L, y2), color, th, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - L), color, th, cv2.LINE_AA)

        cv2.line(img, (x2, y2), (x2 - L, y2), color, th, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - L), color, th, cv2.LINE_AA)

    def _draw_track_box(self, img, tr: Track, is_top: bool) -> None:
        H, W = img.shape[:2]
        x1 = int(tr.cx - tr.w / 2.0)
        y1 = int(tr.cy - tr.h / 2.0)
        x2 = int(tr.cx + tr.w / 2.0)
        y2 = int(tr.cy + tr.h / 2.0)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        risk_c = self._risk_color(float(tr.risk_ema))
        teal = self._pcolor("overlay_teal_bgr", (180, 200, 0))
        txtc = self._pcolor("overlay_text_bgr", (238, 238, 238))
        panel = self._pcolor("overlay_panel_bgr", (12, 18, 32))

        color = risk_c if is_top else teal
        th = 2 if is_top else 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
        if is_top:
            self._draw_corner_marks(img, x1, y1, x2, y2, color, th=2, L=22)

        alpha = float(self.get_parameter("overlay_bbox_chip_alpha").value)
        font = self._font()
        thickness = max(1, int(self.get_parameter("overlay_thickness").value))
        scale = 0.42 if font != cv2.FONT_HERSHEY_PLAIN else 0.85

        cls = _ascii_safe(str(tr.class_id))
        label = f"id{tr.tid} cls={cls} conf={tr.score:.2f} r={tr.risk_ema:.2f}"
        (tw, thh), base = cv2.getTextSize(label, font, scale, thickness)

        chip_x1 = x1
        chip_y2 = max(18, y1 - 2)
        chip_y1 = max(0, chip_y2 - (thh + base + 8))
        chip_x2 = min(W - 1, chip_x1 + tw + 12)

        self._alpha_rect(img, chip_x1, chip_y1, chip_x2, chip_y2, panel, alpha)
        cv2.rectangle(img, (chip_x1, chip_y1), (chip_x2, chip_y2), color, 1)
        cv2.putText(img, label, (chip_x1 + 6, chip_y2 - 4), font, scale, txtc, thickness, cv2.LINE_AA)

    def _draw_bearing_ruler(self, img, top: Optional[Track]) -> None:
        H, W = img.shape[:2]
        hfov = float(self.get_parameter("camera_hfov_deg").value)
        if hfov <= 1e-6:
            return

        ruler_h = max(26, min(64, int(self.get_parameter("overlay_ruler_h_px").value)))
        alpha = clamp(float(self.get_parameter("overlay_ruler_alpha").value), 0.06, 0.65)
        tick_deg = int(self.get_parameter("overlay_ruler_tick_deg").value)
        if tick_deg <= 0:
            tick_deg = 10

        bg = self._pcolor("overlay_panel_bgr", (12, 18, 32))
        teal = self._pcolor("overlay_teal_bgr", (180, 200, 0))
        grid = self._pcolor("overlay_grid_bgr", (110, 140, 170))
        txtc = self._pcolor("overlay_text_bgr", (238, 238, 238))
        muted = self._pcolor("overlay_muted_bgr", (170, 170, 170))

        xL, xR = 10, W - 10
        y1 = 10
        y2 = y1 + ruler_h
        self._alpha_rect(img, xL, y1, xR, y2, bg, alpha)

        # PORT / STBD tags
        font = self._font()
        thickness = max(1, int(self.get_parameter("overlay_thickness").value))
        scale = 0.40 if font != cv2.FONT_HERSHEY_PLAIN else 0.85
        cv2.putText(img, "PORT", (xL + 8, y2 - 10), font, scale, muted, thickness, cv2.LINE_AA)
        tw_stbd = cv2.getTextSize("STBD", font, scale, thickness)[0][0]
        cv2.putText(img, "STBD", (xR - 8 - tw_stbd, y2 - 10), font, scale, muted, thickness, cv2.LINE_AA)

        # center line
        cx = W // 2
        self._alpha_line(img, (cx, y1 + 6), (cx, y2 - 8), teal, 1, 0.45)

        # ticks & labels
        half = hfov * 0.5
        deg = -int(half // tick_deg) * tick_deg
        if deg < -half:
            deg += tick_deg

        while deg <= half + 1e-6:
            xr = 0.5 + (deg / hfov)
            x = int(clamp(xr, 0.0, 1.0) * (W - 1))
            major = (deg % (tick_deg * 2) == 0)
            tlen = 16 if major else 10
            self._alpha_line(img, (x, y2 - 8), (x, y2 - 8 - tlen), grid, 1, 0.40)

            if major:
                lab = f"{deg:+d}"
                tw = cv2.getTextSize(lab, font, scale, thickness)[0][0]
                lx = clampi(x - tw // 2, xL + 6, xR - 6 - tw)  # clamp so label never cut
                cv2.putText(img, lab, (lx, y1 + 20), font, scale, txtc, thickness, cv2.LINE_AA)

            deg += tick_deg

        # target marker (triangle + label)
        if top is not None:
            b = float(top.bearing_deg)
            xr = 0.5 + (b / hfov)
            x = int(clamp(xr, 0.0, 1.0) * (W - 1))
            tri = np.array([[x, y1 + 6], [x - 8, y1 + 22], [x + 8, y1 + 22]], dtype=np.int32)
            overlay = img.copy()
            cv2.fillConvexPoly(overlay, tri, teal)
            img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

            lab = f"BRG {b:+.1f}deg"
            tw = cv2.getTextSize(lab, font, scale, thickness)[0][0]
            lx = clampi(x + 10, xL + 6, xR - 6 - tw)
            cv2.putText(img, lab, (lx, y1 + 24), font, scale, txtc, thickness, cv2.LINE_AA)

    def _draw_proc_chip(self, img, metrics: dict) -> None:
        proc_ms = metrics.get("proc_ms", None)
        if proc_ms is None:
            return

        font = self._font()
        thickness = max(1, int(self.get_parameter("overlay_thickness").value))
        scale = 0.40 if font != cv2.FONT_HERSHEY_PLAIN else 0.85

        txt = f"PROC {float(proc_ms):.0f}ms"
        (tw, thh), base = cv2.getTextSize(txt, font, scale, thickness)

        x1, y1 = 12, 12
        x2, y2 = x1 + tw + 14, y1 + thh + base + 10
        bg = self._pcolor("overlay_panel_bgr", (12, 18, 32))
        teal = self._pcolor("overlay_teal_bgr", (180, 200, 0))
        txtc = self._pcolor("overlay_text_bgr", (238, 238, 238))
        self._alpha_rect(img, x1, y1, x2, y2, bg, 0.38)
        cv2.rectangle(img, (x1, y1), (x2, y2), teal, 1)
        cv2.putText(img, txt, (x1 + 7, y2 - 6), font, scale, txtc, thickness, cv2.LINE_AA)

    def _draw_hud(self, img, metrics: dict, top: Optional[Track]) -> None:
        H, W = img.shape[:2]
        margin = int(self.get_parameter("overlay_margin_px").value)
        pad = int(self.get_parameter("overlay_padding_px").value)
        alpha_bg = float(self.get_parameter("overlay_alpha_bg").value)
        border_th = int(self.get_parameter("overlay_border_thickness").value)

        font = self._font()
        th = max(1, int(self.get_parameter("overlay_thickness").value))
        s_head = float(self.get_parameter("overlay_scale_head").value)
        s_body = float(self.get_parameter("overlay_scale_body").value)

        if font == cv2.FONT_HERSHEY_PLAIN:
            s_head = max(0.85, s_head * 1.7)
            s_body = max(0.75, s_body * 1.7)

        (wAg, hAg), baseAg = cv2.getTextSize("Ag", font, s_body, th)
        line_h = hAg + baseAg + 6

        max_w = int(float(self.get_parameter("overlay_max_width_ratio").value) * W)
        max_w = max(360, min(max_w, W - 2 * margin))

        risk = float(metrics.get("risk", 0.0))
        accent = self._risk_color(risk)

        cmd = _ascii_safe(str(metrics.get("cmd", "HOLD_COURSE")).replace("_", " "))
        vq = float(metrics.get("vision_quality", 1.0))
        ntrk = int(metrics.get("num_tracks", 0))
        avoid = bool(metrics.get("avoid_mode", False))
        situation = _ascii_safe(str(metrics.get("situation", "UNKNOWN")).replace("_", " "))
        vmode = _ascii_safe(str(metrics.get("vision_mode", "NORMAL")))
        colregs = _ascii_safe(str(metrics.get("colregs", "COLREG: --")))

        det_dt = metrics.get("det_dt_ms", None)
        fps = metrics.get("fps", None)
        det_dt_txt = "--" if det_dt is None else f"{float(det_dt):.0f}ms"
        fps_txt = "--" if fps is None else f"{float(fps):.1f}"

        tg = metrics.get("target", None) if isinstance(metrics.get("target", None), dict) else None
        comp = metrics.get("components", {}) if isinstance(metrics.get("components", None), dict) else {}
        topk = metrics.get("topk", []) if isinstance(metrics.get("topk", None), list) else []

        # lines (raw, will be auto-fit later)
        lines: List[str] = []
        lines.append(f"CMD: {cmd}")
        lines.append(f"MODE: {vmode}   AVOID: {'ON' if avoid else 'OFF'}")
        lines.append(f"VQ: {vq:.2f}   TRK: {ntrk}   DET: {det_dt_txt}   FPS: {fps_txt}")
        lines.append(f"SITUATION: {situation}")
        lines.append(f"{colregs}")

        if tg:
            tid = int(tg.get("track_id", 0))
            cls = _ascii_safe(str(tg.get("class_id", "")))
            conf = float(tg.get("score", 0.0))
            x = float(tg.get("x_ratio", 0.0))
            by = float(tg.get("bottom_y_ratio", 0.0))
            area = float(tg.get("area_ratio", 0.0))
            corridor = "YES" if bool(tg.get("in_corridor", False)) else "NO"
            b = float(tg.get("bearing_deg", 0.0))
            br = float(tg.get("bearing_rate_dps", 0.0))
            dlog = float(tg.get("dlog_area_dt", 0.0))
            ttc = tg.get("ttc_proxy_s", None)
            treason = _ascii_safe(str(tg.get("ttc_reason", "")))

            if ttc is None:
                if treason == "no_approach":
                    ttc_txt = "-- (NO APPROACH)"
                elif treason == "disabled":
                    ttc_txt = "-- (OFF)"
                else:
                    ttc_txt = "--"
            else:
                ttc_txt = f"{float(ttc):.1f}s"

            lines.append(f"TARGET: id={tid} cls={cls} conf={conf:.2f} corridor={corridor}")
            lines.append(f"  x={x:.2f} bot={by:.2f} area={area:.3f}")
            lines.append(f"  brg={b:+.1f}deg  rate={br:+.1f}dps  dlog={dlog:+.2f}")
            lines.append(f"  TCPA: {ttc_txt}")
        else:
            lines.append("TARGET: --")

        if comp:
            prox = float(comp.get("prox", 0.0))
            cen = float(comp.get("center", 0.0))
            app = float(comp.get("approach", 0.0))
            bc = float(comp.get("bconst", 0.0))
            lines.append(f"COMP: prox={prox:.2f} cen={cen:.2f} app={app:.2f} bc={bc:.2f}")

        if topk:
            lines.append("TOP:")
            for item in topk[: max(1, int(self.get_parameter("overlay_show_topk").value))]:
                try:
                    rnk = int(item.get("rank", 0))
                    tid = int(item.get("track_id", 0))
                    cls = _ascii_safe(str(item.get("class_id", "")))
                    conf = float(item.get("score", 0.0))
                    rr = float(item.get("risk", 0.0))
                    bb = float(item.get("bearing_deg", 0.0))
                    lines.append(f"  {rnk}) id{tid} cls={cls} c={conf:.2f} r={rr:.2f} b={bb:+.1f}")
                except Exception:
                    continue

        # panel sizing
        header_h = (hAg + baseAg + 10) + 14
        bar_h = int(self.get_parameter("overlay_riskbar_h_px").value)
        body_h = (len(lines) * line_h) + 6
        total_h = pad + header_h + 6 + bar_h + 12 + body_h + pad
        total_h = min(total_h, H - 2 * margin)

        # anchor decision
        anchor = str(self.get_parameter("overlay_anchor").value).lower()
        x_right = W - margin - max_w
        x_left = margin
        y_top = margin

        chosen_x = x_right if anchor != "left" else x_left
        if anchor == "auto" and top is not None:
            tx1 = int(top.cx - top.w / 2.0)
            ty1 = int(top.cy - top.h / 2.0)
            tx2 = int(top.cx + top.w / 2.0)
            ty2 = int(top.cy + top.h / 2.0)
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(W - 1, tx2), min(H - 1, ty2)

            rect_right = (x_right, y_top, x_right + max_w, y_top + total_h)
            rect_left = (x_left, y_top, x_left + max_w, y_top + total_h)

            inter_r = _rect_intersection_area((tx1, ty1, tx2, ty2), rect_right)
            inter_l = _rect_intersection_area((tx1, ty1, tx2, ty2), rect_left)
            tgt_area = max(1, (tx2 - tx1) * (ty2 - ty1))

            chosen_x = x_left if (inter_r / tgt_area) > (inter_l / tgt_area) else x_right

        x1 = int(chosen_x)
        y1 = int(y_top)
        x2 = int(chosen_x + max_w)
        y2 = int(y_top + total_h)

        bg = self._pcolor("overlay_bg_bgr", (16, 24, 40))
        panel = self._pcolor("overlay_panel_bgr", (12, 18, 32))
        teal = self._pcolor("overlay_teal_bgr", (180, 200, 0))
        txtc = self._pcolor("overlay_text_bgr", (238, 238, 238))

        # shadow + panel
        self._alpha_rect(img, x1 + 4, y1 + 4, x2 + 4, y2 + 4, (0, 0, 0), 0.18)
        self._alpha_rect(img, x1, y1, x2, y2, panel, alpha_bg)
        cv2.rectangle(img, (x1, y1), (x2, y2), accent, border_th)

        # header strip (maritime vibe)
        strip_h = 6
        cv2.rectangle(img, (x1, y1), (x2, y1 + strip_h), teal, -1)

        # header text
        hx = x1 + pad
        hy = y1 + pad + (hAg + baseAg + 2)

        title = "SEANO | CA HUD"
        risk_txt = f"RISK {risk:.2f}"
        self._put_text(img, title, hx, hy, s_head, th, txtc, shadow=True)

        (tw_r, _), _ = cv2.getTextSize(risk_txt, font, s_head, th)
        self._put_text(img, risk_txt, x2 - pad - tw_r, hy, s_head, th, txtc, shadow=True)

        # risk bar (full inside)
        bar_w = int(max_w * 0.62)
        bx1 = x1 + pad
        bx2 = bx1 + bar_w
        by1 = y1 + pad + header_h - 8
        by2 = by1 + bar_h

        self._alpha_rect(img, bx1, by1, bx2, by2, bg, 0.55)
        segments = 6
        for i in range(1, segments):
            xx = bx1 + int(bar_w * i / segments)
            cv2.line(img, (xx, by1), (xx, by2), (90, 90, 90), 1, cv2.LINE_AA)

        fill = int(bar_w * clamp(risk, 0.0, 1.0))
        cv2.rectangle(img, (bx1, by1), (bx1 + fill, by2), accent, -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (160, 160, 160), 1)

        # body (AUTO-FIT so nothing overflows)
        avail_w = max(10, (x2 - x1) - 2 * pad)
        cy = by2 + 14 + line_h
        for s in lines:
            fitted = self._fit_text(s, avail_w, s_body, th)
            if fitted:
                self._put_text(img, fitted, hx, cy, s_body, th, txtc, shadow=True)
            cy += line_h

    def _publish_debug_overlay(self, det_msg: Detection2DArray, metrics: dict, top: Optional[Track]) -> None:
        if not bool(self.get_parameter("overlay_enabled").value):
            return
        if self.bridge is None or not _HAS_CV:
            return

        det_stamp_sec = _stamp_to_sec(det_msg.header.stamp)
        img_msg = self._pick_image_for_stamp(det_stamp_sec) if det_stamp_sec > 0 else (self.image_buf[-1] if self.image_buf else None)
        if img_msg is None:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception:
            try:
                img = self.bridge.imgmsg_to_cv2(img_msg)
            except Exception:
                return

        if bool(self.get_parameter("overlay_draw_bearing_ruler").value):
            self._draw_bearing_ruler(img, top)

        if bool(self.get_parameter("overlay_draw_corridor").value):
            self._draw_corridor(img)
        if bool(self.get_parameter("overlay_draw_grid").value):
            self._draw_grid(img)

        if self.tracks:
            for tr in self.tracks.values():
                self._draw_track_box(img, tr, is_top=(top is not None and tr.tid == top.tid))

        self._draw_proc_chip(img, metrics)
        self._draw_hud(img, metrics, top)

        try:
            out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out.header = img_msg.header
            self.pub_dbg.publish(out)
        except Exception:
            return


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RiskEvaluatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()