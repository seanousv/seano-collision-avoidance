#!/usr/bin/env python3
"""
SEANO — risk_evaluator_node.py (Vision-only Collision Avoidance)

Apa ini?
- Node ROS2 yang membaca hasil deteksi YOLO (/camera/detections),
  lalu menghitung:
  1) metrik penting (bearing, bearing_rate, area_ratio, TTC proxy, in_corridor, dll)
  2) risk score 0..1 (/ca/risk)
  3) command avoidance (/ca/command)
  4) metrics JSON (/ca/metrics)
  5) (opsional) debug overlay image (/ca/debug_image)

Kenapa ini penting?
- Detektor YOLO hanya memberi “kotak” (bbox). Node ini yang mengubah kotak itu
  menjadi keputusan “harus ngapain” untuk collision avoidance.

Topik default:
- Sub:
  /camera/detections          (vision_msgs/msg/Detection2DArray)
  /camera/image_raw_reliable  (sensor_msgs/msg/Image)  -> untuk ukuran frame + vision quality
  /camera/image_annotated     (sensor_msgs/msg/Image)  -> sumber overlay debug (opsional)
- Pub:
  /ca/risk          (std_msgs/msg/Float32)
  /ca/command       (std_msgs/msg/String)
  /ca/metrics       (std_msgs/msg/String)  -> JSON
  /ca/vision_quality(std_msgs/msg/Float32)
  /ca/debug_image   (sensor_msgs/msg/Image) [opsional]

Run contoh:
  ros2 run seano_vision risk_evaluator_node --ros-args -p publish_debug_image:=true

Filter class contoh (deny person COCO id=0):
  ros2 run seano_vision risk_evaluator_node --ros-args \
    -p deny_class_ids:='["0"]' -p publish_debug_image:=true

CATATAN PENTING (anti error parameter):
- allow_class_ids & deny_class_ids DIPAKSA bertipe STRING_ARRAY (bukan BYTE_ARRAY),
  jadi aman di-set dari CLI dengan format JSON array: '["0","1"]' atau '[]'
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

# OpenCV + cv_bridge optional (node tetap bisa jalan tanpa debug image & vision quality)
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


# ---------------------------
# util
# ---------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    # 0..1 smooth transition
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU for bbox in (cx, cy, w, h) pixels."""
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
    """Convert parameter value to list[str] safely (for STRING_ARRAY)."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            out.append(s)
        return out
    # fallback (shouldn't happen if param type is STRING_ARRAY)
    s = str(v).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


# ---------------------------
# data
# ---------------------------
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

    # derived
    bearing_deg: float = 0.0
    bearing_rate_dps: float = 0.0
    log_area: float = 0.0
    dlog_area_dt: float = 0.0

    # smoothed risk
    risk_ema: float = 0.0


# ---------------------------
# node
# ---------------------------
class RiskEvaluatorNode(Node):
    def __init__(self) -> None:
        super().__init__("risk_evaluator_node")

        # QoS: RELIABLE (biar match dengan pipeline kamu yang stabil saat reliable)
        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------------------------
        # Parameters
        # ---------------------------
        self.declare_parameter("detections_topic", "/camera/detections")
        self.declare_parameter("image_topic", "/camera/image_raw_reliable")
        self.declare_parameter("annotated_topic", "/camera/image_annotated")

        self.declare_parameter("risk_topic", "/ca/risk")
        self.declare_parameter("command_topic", "/ca/command")
        self.declare_parameter("metrics_topic", "/ca/metrics")
        self.declare_parameter("vision_quality_topic", "/ca/vision_quality")
        self.declare_parameter("debug_image_topic", "/ca/debug_image")

        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("min_det_score", 0.35)

        # IMPORTANT: force STRING_ARRAY (anti BYTE_ARRAY mismatch)
        self.declare_parameter(
            "allow_class_ids",
            [""],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="Whitelist class_id (STRING_ARRAY). Empty => allow all.",
            ),
        )
        self.declare_parameter(
            "deny_class_ids",
            [""],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="Blacklist class_id (STRING_ARRAY). Example: ['0'] to ignore person.",
            ),
        )

        # tracking
        self.declare_parameter("enable_tracking", True)
        self.declare_parameter("iou_match_thresh", 0.25)
        self.declare_parameter("track_timeout_s", 1.0)
        self.declare_parameter("max_tracks", 30)

        # camera geometry proxy
        self.declare_parameter("camera_hfov_deg", 70.0)  # default proxy, nanti kamu bisa kalibrasi

        # ship domain in image space
        self.declare_parameter("center_band_ratio", 0.30)   # lebar band tengah (0..1)
        self.declare_parameter("bottom_danger_ratio", 0.65) # y ratio mulai dianggap dekat (0..1)
        self.declare_parameter("near_area_ratio", 0.08)     # bbox area ratio dianggap dekat

        # risk weights
        self.declare_parameter("w_proximity", 0.50)
        self.declare_parameter("w_center", 0.20)
        self.declare_parameter("w_approach", 0.20)
        self.declare_parameter("w_bearing_const", 0.10)

        # bearing rate scaling
        self.declare_parameter("bearing_rate_bad_dps", 12.0)  # > ini dianggap bearing berubah cepat (lebih aman)

        # TTC proxy (berbasis area_ratio)
        self.declare_parameter("ttc_area_threshold", 0.10)  # area_ratio target untuk "sangat dekat"
        self.declare_parameter("ttc_max_s", 60.0)

        # hysteresis + hold-time (anti jitter)
        self.declare_parameter("enter_avoid_risk", 0.55)
        self.declare_parameter("exit_avoid_risk", 0.35)
        self.declare_parameter("min_cmd_hold_s", 0.6)

        # command strings (supaya gampang ganti)
        self.declare_parameter("cmd_hold", "HOLD_COURSE")
        self.declare_parameter("cmd_slow", "SLOW_DOWN")
        self.declare_parameter("cmd_turn_left_slow", "TURN_LEFT_SLOW")
        self.declare_parameter("cmd_turn_right_slow", "TURN_RIGHT_SLOW")
        self.declare_parameter("cmd_turn_left", "TURN_LEFT")
        self.declare_parameter("cmd_turn_right", "TURN_RIGHT")
        self.declare_parameter("cmd_stop", "STOP")

        # behavior preference
        self.declare_parameter("prefer_starboard", True)      # COLREG-ish: prefer belok kanan
        self.declare_parameter("emergency_turn_away", True)   # kalau target sudah ekstrem kanan/kiri dan sangat dekat

        # vision quality
        self.declare_parameter("use_vision_quality", True)
        self.declare_parameter("vq_min", 0.25)               # kalau di bawah ini, lebih konservatif
        self.declare_parameter("vq_check_every_n_frames", 6) # hitung quality tiap N frame

        # ---------------------------
        # Internal state
        # ---------------------------
        self.bridge = CvBridge() if (_HAS_CV and CvBridge is not None) else None
        self.last_raw_msg: Optional[Image] = None
        self.last_ann_msg: Optional[Image] = None
        self.image_w: Optional[int] = None
        self.image_h: Optional[int] = None

        self.frame_count = 0
        self.vision_quality = 1.0

        self.tracks: Dict[int, Track] = {}
        self.next_tid = 1

        self.avoid_mode = False
        self.last_cmd = self.get_parameter("cmd_hold").value
        self.last_cmd_time = 0.0

        # allow/deny cache
        self.allow_ids: set[str] = set()
        self.deny_ids: set[str] = set()
        self._refresh_filters()

        # parameters callback (kalau diubah runtime)
        self.add_on_set_parameters_callback(self._on_params)

        # ---------------------------
        # Publishers
        # ---------------------------
        self.pub_risk = self.create_publisher(Float32, self.get_parameter("risk_topic").value, self.qos)
        self.pub_cmd = self.create_publisher(String, self.get_parameter("command_topic").value, self.qos)
        self.pub_metrics = self.create_publisher(String, self.get_parameter("metrics_topic").value, self.qos)
        self.pub_vq = self.create_publisher(Float32, self.get_parameter("vision_quality_topic").value, self.qos)
        self.pub_dbg = self.create_publisher(Image, self.get_parameter("debug_image_topic").value, self.qos)

        # ---------------------------
        # Subscriptions
        # ---------------------------
        self.sub_det = self.create_subscription(
            Detection2DArray,
            self.get_parameter("detections_topic").value,
            self.on_detections,
            self.qos
        )
        self.sub_img = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.on_raw_image,
            self.qos
        )
        self.sub_ann = self.create_subscription(
            Image,
            self.get_parameter("annotated_topic").value,
            self.on_annotated_image,
            self.qos
        )

        self.get_logger().info(
            "risk_evaluator_node started | "
            f"det={self.get_parameter('detections_topic').value} "
            f"raw={self.get_parameter('image_topic').value} "
            f"ann={self.get_parameter('annotated_topic').value} "
            f"cv={_HAS_CV}"
        )

    # ---------------------------
    # Params callback
    # ---------------------------
    def _on_params(self, params: List[Parameter]) -> SetParametersResult:
        res = SetParametersResult()
        res.successful = True
        res.reason = "ok"
        try:
            names = {p.name for p in params}
            if "allow_class_ids" in names or "deny_class_ids" in names:
                self._refresh_filters()
        except Exception as e:
            self.get_logger().warn(f"param refresh warning: {e}")
        return res

    def _refresh_filters(self) -> None:
        allow_v = self.get_parameter("allow_class_ids").value
        deny_v = self.get_parameter("deny_class_ids").value
        self.allow_ids = set(_clean_str_list(allow_v))
        self.deny_ids = set(_clean_str_list(deny_v))

        # special case: [''] means empty
        self.allow_ids.discard("")
        self.deny_ids.discard("")

    # ---------------------------
    # Image callbacks
    # ---------------------------
    def on_raw_image(self, msg: Image) -> None:
        self.last_raw_msg = msg
        if msg.width > 0 and msg.height > 0:
            self.image_w = int(msg.width)
            self.image_h = int(msg.height)

        # vision quality update (optional)
        if not bool(self.get_parameter("use_vision_quality").value):
            self.vision_quality = 1.0
            return
        if self.bridge is None:
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
            # kalau gagal, jangan matiin node
            self.vision_quality = 0.0
            self.pub_vq.publish(Float32(data=float(self.vision_quality)))

    def on_annotated_image(self, msg: Image) -> None:
        self.last_ann_msg = msg
        if msg.width > 0 and msg.height > 0:
            self.image_w = int(msg.width)
            self.image_h = int(msg.height)

    def _compute_vision_quality(self, bgr) -> float:
        """Return 0..1 score."""
        if not _HAS_CV:
            return 1.0
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

    # ---------------------------
    # Detection callback
    # ---------------------------
    def on_detections(self, msg: Detection2DArray) -> None:
        # need image size
        if self.image_w is None or self.image_h is None:
            return

        t = time.time()
        dets = self._parse_detections(msg)
        dets = self._apply_filters(dets)

        # update tracks
        if bool(self.get_parameter("enable_tracking").value):
            self._update_tracks(dets, t)
        else:
            self._tracks_from_dets(dets, t)

        # evaluate risk + choose target
        overall_risk, top, metrics = self._evaluate(t)

        # command decision
        cmd = self._decide_command(t, overall_risk, top, metrics)

        # publish outputs
        self.pub_risk.publish(Float32(data=float(overall_risk)))
        self.pub_cmd.publish(String(data=str(cmd)))

        # publish metrics JSON (selalu, meskipun kosong)
        try:
            self.pub_metrics.publish(String(data=json.dumps(metrics)))
        except Exception:
            # jangan matiin node
            pass

        # debug image overlay (optional)
        if bool(self.get_parameter("publish_debug_image").value):
            self._publish_debug_overlay(metrics, top)

    # ---------------------------
    # Parse & filter detections
    # ---------------------------
    def _parse_detections(self, msg: Detection2DArray) -> List[Det]:
        out: List[Det] = []
        min_score = float(self.get_parameter("min_det_score").value)

        for d in msg.detections:
            if len(d.results) == 0:
                continue

            # ambil hypothesis terbaik
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

            if best_cid is None:
                continue
            if best_score < min_score:
                continue

            # bbox center (vision_msgs uses Pose2D -> center.position.x/y)
            try:
                cx = float(d.bbox.center.position.x)
                cy = float(d.bbox.center.position.y)
            except Exception:
                # fallback (very unlikely)
                cx = float(getattr(d.bbox.center, "x", 0.0))
                cy = float(getattr(d.bbox.center, "y", 0.0))

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

    # ---------------------------
    # Tracking
    # ---------------------------
    def _tracks_from_dets(self, dets: List[Det], t: float) -> None:
        self.tracks.clear()
        self.next_tid = 1
        for d in dets[: int(self.get_parameter("max_tracks").value)]:
            tr = self._make_track(self.next_tid, d, t, prev=None)
            self.tracks[self.next_tid] = tr
            self.next_tid += 1

    def _update_tracks(self, dets: List[Det], t: float) -> None:
        # drop old tracks
        timeout_s = float(self.get_parameter("track_timeout_s").value)
        dead = [tid for tid, tr in self.tracks.items() if (t - tr.last_t) > timeout_s]
        for tid in dead:
            self.tracks.pop(tid, None)

        # match by IoU (greedy)
        iou_thr = float(self.get_parameter("iou_match_thresh").value)

        # build candidates (iou, tid, det_index)
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
            if j >= len(dets):
                continue

            # update existing track
            prev = self.tracks.get(tid)
            if prev is None:
                continue
            self.tracks[tid] = self._make_track(tid, dets[j], t, prev=prev)

            used_tids.add(tid)
            used_det.add(j)

        # create new tracks for unmatched dets
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
        # compute bearing from x_ratio (proxy using HFOV)
        W = float(self.image_w or 1)
        H = float(self.image_h or 1)
        x_ratio = d.cx / max(W, 1e-9)
        hfov = float(self.get_parameter("camera_hfov_deg").value)
        bearing_deg = (x_ratio - 0.5) * hfov

        # compute log-area & expansion
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

    # ---------------------------
    # Risk evaluation
    # ---------------------------
    def _evaluate(self, t: float) -> Tuple[float, Optional[Track], dict]:
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

        # vision quality factor
        use_vq = bool(self.get_parameter("use_vision_quality").value)
        vq = float(self.vision_quality) if use_vq else 1.0
        self.pub_vq.publish(Float32(data=float(vq)))

        top: Optional[Track] = None
        best = 0.0

        # default metrics (no detections)
        metrics: dict = {
            "ts": t,
            "status": "no_detections",
            "risk": 0.0,
            "cmd": str(self.last_cmd),
            "vision_quality": vq,
            "num_tracks": len(self.tracks),
        }

        if not self.tracks:
            return 0.0, None, metrics

        for tr in self.tracks.values():
            x_ratio = tr.cx / max(W, 1e-9)
            y_ratio = tr.cy / max(H, 1e-9)
            bottom_y_ratio = (tr.cy + tr.h / 2.0) / max(H, 1e-9)
            area_ratio = (tr.w * tr.h) / img_area

            # ship-domain cues (image-space)
            in_corridor = abs(x_ratio - 0.5) <= (center_band / 2.0)
            bottomness = smoothstep(bottom_danger, 1.0, bottom_y_ratio)

            # proximity from area ratio
            proximity = smoothstep(near_area_ratio * 0.25, near_area_ratio, area_ratio)

            # centrality score
            centrality = 1.0 - clamp(abs(x_ratio - 0.5) / max(center_band / 2.0, 1e-6), 0.0, 1.0)

            # approach from bbox expansion rate (dlog_area_dt)
            approach = smoothstep(0.00, 0.55, tr.dlog_area_dt)  # >0 means expanding

            # bearing constancy: small bearing rate => higher collision risk
            bearing_const = 1.0 - clamp(abs(tr.bearing_rate_dps) / max(bearing_rate_bad, 1e-6), 0.0, 1.0)

            # combine proximity with bottomness (lebih “marine-like”)
            prox_combo = clamp(0.60 * proximity + 0.40 * bottomness, 0.0, 1.0)

            # confidence factor
            conf = clamp(tr.score, 0.0, 1.0)

            raw = (
                w_prox * prox_combo +
                w_center * centrality +
                w_app * approach +
                w_bconst * bearing_const
            )
            raw = clamp(raw * (0.55 + 0.45 * conf), 0.0, 1.0)

            # down-weight by vision quality (kalau jelek, jangan terlalu agresif)
            raw = clamp(raw * (0.5 + 0.5 * vq), 0.0, 1.0)

            # EMA smoothing per track
            alpha = 0.35
            tr.risk_ema = alpha * raw + (1.0 - alpha) * tr.risk_ema

            if tr.risk_ema > best:
                best = tr.risk_ema
                top = tr

        # build metrics for top
        if top is not None:
            x_ratio = top.cx / max(W, 1e-9)
            y_ratio = top.cy / max(H, 1e-9)
            bottom_y_ratio = (top.cy + top.h / 2.0) / max(H, 1e-9)
            area_ratio = (top.w * top.h) / img_area
            in_corridor = abs(x_ratio - 0.5) <= (center_band / 2.0)

            # TTC proxy using dlog(area)/dt:
            # if dlog_area_dt > 0 then area grows exponentially-ish => TTC approx:
            # ttc = (log(A_th/A_now)) / dlog_dt
            ttc_max = float(self.get_parameter("ttc_max_s").value)
            a_th = float(self.get_parameter("ttc_area_threshold").value)
            ttc_proxy = None
            if top.dlog_area_dt > 1e-6 and area_ratio > 1e-9 and a_th > area_ratio:
                ttc = (math.log(max(a_th, 1e-9)) - math.log(max(area_ratio, 1e-9))) / top.dlog_area_dt
                ttc = clamp(ttc, 0.0, ttc_max)
                ttc_proxy = float(ttc)

            metrics = {
                "ts": t,
                "status": "ok",
                "risk": float(clamp(best, 0.0, 1.0)),
                "cmd": str(self.last_cmd),
                "vision_quality": float(vq),
                "num_tracks": len(self.tracks),

                "target": {
                    "track_id": int(top.tid),
                    "class_id": str(top.class_id),
                    "score": float(top.score),

                    "x_ratio": float(x_ratio),
                    "y_ratio": float(y_ratio),
                    "bottom_y_ratio": float(bottom_y_ratio),
                    "area_ratio": float(area_ratio),

                    "bearing_deg": float(top.bearing_deg),
                    "bearing_rate_dps": float(top.bearing_rate_dps),
                    "dlog_area_dt": float(top.dlog_area_dt),

                    "in_corridor": bool(in_corridor),
                    "ttc_proxy_s": ttc_proxy,
                }
            }
            return float(clamp(best, 0.0, 1.0)), top, metrics

        return 0.0, None, metrics

    # ---------------------------
    # Command decision (with hysteresis)
    # ---------------------------
    def _decide_command(self, t: float, risk: float, top: Optional[Track], metrics: dict) -> str:
        cmd_hold = self.get_parameter("cmd_hold").value
        cmd_slow = self.get_parameter("cmd_slow").value
        cmd_tls = self.get_parameter("cmd_turn_left_slow").value
        cmd_trs = self.get_parameter("cmd_turn_right_slow").value
        cmd_tl = self.get_parameter("cmd_turn_left").value
        cmd_tr = self.get_parameter("cmd_turn_right").value
        cmd_stop = self.get_parameter("cmd_stop").value

        enter_r = float(self.get_parameter("enter_avoid_risk").value)
        exit_r = float(self.get_parameter("exit_avoid_risk").value)
        hold_s = float(self.get_parameter("min_cmd_hold_s").value)

        prefer_starboard = bool(self.get_parameter("prefer_starboard").value)
        emergency_turn_away = bool(self.get_parameter("emergency_turn_away").value)

        vq_min = float(self.get_parameter("vq_min").value)
        vq = float(self.vision_quality)

        # update avoid_mode hysteresis
        if not self.avoid_mode and risk >= enter_r:
            self.avoid_mode = True
        elif self.avoid_mode and risk <= exit_r:
            self.avoid_mode = False

        # if vision quality is too low => conservative (slow / hold)
        if bool(self.get_parameter("use_vision_quality").value) and (vq < vq_min):
            cmd = cmd_slow if risk > 0.25 else cmd_hold
            self._maybe_update_cmd(t, cmd, hold_s)
            metrics["cmd"] = str(self.last_cmd)
            return self.last_cmd

        # no target
        if top is None:
            self._maybe_update_cmd(t, cmd_hold, hold_s)
            metrics["cmd"] = str(self.last_cmd)
            return self.last_cmd

        # target data
        W = float(self.image_w or 1)
        x_ratio = top.cx / max(W, 1e-9)
        in_corridor = False
        try:
            in_corridor = bool(metrics.get("target", {}).get("in_corridor", False))
        except Exception:
            in_corridor = False

        # emergency stop if extremely high risk
        if risk >= 0.92:
            self._maybe_update_cmd(t, cmd_stop, hold_s)
            metrics["cmd"] = str(self.last_cmd)
            return self.last_cmd

        # If not in avoid mode and risk small => hold
        if (not self.avoid_mode) and risk < 0.45:
            self._maybe_update_cmd(t, cmd_hold, hold_s)
            metrics["cmd"] = str(self.last_cmd)
            return self.last_cmd

        # Decide turning direction
        # - "COLREG-ish default": prefer starboard (turn right)
        # - "turn away" emergency: if object already extreme right & very close => turn left to avoid immediate collision
        extreme = 0.82
        very_close = False
        try:
            area_ratio = float(metrics.get("target", {}).get("area_ratio", 0.0))
            near_area_ratio = float(self.get_parameter("near_area_ratio").value)
            very_close = area_ratio >= (near_area_ratio * 1.2)
        except Exception:
            very_close = False

        # choose base direction
        if prefer_starboard:
            direction = "RIGHT"
        else:
            direction = "LEFT" if x_ratio > 0.5 else "RIGHT"  # turn away from obstacle

        # emergency override: if obstacle is extreme on one side AND very close -> turn away
        if emergency_turn_away and very_close:
            if x_ratio > extreme:
                direction = "LEFT"
            elif x_ratio < (1.0 - extreme):
                direction = "RIGHT"

        # action intensity
        # - if risk medium: slow down (or slow turn if in corridor)
        # - if risk high: turn slow
        # - if risk very high: turn fast
        if risk < 0.55:
            cmd = cmd_slow
        elif risk < 0.75:
            cmd = cmd_trs if direction == "RIGHT" else cmd_tls
        else:
            cmd = cmd_tr if direction == "RIGHT" else cmd_tl

        # If target is NOT in corridor, and risk medium, prefer slow (avoid unnecessary turns)
        if (not in_corridor) and risk < 0.70:
            cmd = cmd_slow

        self._maybe_update_cmd(t, cmd, hold_s)
        metrics["cmd"] = str(self.last_cmd)
        return self.last_cmd

    def _maybe_update_cmd(self, t: float, desired: str, hold_s: float) -> None:
        # anti flip-flop: command hanya boleh berubah setelah minimal hold_s
        if desired == self.last_cmd:
            return
        if (t - self.last_cmd_time) < hold_s:
            return
        self.last_cmd = desired
        self.last_cmd_time = t

    # ---------------------------
    # Debug overlay
    # ---------------------------
    def _publish_debug_overlay(self, metrics: dict, top: Optional[Track]) -> None:
        if self.bridge is None or not _HAS_CV:
            return

        # choose annotated image if available else raw
        src_msg = self.last_ann_msg if self.last_ann_msg is not None else self.last_raw_msg
        if src_msg is None:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(src_msg, desired_encoding="bgr8")
        except Exception:
            try:
                img = self.bridge.imgmsg_to_cv2(src_msg)
            except Exception:
                return

        H, W = img.shape[:2]

        # draw ship-domain corridor + bottom danger line
        center_band = float(self.get_parameter("center_band_ratio").value)
        x0 = int((0.5 - center_band / 2.0) * W)
        x1 = int((0.5 + center_band / 2.0) * W)
        cv2.rectangle(img, (x0, 0), (x1, H), (0, 255, 255), 1)

        bottom_danger = float(self.get_parameter("bottom_danger_ratio").value)
        yb = int(bottom_danger * H)
        cv2.line(img, (0, yb), (W, yb), (0, 255, 255), 1)

        # text
        risk = float(metrics.get("risk", 0.0))
        cmd = str(metrics.get("cmd", self.last_cmd))
        vq = float(metrics.get("vision_quality", self.vision_quality))
        txt1 = f"risk={risk:.2f} cmd={cmd}"
        txt2 = f"vq={vq:.2f} tracks={len(self.tracks)} mode={'AVOID' if self.avoid_mode else 'NORMAL'}"
        cv2.putText(img, txt1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(img, txt2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # highlight top target
        if top is not None:
            x1b = int(top.cx - top.w / 2.0)
            y1b = int(top.cy - top.h / 2.0)
            x2b = int(top.cx + top.w / 2.0)
            y2b = int(top.cy + top.h / 2.0)
            x1b, y1b = max(0, x1b), max(0, y1b)
            x2b, y2b = min(W - 1, x2b), min(H - 1, y2b)
            cv2.rectangle(img, (x1b, y1b), (x2b, y2b), (0, 0, 255), 3)

            br = float(top.bearing_deg)
            brate = float(top.bearing_rate_dps)
            dlog = float(top.dlog_area_dt)
            cv2.putText(
                img,
                f"id={top.tid} cls={top.class_id} score={top.score:.2f} b={br:.1f}deg brate={brate:.1f} dlog={dlog:.2f}",
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        # publish
        try:
            out = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            out.header = src_msg.header
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
