#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO Collision Avoidance - Risk Evaluator Node (ROS2 Humble)

Subscribe (core):
  - /camera/detections_filtered   (vision_msgs/Detection2DArray)
  - /camera/image_raw_reliable    (sensor_msgs/Image)

Subscribe (vision health / optional but recommended):
  - /vision/quality               (std_msgs/Float32)  -> vision quality external
  - /vision/freeze                (std_msgs/Bool)     -> freeze flag
  - /vision/freeze_reason         (std_msgs/String)   -> e.g. "still", "moving", "timeout"

Publish:
  - /ca/risk            (std_msgs/Float32)
  - /ca/command         (std_msgs/String)
  - /ca/mode            (std_msgs/String)      -> local evaluator mode
  - /ca/metrics         (std_msgs/String JSON)
  - /ca/vision_quality  (std_msgs/Float32)
  - /ca/debug_image     (sensor_msgs/Image)

Notes:
- ASCII-only overlay text (avoid unicode rendering issues in cv2).
- Maritime-style HUD theme (navy/teal), compact & readable.
- Bearing ruler overlay (camera-relative) for navigational feel.
- Image buffer by stamp to better align debug overlay with detections.
- Local evaluator state machine: NORMAL / CAUTION / LOST_PERCEPTION.
- Revised for paper-grade analysis:
    * pseudo-TTC integrated into risk
    * situation-aware maneuver bias
    * richer reason logging
    * vTTC label (not TCPA)
- This node is the risk/perception evaluator layer; it is not the full mission supervisor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import time
from typing import Deque, Dict, List, Optional, Tuple

from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from vision_msgs.msg import Detection2DArray

try:
    import cv2  # type: ignore
    from cv_bridge import CvBridge  # type: ignore
    import numpy as np  # type: ignore

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

        # Topics (core)
        self.declare_parameter("detections_topic", "/camera/detections_filtered")
        self.declare_parameter("image_topic", "/camera/image_raw_reliable")

        # Topics (publish)
        self.declare_parameter("risk_topic", "/ca/risk")
        self.declare_parameter("command_topic", "/ca/command")
        self.declare_parameter("mode_topic", "/ca/mode")
        self.declare_parameter("avoid_active_topic", "/ca/avoid_active")
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
                description="Whitelist class_id. Empty => allow all",
            ),
        )
        self.declare_parameter(
            "deny_class_ids",
            [""],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY, description="Blacklist class_id"
            ),
        )

        # Tracking
        self.declare_parameter("enable_tracking", True)
        self.declare_parameter("iou_match_thresh", 0.25)
        self.declare_parameter("track_timeout_s", 1.0)
        self.declare_parameter("max_tracks", 30)

        # Geometry profile / platform metadata
        self.declare_parameter("geometry_profile_name", "UNSPECIFIED")
        self.declare_parameter("visual_params_source", "UNSPECIFIED")
        self.declare_parameter("vehicle_length_m", 0.70)
        self.declare_parameter("vehicle_beam_m", 0.50)
        self.declare_parameter("camera_height_m", 0.58)
        self.declare_parameter("expected_image_width", 640)
        self.declare_parameter("expected_image_height", 480)
        self.declare_parameter("startup_image_geometry_grace_s", 2.0)

        # Camera proxy
        # These must be supplied explicitly from the active platform YAML.
        # Do not silently inherit legacy SEANO BIMA30 values here.
        self.declare_parameter("camera_hfov_deg", -1.0)

        # Domain
        self.declare_parameter("center_band_ratio", -1.0)
        self.declare_parameter("bottom_danger_ratio", -1.0)
        self.declare_parameter("near_area_ratio", -1.0)

        # Risk weights
        self.declare_parameter("w_proximity", 0.40)
        self.declare_parameter("w_center", 0.18)
        self.declare_parameter("w_approach", 0.16)
        self.declare_parameter("w_bearing_const", 0.10)
        self.declare_parameter("w_ttc", 0.16)
        self.declare_parameter("bearing_rate_bad_dps", 12.0)
        self.declare_parameter("risk_ema_alpha", 0.35)
        self.declare_parameter("vq_risk_floor", 0.80)

        # TTC proxy
        self.declare_parameter("ttc_area_threshold", 0.10)
        self.declare_parameter("ttc_max_s", 60.0)
        self.declare_parameter("ttc_score_horizon_s", 6.0)

        # Avoid hysteresis
        self.declare_parameter("enter_avoid_risk", 0.55)
        self.declare_parameter("exit_avoid_risk", 0.35)
        self.declare_parameter("min_cmd_hold_s", 0.6)

        # Command severity thresholds
        self.declare_parameter("risk_slow_threshold", 0.45)
        self.declare_parameter("risk_turn_slow_threshold", 0.55)
        self.declare_parameter("risk_turn_threshold", 0.75)
        self.declare_parameter("risk_stop_threshold", 0.92)
        self.declare_parameter("vttc_turn_threshold_s", 4.0)
        self.declare_parameter("vttc_stop_threshold_s", 1.2)

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

        # --------------------------
        # Vision Quality sources
        # --------------------------
        self.declare_parameter("use_internal_vision_quality", True)
        self.declare_parameter("vq_check_every_n_frames", 6)

        self.declare_parameter("use_external_vision_quality", True)
        self.declare_parameter("external_vq_topic", "/vision/quality")
        self.declare_parameter("external_vq_timeout_s", 1.5)

        self.declare_parameter("vq_min", 0.25)

        # --------------------------
        # Freeze Detector topics
        # --------------------------
        self.declare_parameter("use_freeze_detector", True)
        self.declare_parameter("freeze_topic", "/vision/freeze")
        self.declare_parameter("freeze_reason_topic", "/vision/freeze_reason")
        self.declare_parameter("freeze_timeout_s", 1.5)

        # --------------------------
        # State machine
        # --------------------------
        self.declare_parameter("tick_hz", 10.0)

        self.declare_parameter("vq_caution_enter", 0.35)
        self.declare_parameter("vq_caution_exit", 0.55)

        self.declare_parameter("image_timeout_s", 2.0)
        self.declare_parameter("lost_dark_vq", 0.25)
        self.declare_parameter("lost_dark_freeze_hold_s", 1.2)

        self.declare_parameter("lost_min_hold_s", 1.0)
        self.declare_parameter("recover_vq", 0.55)
        self.declare_parameter("recover_ok_hold_s", 0.6)

        self.declare_parameter("detections_stale_s", 1.0)

        # --------------------------
        # Image buffer sync
        # --------------------------
        self.declare_parameter("image_buffer_size", 12)
        self.declare_parameter("max_image_age_s", 0.40)

        # --------------------------
        # Overlay (maritime HUD)
        # --------------------------
        self.declare_parameter("overlay_enabled", True)
        self.declare_parameter("overlay_anchor", "auto")
        self.declare_parameter("overlay_max_width_ratio", 0.42)
        self.declare_parameter("overlay_margin_px", 12)
        self.declare_parameter("overlay_padding_px", 10)
        self.declare_parameter("overlay_alpha_bg", 0.80)
        self.declare_parameter("overlay_border_thickness", 1)

        self.declare_parameter("overlay_font_face", "simplex")
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
        self.declare_parameter("overlay_bg_bgr", [16, 24, 40])
        self.declare_parameter("overlay_panel_bgr", [12, 18, 32])
        self.declare_parameter("overlay_teal_bgr", [180, 200, 0])
        self.declare_parameter("overlay_grid_bgr", [110, 140, 170])
        self.declare_parameter("overlay_text_bgr", [238, 238, 238])
        self.declare_parameter("overlay_muted_bgr", [170, 170, 170])

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
        self.node_start_time = time.time()

        # VQ internal/external
        self.vision_quality_internal = 1.0
        self.vision_quality_external = 1.0
        self.vq_ext_last_t = 0.0

        # Freeze detector
        self.freeze_flag = False
        self.freeze_last_t = 0.0
        self.freeze_reason = "unknown"
        self.freeze_reason_last_t = 0.0

        # Received timestamps
        self.last_img_rx_t = 0.0
        self.last_det_rx_t = 0.0

        # Mode state machine
        self.mode = "NORMAL"
        self.lost_since = 0.0
        self.ok_since = 0.0
        self.dark_freeze_since = 0.0

        # Tracking
        self.tracks: Dict[int, Track] = {}
        self.next_tid = 1

        # Avoid command latch
        self.avoid_mode = False
        self.last_cmd = str(self.get_parameter("cmd_hold").value)
        self.last_cmd_time = 0.0

        # Filters
        self.allow_ids: set[str] = set()
        self.deny_ids: set[str] = set()
        self._refresh_filters()

        self.last_det_t: Optional[float] = None

        self.image_buf: Deque[Image] = deque(
            maxlen=max(4, int(self.get_parameter("image_buffer_size").value))
        )

        # Validate startup parameters before continuing
        ok, reason = self._validate_param_values({})
        if not ok:
            raise ValueError(f"Invalid risk_evaluator_node startup parameters: {reason}")

        self.add_on_set_parameters_callback(self._on_params)

        # pubs
        self.pub_risk = self.create_publisher(
            Float32, str(self.get_parameter("risk_topic").value), self.qos
        )
        self.pub_cmd = self.create_publisher(
            String, str(self.get_parameter("command_topic").value), self.qos
        )
        self.pub_mode = self.create_publisher(
            String, str(self.get_parameter("mode_topic").value), self.qos
        )
        self.pub_avoid_active = self.create_publisher(
            Bool, str(self.get_parameter("avoid_active_topic").value), self.qos
        )

        # Sensitive avoid-active latch for low-FPS monocular video.
        # This makes /ca/avoid_active usable by the mode manager even when
        # the detector/risk pulse is only one or two frames long.
        self.declare_parameter("avoid_active_enter_risk", 0.45)
        self.declare_parameter("avoid_active_exit_risk", 0.20)
        self.declare_parameter("avoid_active_hold_s", 0.75)
        self.declare_parameter("avoid_active_force_from_risk", False)
        self._avoid_active_latched = False
        self._avoid_hold_until_sec = 0.0
        self._last_published_risk = 0.0
        self.pub_metrics = self.create_publisher(
            String, str(self.get_parameter("metrics_topic").value), self.qos
        )
        self.pub_vq = self.create_publisher(
            Float32, str(self.get_parameter("vision_quality_topic").value), self.qos
        )
        self.pub_dbg = self.create_publisher(
            Image, str(self.get_parameter("debug_image_topic").value), self.qos
        )

        # subs
        self.sub_det = self.create_subscription(
            Detection2DArray,
            str(self.get_parameter("detections_topic").value),
            self.on_detections,
            self.qos,
        )
        self.sub_img = self.create_subscription(
            Image, str(self.get_parameter("image_topic").value), self.on_raw_image, self.qos
        )

        # optional subs (external VQ + freeze)
        if bool(self.get_parameter("use_external_vision_quality").value):
            self.sub_vq = self.create_subscription(
                Float32, str(self.get_parameter("external_vq_topic").value), self.on_external_vq, 10
            )
        else:
            self.sub_vq = None

        if bool(self.get_parameter("use_freeze_detector").value):
            self.sub_freeze = self.create_subscription(
                Bool, str(self.get_parameter("freeze_topic").value), self.on_freeze, 10
            )
            self.sub_freeze_reason = self.create_subscription(
                String,
                str(self.get_parameter("freeze_reason_topic").value),
                self.on_freeze_reason,
                10,
            )
        else:
            self.sub_freeze = None
            self.sub_freeze_reason = None

        # timer (failsafe tick)
        hz = float(self.get_parameter("tick_hz").value)
        hz = max(1.0, hz)
        self.timer = self.create_timer(1.0 / hz, self.on_tick)

        self.get_logger().info(
            f"risk_evaluator_node started | cv={_HAS_CV} depth={self.qos.depth} "
            f"det={self.get_parameter('detections_topic').value} img={self.get_parameter('image_topic').value} "
            f"profile={self.get_parameter('geometry_profile_name').value} "
            f"visual_src={self.get_parameter('visual_params_source').value} "
            f"expected={self.get_parameter('expected_image_width').value}x"
            f"{self.get_parameter('expected_image_height').value} "
            f"ext_vq={self.get_parameter('use_external_vision_quality').value} "
            f"freeze={self.get_parameter('use_freeze_detector').value}"
        )

    # --------------------------
    # Params
    # --------------------------
    def _candidate_value(self, name: str, incoming: Dict[str, Parameter]):
        if name in incoming:
            return incoming[name].value
        return self.get_parameter(name).value

    def _validate_param_values(self, incoming: Dict[str, Parameter]) -> Tuple[bool, str]:
        try:
            qos_depth = int(self._candidate_value("qos_depth", incoming))
            min_det_score = float(self._candidate_value("min_det_score", incoming))

            enable_tracking = bool(self._candidate_value("enable_tracking", incoming))
            iou_match_thresh = float(self._candidate_value("iou_match_thresh", incoming))
            track_timeout_s = float(self._candidate_value("track_timeout_s", incoming))
            max_tracks = int(self._candidate_value("max_tracks", incoming))

            geometry_profile_name = str(
                self._candidate_value("geometry_profile_name", incoming)
            ).strip()
            visual_params_source = str(
                self._candidate_value("visual_params_source", incoming)
            ).strip()
            vehicle_length_m = float(self._candidate_value("vehicle_length_m", incoming))
            vehicle_beam_m = float(self._candidate_value("vehicle_beam_m", incoming))
            camera_height_m = float(self._candidate_value("camera_height_m", incoming))
            expected_image_width = int(self._candidate_value("expected_image_width", incoming))
            expected_image_height = int(self._candidate_value("expected_image_height", incoming))
            startup_image_geometry_grace_s = float(
                self._candidate_value("startup_image_geometry_grace_s", incoming)
            )

            camera_hfov_deg = float(self._candidate_value("camera_hfov_deg", incoming))
            center_band_ratio = float(self._candidate_value("center_band_ratio", incoming))
            bottom_danger_ratio = float(self._candidate_value("bottom_danger_ratio", incoming))
            near_area_ratio = float(self._candidate_value("near_area_ratio", incoming))

            w_proximity = float(self._candidate_value("w_proximity", incoming))
            w_center = float(self._candidate_value("w_center", incoming))
            w_approach = float(self._candidate_value("w_approach", incoming))
            w_bearing_const = float(self._candidate_value("w_bearing_const", incoming))
            w_ttc = float(self._candidate_value("w_ttc", incoming))
            bearing_rate_bad_dps = float(self._candidate_value("bearing_rate_bad_dps", incoming))
            risk_ema_alpha = float(self._candidate_value("risk_ema_alpha", incoming))
            vq_risk_floor = float(self._candidate_value("vq_risk_floor", incoming))

            ttc_area_threshold = float(self._candidate_value("ttc_area_threshold", incoming))
            ttc_max_s = float(self._candidate_value("ttc_max_s", incoming))
            ttc_score_horizon_s = float(self._candidate_value("ttc_score_horizon_s", incoming))

            enter_avoid_risk = float(self._candidate_value("enter_avoid_risk", incoming))
            exit_avoid_risk = float(self._candidate_value("exit_avoid_risk", incoming))
            min_cmd_hold_s = float(self._candidate_value("min_cmd_hold_s", incoming))

            risk_slow_threshold = float(self._candidate_value("risk_slow_threshold", incoming))
            risk_turn_slow_threshold = float(
                self._candidate_value("risk_turn_slow_threshold", incoming)
            )
            risk_turn_threshold = float(self._candidate_value("risk_turn_threshold", incoming))
            risk_stop_threshold = float(self._candidate_value("risk_stop_threshold", incoming))
            vttc_turn_threshold_s = float(self._candidate_value("vttc_turn_threshold_s", incoming))
            vttc_stop_threshold_s = float(self._candidate_value("vttc_stop_threshold_s", incoming))

            use_internal_vision_quality = bool(
                self._candidate_value("use_internal_vision_quality", incoming)
            )
            vq_check_every_n_frames = int(
                self._candidate_value("vq_check_every_n_frames", incoming)
            )
            use_external_vision_quality = bool(
                self._candidate_value("use_external_vision_quality", incoming)
            )
            external_vq_timeout_s = float(self._candidate_value("external_vq_timeout_s", incoming))
            vq_min = float(self._candidate_value("vq_min", incoming))

            use_freeze_detector = bool(self._candidate_value("use_freeze_detector", incoming))
            freeze_timeout_s = float(self._candidate_value("freeze_timeout_s", incoming))

            tick_hz = float(self._candidate_value("tick_hz", incoming))
            vq_caution_enter = float(self._candidate_value("vq_caution_enter", incoming))
            vq_caution_exit = float(self._candidate_value("vq_caution_exit", incoming))

            image_timeout_s = float(self._candidate_value("image_timeout_s", incoming))
            lost_dark_vq = float(self._candidate_value("lost_dark_vq", incoming))
            lost_dark_freeze_hold_s = float(
                self._candidate_value("lost_dark_freeze_hold_s", incoming)
            )
            lost_min_hold_s = float(self._candidate_value("lost_min_hold_s", incoming))
            recover_vq = float(self._candidate_value("recover_vq", incoming))
            recover_ok_hold_s = float(self._candidate_value("recover_ok_hold_s", incoming))
            detections_stale_s = float(self._candidate_value("detections_stale_s", incoming))

            image_buffer_size = int(self._candidate_value("image_buffer_size", incoming))
            max_image_age_s = float(self._candidate_value("max_image_age_s", incoming))

            overlay_alpha_bg = float(self._candidate_value("overlay_alpha_bg", incoming))
            overlay_border_thickness = int(
                self._candidate_value("overlay_border_thickness", incoming)
            )
            overlay_line_alpha = float(self._candidate_value("overlay_line_alpha", incoming))
            overlay_riskbar_h_px = int(self._candidate_value("overlay_riskbar_h_px", incoming))
            overlay_bbox_chip_alpha = float(
                self._candidate_value("overlay_bbox_chip_alpha", incoming)
            )
            overlay_ruler_h_px = int(self._candidate_value("overlay_ruler_h_px", incoming))
            overlay_ruler_alpha = float(self._candidate_value("overlay_ruler_alpha", incoming))
            overlay_ruler_tick_deg = int(self._candidate_value("overlay_ruler_tick_deg", incoming))
            overlay_show_topk = int(self._candidate_value("overlay_show_topk", incoming))

        except Exception as e:
            return False, f"parameter parse error: {e}"

        if qos_depth < 1:
            return False, "qos_depth must be >= 1"

        if not (0.0 <= min_det_score <= 1.0):
            return False, "min_det_score must be in [0, 1]"

        if enable_tracking:
            if not (0.0 <= iou_match_thresh <= 1.0):
                return False, "iou_match_thresh must be in [0, 1]"
            if track_timeout_s <= 0.0:
                return False, "track_timeout_s must be > 0"
            if max_tracks < 1:
                return False, "max_tracks must be >= 1"

        if not geometry_profile_name:
            return False, "geometry_profile_name must not be empty"
        if not visual_params_source:
            return False, "visual_params_source must not be empty"
        if vehicle_length_m <= 0.0:
            return False, "vehicle_length_m must be > 0"
        if vehicle_beam_m <= 0.0:
            return False, "vehicle_beam_m must be > 0"
        if camera_height_m <= 0.0:
            return False, "camera_height_m must be > 0"
        if expected_image_width < 1 or expected_image_height < 1:
            return False, "expected_image_width and expected_image_height must be >= 1"
        if startup_image_geometry_grace_s < 0.0:
            return False, "startup_image_geometry_grace_s must be >= 0"

        if camera_hfov_deg <= 0.0 or camera_hfov_deg > 179.0:
            return False, "camera_hfov_deg must be in (0, 179]"

        if not (0.0 < center_band_ratio <= 1.0):
            return False, "center_band_ratio must be in (0, 1]"
        if not (0.0 < bottom_danger_ratio <= 1.0):
            return False, "bottom_danger_ratio must be in (0, 1]"
        if near_area_ratio <= 0.0:
            return False, "near_area_ratio must be > 0"

        for name, value in (
            ("w_proximity", w_proximity),
            ("w_center", w_center),
            ("w_approach", w_approach),
            ("w_bearing_const", w_bearing_const),
            ("w_ttc", w_ttc),
        ):
            if value < 0.0:
                return False, f"{name} must be >= 0"

        if bearing_rate_bad_dps <= 0.0:
            return False, "bearing_rate_bad_dps must be > 0"

        if not (0.0 < risk_ema_alpha <= 1.0):
            return False, "risk_ema_alpha must be in (0, 1]"

        if not (0.0 <= vq_risk_floor <= 1.0):
            return False, "vq_risk_floor must be in [0, 1]"

        if ttc_area_threshold <= 0.0:
            return False, "ttc_area_threshold must be > 0"
        if ttc_max_s <= 0.0:
            return False, "ttc_max_s must be > 0"
        if ttc_score_horizon_s <= 0.0:
            return False, "ttc_score_horizon_s must be > 0"

        if not (0.0 <= exit_avoid_risk < enter_avoid_risk <= 1.0):
            return False, "must satisfy 0 <= exit_avoid_risk < enter_avoid_risk <= 1"

        if min_cmd_hold_s < 0.0:
            return False, "min_cmd_hold_s must be >= 0"

        if not (
            0.0
            <= risk_slow_threshold
            <= risk_turn_slow_threshold
            <= risk_turn_threshold
            <= risk_stop_threshold
            <= 1.0
        ):
            return (
                False,
                "must satisfy risk_slow_threshold <= risk_turn_slow_threshold <= "
                "risk_turn_threshold <= risk_stop_threshold, all in [0,1]",
            )

        if vttc_turn_threshold_s <= 0.0 or vttc_stop_threshold_s <= 0.0:
            return False, "vttc_turn_threshold_s and vttc_stop_threshold_s must be > 0"

        if vttc_stop_threshold_s > vttc_turn_threshold_s:
            return False, "vttc_stop_threshold_s should be <= vttc_turn_threshold_s"

        if use_internal_vision_quality and vq_check_every_n_frames < 1:
            return False, "vq_check_every_n_frames must be >= 1"

        if use_external_vision_quality and external_vq_timeout_s <= 0.0:
            return False, "external_vq_timeout_s must be > 0"

        if not (0.0 <= vq_min <= 1.0):
            return False, "vq_min must be in [0, 1]"

        if use_freeze_detector and freeze_timeout_s <= 0.0:
            return False, "freeze_timeout_s must be > 0"

        if tick_hz <= 0.0:
            return False, "tick_hz must be > 0"

        if not (0.0 <= vq_caution_enter < vq_caution_exit <= 1.0):
            return False, "must satisfy 0 <= vq_caution_enter < vq_caution_exit <= 1"

        if image_timeout_s <= 0.0:
            return False, "image_timeout_s must be > 0"

        if not (0.0 <= lost_dark_vq <= 1.0):
            return False, "lost_dark_vq must be in [0, 1]"

        if lost_dark_freeze_hold_s < 0.0:
            return False, "lost_dark_freeze_hold_s must be >= 0"

        if lost_min_hold_s < 0.0:
            return False, "lost_min_hold_s must be >= 0"

        if not (0.0 <= recover_vq <= 1.0):
            return False, "recover_vq must be in [0, 1]"

        if recover_ok_hold_s < 0.0:
            return False, "recover_ok_hold_s must be >= 0"

        if detections_stale_s <= 0.0:
            return False, "detections_stale_s must be > 0"

        if image_buffer_size < 4:
            return False, "image_buffer_size must be >= 4"

        if max_image_age_s <= 0.0:
            return False, "max_image_age_s must be > 0"

        if not (0.0 <= overlay_alpha_bg <= 1.0):
            return False, "overlay_alpha_bg must be in [0, 1]"

        if overlay_border_thickness < 1:
            return False, "overlay_border_thickness must be >= 1"

        if not (0.0 <= overlay_line_alpha <= 1.0):
            return False, "overlay_line_alpha must be in [0, 1]"

        if overlay_riskbar_h_px < 1:
            return False, "overlay_riskbar_h_px must be >= 1"

        if not (0.0 <= overlay_bbox_chip_alpha <= 1.0):
            return False, "overlay_bbox_chip_alpha must be in [0, 1]"

        if overlay_ruler_h_px < 1:
            return False, "overlay_ruler_h_px must be >= 1"

        if not (0.0 <= overlay_ruler_alpha <= 1.0):
            return False, "overlay_ruler_alpha must be in [0, 1]"

        if overlay_ruler_tick_deg < 1:
            return False, "overlay_ruler_tick_deg must be >= 1"

        if overlay_show_topk < 0:
            return False, "overlay_show_topk must be >= 0"

        return True, "ok"

    def _on_params(self, params: List[Parameter]) -> SetParametersResult:
        incoming = {p.name: p for p in params}

        ok, reason = self._validate_param_values(incoming)
        if not ok:
            return SetParametersResult(successful=False, reason=reason)

        try:
            if "allow_class_ids" in incoming or "deny_class_ids" in incoming:
                allow_v = (
                    incoming["allow_class_ids"].value if "allow_class_ids" in incoming else None
                )
                deny_v = incoming["deny_class_ids"].value if "deny_class_ids" in incoming else None
                self._refresh_filters(allow_v=allow_v, deny_v=deny_v)

            if "image_buffer_size" in incoming:
                n = max(4, int(incoming["image_buffer_size"].value))
                self.image_buf = deque(self.image_buf, maxlen=n)

        except Exception as e:
            return SetParametersResult(successful=False, reason=f"param apply error: {e}")

        return SetParametersResult(successful=True, reason="ok")

    def _refresh_filters(self, allow_v=None, deny_v=None) -> None:
        if allow_v is None:
            allow_v = self.get_parameter("allow_class_ids").value
        if deny_v is None:
            deny_v = self.get_parameter("deny_class_ids").value

        self.allow_ids = set(_clean_str_list(allow_v))
        self.deny_ids = set(_clean_str_list(deny_v))
        self.allow_ids.discard("")
        self.deny_ids.discard("")

    # --------------------------
    # External health callbacks
    # --------------------------
    def on_external_vq(self, msg: Float32) -> None:
        self.vision_quality_external = float(msg.data)
        self.vq_ext_last_t = time.time()

    def on_freeze(self, msg: Bool) -> None:
        self.freeze_flag = bool(msg.data)
        self.freeze_last_t = time.time()

    def on_freeze_reason(self, msg: String) -> None:
        self.freeze_reason = str(msg.data)
        self.freeze_reason_last_t = time.time()

    # --------------------------
    # Image callback (buffer + optional internal VQ)
    # --------------------------
    def on_raw_image(self, msg: Image) -> None:
        self.last_img_rx_t = time.time()
        self.image_buf.append(msg)

        if msg.width > 0 and msg.height > 0:
            self.image_w = int(msg.width)
            self.image_h = int(msg.height)

        if not bool(self.get_parameter("use_internal_vision_quality").value):
            return
        if self.bridge is None or not _HAS_CV:
            self.vision_quality_internal = 1.0
            return

        self.frame_count += 1
        n = int(self.get_parameter("vq_check_every_n_frames").value)
        if n < 1:
            n = 1
        if (self.frame_count % n) != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.vision_quality_internal = float(self._compute_vision_quality(frame))
        except Exception:
            self.vision_quality_internal = 0.0

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

        score = (
            (0.35 * blur_score)
            + (0.30 * bright_score)
            + (0.20 * contrast_score)
            + (0.15 * glare_score)
        )
        return clamp(score, 0.0, 1.0)

    def _image_geometry_ready(self, t: float) -> Tuple[bool, str]:
        expected_w = int(self.get_parameter("expected_image_width").value)
        expected_h = int(self.get_parameter("expected_image_height").value)
        grace_s = float(self.get_parameter("startup_image_geometry_grace_s").value)

        if self.image_w is None or self.image_h is None:
            if (t - self.node_start_time) < grace_s:
                return False, "startup_waiting_image"
            return False, "missing_image_dimensions"

        if int(self.image_w) != expected_w or int(self.image_h) != expected_h:
            return (
                False,
                f"image_size_mismatch_{int(self.image_w)}x{int(self.image_h)}_expected_{expected_w}x{expected_h}",
            )

        return True, "ok"

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
    # Detections callback (update tracks; decision also happens here)
    # --------------------------
    def on_detections(self, msg: Detection2DArray) -> None:
        self.last_det_rx_t = time.time()
        self._process_once(det_msg=msg, source="detections_cb")

    # --------------------------
    # Timer tick (failsafe publishing)
    # --------------------------
    def on_tick(self) -> None:
        t = time.time()
        if (t - self.last_det_rx_t) < 0.05:
            return
        self._process_once(det_msg=None, source="timer")

    # --------------------------
    # Core process (shared)
    # --------------------------
    def _process_once(self, det_msg: Optional[Detection2DArray], source: str) -> None:
        t0 = time.time()
        t = t0

        self._update_mode_state(t)

        det_dt_ms = None
        fps = None
        if det_msg is not None:
            if self.last_det_t is not None:
                det_dt_ms = float((t - self.last_det_t) * 1000.0)
                if det_dt_ms > 1e-6:
                    fps = 1000.0 / det_dt_ms
            self.last_det_t = t

        dets: List[Det] = []
        if det_msg is not None:
            dets = self._parse_detections(det_msg)
            dets = self._apply_filters(dets)

            if bool(self.get_parameter("enable_tracking").value):
                self._update_tracks(dets, t)
            else:
                self._tracks_from_dets(dets, t)
        else:
            self._prune_tracks(t)

        overall_risk, top, metrics = self._evaluate(t, det_dt_ms)

        # local_mode explicitly indicates evaluator-local state
        metrics["local_mode"] = self.mode
        if fps is not None:
            metrics["fps"] = float(fps)

        geom_ready = bool(metrics.get("image_geometry_ready", True))
        geom_status = str(metrics.get("image_geometry_status", "ok"))

        if (not geom_ready) and self.mode != "LOST_PERCEPTION":
            if geom_status == "startup_waiting_image":
                cmd = str(self.get_parameter("cmd_hold").value)
                metrics["vision_mode"] = "STARTUP_WAIT_IMAGE"
                metrics["cmd"] = cmd
                metrics["decision_gate"] = "WAIT_IMAGE_GEOMETRY"
                metrics["reason_codes"] = ["WAIT_IMAGE_GEOMETRY"]
            else:
                cmd = str(self.get_parameter("cmd_stop").value)
                overall_risk = max(float(overall_risk), 1.0 if self.last_det_rx_t > 0.0 else 0.60)
                metrics["vision_mode"] = "GEOMETRY_INVALID"
                metrics["cmd"] = cmd
                metrics["decision_gate"] = "FORCED_GEOMETRY_INVALID"
                metrics["reason_codes"] = ["IMAGE_GEOMETRY_INVALID", _ascii_safe(geom_status)]
                self.tracks.clear()
        elif self.mode == "LOST_PERCEPTION":
            cmd = str(self.get_parameter("cmd_stop").value)
            overall_risk = 1.0
            metrics["vision_mode"] = "LOST_PERCEPTION"
            metrics["cmd"] = cmd
            metrics["decision_gate"] = "FORCED_LOST_PERCEPTION"
            metrics["reason_codes"] = ["LOST_PERCEPTION", "FORCED_STOP"]
        else:
            cmd = self._decide_command(t, overall_risk, top, metrics)

            if self.mode == "CAUTION":
                metrics["vision_mode"] = "CAUTION"
                if cmd.startswith("TURN"):
                    cmd = str(self.get_parameter("cmd_slow").value)
                    metrics.setdefault("reason_codes", []).append("CAUTION_DEESCALATE_TURN")
                if overall_risk < 0.25:
                    cmd = str(self.get_parameter("cmd_hold").value)
                    metrics.setdefault("reason_codes", []).append("CAUTION_HOLD_LOW_RISK")
                self._maybe_update_cmd(t, cmd, float(self.get_parameter("min_cmd_hold_s").value))
                cmd = self.last_cmd
                metrics["cmd"] = cmd
            else:
                metrics["vision_mode"] = "NORMAL"

        proc_ms = (time.time() - t0) * 1000.0
        metrics["proc_ms"] = float(proc_ms)
        metrics["source"] = source
        metrics["mode"] = self.mode
        metrics["active_geometry_profile"] = str(self.get_parameter("geometry_profile_name").value)

        self._last_published_risk = float(float(overall_risk))
        self.pub_risk.publish(Float32(data=float(overall_risk)))
        self.pub_cmd.publish(String(data=str(cmd)))
        self.pub_mode.publish(String(data=str(self.mode)))
        self.pub_avoid_active.publish(
            Bool(
                data=self._govern_avoid_active(
                    bool(self._seano_final_avoid_active(locals())),
                    float(getattr(self, "_last_published_risk", 0.0)),
                )
            )
        )

        vq = float(metrics.get("vision_quality", 1.0))
        self.pub_vq.publish(Float32(data=float(vq)))

        try:
            self.pub_metrics.publish(String(data=json.dumps(metrics)))
        except Exception:
            pass

        if bool(self.get_parameter("publish_debug_image").value):
            self._publish_debug_overlay(det_msg, metrics, top)

    # --------------------------
    # Mode state machine
    # --------------------------
    def _get_vq(self, t: float) -> Tuple[float, str]:
        use_ext = bool(self.get_parameter("use_external_vision_quality").value)
        ext_timeout = float(self.get_parameter("external_vq_timeout_s").value)

        if use_ext and (t - self.vq_ext_last_t) <= ext_timeout and self.vq_ext_last_t > 0.0:
            return float(self.vision_quality_external), "external"
        return float(self.vision_quality_internal), "internal"

    def _get_freeze(self, t: float) -> Tuple[bool, str, str]:
        use_fr = bool(self.get_parameter("use_freeze_detector").value)
        fr_timeout = float(self.get_parameter("freeze_timeout_s").value)
        if (not use_fr) or (self.freeze_last_t <= 0.0) or ((t - self.freeze_last_t) > fr_timeout):
            return False, "unknown", "stale_or_disabled"
        reason = "unknown"
        if self.freeze_reason_last_t > 0.0 and (t - self.freeze_reason_last_t) <= fr_timeout:
            reason = str(self.freeze_reason)
        return bool(self.freeze_flag), str(reason), "ok"

    def _update_mode_state(self, t: float) -> None:
        vq, _vq_src = self._get_vq(t)
        freeze, freeze_reason, _freeze_src = self._get_freeze(t)

        image_timeout_s = float(self.get_parameter("image_timeout_s").value)
        img_age = (t - self.last_img_rx_t) if (self.last_img_rx_t > 0.0) else 999.0

        vq_caution_enter = float(self.get_parameter("vq_caution_enter").value)
        vq_caution_exit = float(self.get_parameter("vq_caution_exit").value)

        lost_dark_vq = float(self.get_parameter("lost_dark_vq").value)
        lost_dark_hold = float(self.get_parameter("lost_dark_freeze_hold_s").value)

        lost_min_hold = float(self.get_parameter("lost_min_hold_s").value)
        recover_vq = float(self.get_parameter("recover_vq").value)
        recover_ok_hold = float(self.get_parameter("recover_ok_hold_s").value)

        freeze_timeout = str(freeze_reason).strip().lower() == "timeout"

        lost_trigger = False

        if img_age >= image_timeout_s:
            lost_trigger = True

        if freeze_timeout and img_age >= image_timeout_s * 0.7:
            lost_trigger = True

        dark_freeze = (freeze is True) and (vq <= lost_dark_vq) and (not freeze_timeout)
        if dark_freeze:
            if self.dark_freeze_since <= 0.0:
                self.dark_freeze_since = t
            if (t - self.dark_freeze_since) >= lost_dark_hold:
                lost_trigger = True
        else:
            self.dark_freeze_since = 0.0

        if self.mode == "LOST_PERCEPTION":
            if self.lost_since <= 0.0:
                self.lost_since = t

            if (t - self.lost_since) < lost_min_hold:
                return

            ok = (img_age < image_timeout_s) and (not freeze_timeout) and (vq >= recover_vq)
            if ok:
                if self.ok_since <= 0.0:
                    self.ok_since = t
                if (t - self.ok_since) >= recover_ok_hold:
                    self.mode = "CAUTION" if vq < vq_caution_exit else "NORMAL"
                    self.lost_since = 0.0
                    self.ok_since = 0.0
            else:
                self.ok_since = 0.0
            return

        if lost_trigger:
            self.mode = "LOST_PERCEPTION"
            self.lost_since = t
            self.ok_since = 0.0
            return

        if self.mode == "NORMAL":
            if vq <= vq_caution_enter:
                self.mode = "CAUTION"
        elif self.mode == "CAUTION":
            if vq >= vq_caution_exit:
                self.mode = "NORMAL"

    def _prune_tracks(self, t: float) -> None:
        timeout_s = float(self.get_parameter("track_timeout_s").value)
        dead = [tid for tid, tr in self.tracks.items() if (t - tr.last_t) > timeout_s]
        for tid in dead:
            self.tracks.pop(tid, None)

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
    def _evaluate(
        self, t: float, det_dt_ms: Optional[float]
    ) -> Tuple[float, Optional[Track], dict]:
        geom_ready, geom_status = self._image_geometry_ready(t)

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
        w_ttc = float(self.get_parameter("w_ttc").value)
        risk_alpha = clamp(float(self.get_parameter("risk_ema_alpha").value), 0.01, 1.0)
        vq_risk_floor = clamp(float(self.get_parameter("vq_risk_floor").value), 0.0, 1.0)

        vq, vq_src = self._get_vq(t)
        freeze, freeze_reason, freeze_src = self._get_freeze(t)
        img_age = (t - self.last_img_rx_t) if (self.last_img_rx_t > 0.0) else 999.0
        det_age = (t - self.last_det_rx_t) if (self.last_det_rx_t > 0.0) else 999.0

        metrics: dict = {
            "ts": t,
            "status": "no_detections" if not self.tracks else "ok",
            "risk": 0.0,
            "cmd": str(self.last_cmd),
            "mode": self.mode,
            "local_mode": self.mode,
            "image_geometry_ready": bool(geom_ready),
            "image_geometry_status": str(geom_status),
            "expected_image_width": int(self.get_parameter("expected_image_width").value),
            "expected_image_height": int(self.get_parameter("expected_image_height").value),
            "image_width": None if self.image_w is None else int(self.image_w),
            "image_height": None if self.image_h is None else int(self.image_h),
            "geometry_profile_name": str(self.get_parameter("geometry_profile_name").value),
            "visual_params_source": str(self.get_parameter("visual_params_source").value),
            "vehicle_length_m": float(self.get_parameter("vehicle_length_m").value),
            "vehicle_beam_m": float(self.get_parameter("vehicle_beam_m").value),
            "camera_height_m": float(self.get_parameter("camera_height_m").value),
            "vision_quality": float(vq),
            "vision_quality_source": str(vq_src),
            "freeze": bool(freeze),
            "freeze_reason": _ascii_safe(str(freeze_reason)),
            "freeze_source": str(freeze_src),
            "img_age_s": float(img_age),
            "det_age_s": float(det_age),
            "num_tracks": int(len(self.tracks)),
            "det_dt_ms": det_dt_ms,
            "avoid_mode": bool(self.avoid_mode),
            "risk_ema_alpha": float(risk_alpha),
            "vq_risk_floor": float(vq_risk_floor),
        }

        det_stale_s = float(self.get_parameter("detections_stale_s").value)
        if det_age > det_stale_s and self.mode != "LOST_PERCEPTION":
            self.tracks.clear()
            metrics["status"] = "detections_stale_cleared"

        if not geom_ready:
            metrics["status"] = str(geom_status)
            return 0.0, None, metrics

        if not self.tracks:
            return 0.0, None, metrics

        top: Optional[Track] = None
        best = 0.0
        top_comp = {}
        top_feat = {}

        ranked: List[Tuple[float, Track, dict, dict]] = []

        a_th = float(self.get_parameter("ttc_area_threshold").value)
        ttc_max = float(self.get_parameter("ttc_max_s").value)
        ttc_horizon = max(0.25, float(self.get_parameter("ttc_score_horizon_s").value))

        for tr in self.tracks.values():
            x_ratio = tr.cx / max(W, 1e-9)
            bottom_y_ratio = (tr.cy + tr.h / 2.0) / max(H, 1e-9)
            area_ratio = (tr.w * tr.h) / img_area

            in_corridor = abs(x_ratio - 0.5) <= (center_band / 2.0)
            bottomness = smoothstep(bottom_danger, 1.0, bottom_y_ratio)

            proximity = smoothstep(near_area_ratio * 0.25, near_area_ratio, area_ratio)
            centrality = 1.0 - clamp(abs(x_ratio - 0.5) / max(center_band / 2.0, 1e-6), 0.0, 1.0)
            approach = smoothstep(0.00, 0.55, tr.dlog_area_dt)
            bearing_const = 1.0 - clamp(
                abs(tr.bearing_rate_dps) / max(bearing_rate_bad, 1e-6), 0.0, 1.0
            )

            prox_combo = clamp(0.60 * proximity + 0.40 * bottomness, 0.0, 1.0)
            conf = clamp(tr.score, 0.0, 1.0)

            ttc_proxy = None
            ttc_reason = "disabled"
            ttc_score = 0.0
            if a_th > 1e-9:
                if area_ratio >= a_th:
                    ttc_proxy = 0.0
                    ttc_reason = "at_threshold"
                    ttc_score = 1.0
                elif tr.dlog_area_dt > 1e-6:
                    ttc = (
                        math.log(max(a_th, 1e-9)) - math.log(max(area_ratio, 1e-9))
                    ) / tr.dlog_area_dt
                    ttc = clamp(ttc, 0.0, ttc_max)
                    ttc_proxy = float(ttc)
                    ttc_reason = "ok"
                    ttc_score = 1.0 - smoothstep(0.0, ttc_horizon, ttc)
                else:
                    ttc_reason = "no_approach"
                    ttc_score = 0.0

            raw_geom = (
                w_prox * prox_combo
                + w_center * centrality
                + w_app * approach
                + w_bconst * bearing_const
                + w_ttc * ttc_score
            )
            raw_geom = clamp(raw_geom, 0.0, 1.0)

            conf_scaled = clamp(raw_geom * (0.55 + 0.45 * conf), 0.0, 1.0)

            # Safer VQ scaling:
            # do not let poor vision unrealistically suppress object risk to near-zero.
            vq_scale = clamp(vq_risk_floor + (1.0 - vq_risk_floor) * vq, vq_risk_floor, 1.0)
            raw = clamp(conf_scaled * vq_scale, 0.0, 1.0)

            tr.risk_ema = risk_alpha * raw + (1.0 - risk_alpha) * tr.risk_ema

            comp = {
                "prox": float(prox_combo),
                "center": float(centrality),
                "approach": float(approach),
                "bconst": float(bearing_const),
                "ttc_score": float(ttc_score),
                "conf": float(conf),
                "raw_geom": float(raw_geom),
                "raw_conf": float(conf_scaled),
                "vq_scale": float(vq_scale),
                "raw": float(raw),
                "ema": float(tr.risk_ema),
            }
            feat = {
                "x_ratio": float(x_ratio),
                "bottom_y_ratio": float(bottom_y_ratio),
                "area_ratio": float(area_ratio),
                "in_corridor": bool(in_corridor),
                "vttc_s": ttc_proxy,
                "vttc_reason": ttc_reason,
            }

            ranked.append((float(tr.risk_ema), tr, comp, feat))

            if tr.risk_ema > best:
                best = tr.risk_ema
                top = tr
                top_comp = comp
                top_feat = feat

        if top is None:
            return 0.0, None, metrics

        situation = self._classify_situation(
            bearing_deg=float(top.bearing_deg),
            bearing_rate_dps=float(top.bearing_rate_dps),
            in_corridor=bool(top_feat.get("in_corridor", False)),
        )

        ranked.sort(key=lambda x: x[0], reverse=True)
        topk_n = max(0, int(self.get_parameter("overlay_show_topk").value))
        topk_list = []
        for i, (r, tr, _c, _f) in enumerate(ranked[:topk_n]):
            topk_list.append(
                {
                    "rank": i + 1,
                    "track_id": int(tr.tid),
                    "class_id": str(tr.class_id),
                    "score": float(tr.score),
                    "risk": float(clamp(r, 0.0, 1.0)),
                    "bearing_deg": float(tr.bearing_deg),
                }
            )

        metrics.update(
            {
                "risk": float(clamp(best, 0.0, 1.0)),
                "risk_stage": self._risk_stage(clamp(best, 0.0, 1.0)),
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
                    "vttc_s": top_feat.get("vttc_s"),
                    "vttc_reason": top_feat.get("vttc_reason", "unknown"),
                },
                "components": top_comp,
                "situation": situation,
                "topk": topk_list,
            }
        )

        return float(clamp(best, 0.0, 1.0)), top, metrics

    def _classify_situation(
        self, bearing_deg: float, bearing_rate_dps: float, in_corridor: bool
    ) -> str:
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

    def _risk_stage(self, risk: float) -> str:
        risk = clamp(risk, 0.0, 1.0)
        if risk >= float(self.get_parameter("risk_stop_threshold").value):
            return "EMERGENCY"
        if risk >= float(self.get_parameter("risk_turn_threshold").value):
            return "HIGH"
        if risk >= float(self.get_parameter("risk_turn_slow_threshold").value):
            return "MEDIUM"
        if risk >= float(self.get_parameter("risk_slow_threshold").value):
            return "LOW"
        return "CLEAR"

    def _dominant_factor(self, components: dict) -> str:
        factor_map = {
            "prox": "proximity",
            "center": "centrality",
            "approach": "approach",
            "bconst": "bearing_consistency",
            "ttc_score": "vttc",
        }
        best_k = None
        best_v = -1.0
        for k in ("prox", "center", "approach", "bconst", "ttc_score"):
            v = float(components.get(k, -1.0))
            if v > best_v:
                best_k = k
                best_v = v
        return factor_map.get(best_k or "", "unknown")

    # --------------------------
    # Command + COLREG hint
    # --------------------------
    def _colregs_hint(self, situation: str) -> str:
        if situation == "HEAD_ON":
            return "COLREG_INSPIRED: HEAD-ON -> STARBOARD_BIAS"
        if situation == "CROSSING_RIGHT":
            return "COLREG_INSPIRED: GIVE-WAY_TO_STARBOARD_TARGET"
        if situation == "CROSSING_LEFT":
            return "COLREG_INSPIRED: STAND-ON_PORT_TARGET"
        if situation == "DIVERGING":
            return "COLREG_INSPIRED: DIVERGING"
        return "COLREG_INSPIRED: UNKNOWN"

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

        risk_slow_th = float(self.get_parameter("risk_slow_threshold").value)
        risk_turn_slow_th = float(self.get_parameter("risk_turn_slow_threshold").value)
        risk_turn_th = float(self.get_parameter("risk_turn_threshold").value)
        risk_stop_th = float(self.get_parameter("risk_stop_threshold").value)
        vttc_turn_th = float(self.get_parameter("vttc_turn_threshold_s").value)
        vttc_stop_th = float(self.get_parameter("vttc_stop_threshold_s").value)

        prefer_starboard = bool(self.get_parameter("prefer_starboard").value)
        emergency_turn_away = bool(self.get_parameter("emergency_turn_away").value)

        vq_min = float(self.get_parameter("vq_min").value)
        vq, _ = self._get_vq(t)

        reason_codes: List[str] = []
        decision_gate = "TRACK"

        if not self.avoid_mode and risk >= enter_r:
            self.avoid_mode = True
            decision_gate = "ENTER_AVOID"
            reason_codes.append("ENTER_AVOID")
        elif self.avoid_mode and risk <= exit_r:
            self.avoid_mode = False
            decision_gate = "EXIT_AVOID"
            reason_codes.append("EXIT_AVOID")
        else:
            decision_gate = "HOLD_AVOID" if self.avoid_mode else "TRACK"

        situation = str(metrics.get("situation", "UNKNOWN"))
        metrics["colregs"] = self._colregs_hint(situation)
        metrics["decision_gate"] = decision_gate

        if vq < vq_min:
            desired = cmd_slow if risk > 0.25 else cmd_hold
            self._maybe_update_cmd(t, desired, hold_s)
            metrics["cmd"] = self.last_cmd
            metrics["vision_guard"] = "vq_below_vq_min"
            metrics["reason_codes"] = reason_codes + ["VQ_BELOW_MIN"]
            metrics["dominant_factor"] = self._dominant_factor(metrics.get("components", {}))
            return self.last_cmd

        if top is None:
            self._maybe_update_cmd(t, cmd_hold, hold_s)
            metrics["cmd"] = self.last_cmd
            metrics["reason_codes"] = reason_codes + ["NO_TARGET"]
            metrics["dominant_factor"] = "none"
            return self.last_cmd

        target_metrics = (
            metrics.get("target", {}) if isinstance(metrics.get("target", {}), dict) else {}
        )
        vttc = target_metrics.get("vttc_s", None)
        in_corridor = bool(target_metrics.get("in_corridor", False))
        area_ratio = float(target_metrics.get("area_ratio", 0.0))

        W = float(self.image_w or 1)
        x_ratio = top.cx / max(W, 1e-9)

        extreme = 0.82
        near_area_ratio = float(self.get_parameter("near_area_ratio").value)
        very_close = area_ratio >= (near_area_ratio * 1.2)
        urgent_vttc = (vttc is not None) and (float(vttc) <= vttc_turn_th)
        emergency_vttc = (vttc is not None) and (float(vttc) <= vttc_stop_th)

        if emergency_vttc:
            reason_codes.append("VTTC_EMERGENCY")
        elif urgent_vttc:
            reason_codes.append("VTTC_URGENT")
        if in_corridor:
            reason_codes.append("IN_CORRIDOR")
        if very_close:
            reason_codes.append("VERY_CLOSE")

        # Determine maneuver side with encounter-aware bias
        if situation in ("HEAD_ON", "CROSSING_RIGHT"):
            direction = "RIGHT"
            reason_codes.append(f"{situation}_STARBOARD_BIAS")
        elif situation == "CROSSING_LEFT":
            direction = "RIGHT" if prefer_starboard else "LEFT"
            reason_codes.append("CROSSING_LEFT_STANDON_BIAS")
        elif situation == "DIVERGING":
            direction = "RIGHT" if prefer_starboard else ("LEFT" if x_ratio > 0.5 else "RIGHT")
            reason_codes.append("DIVERGING_CONSERVATIVE")
        else:
            direction = "RIGHT" if prefer_starboard else ("LEFT" if x_ratio > 0.5 else "RIGHT")
            reason_codes.append("UNKNOWN_DEFAULT_BIAS")

        if emergency_turn_away and very_close:
            if x_ratio > extreme:
                direction = "LEFT"
                reason_codes.append("EDGE_RIGHT_TURN_AWAY_LEFT")
            elif x_ratio < (1.0 - extreme):
                direction = "RIGHT"
                reason_codes.append("EDGE_LEFT_TURN_AWAY_RIGHT")

        desired = cmd_hold

        if risk >= risk_stop_th or emergency_vttc:
            desired = cmd_stop
            reason_codes.append("EMERGENCY_STOP")
        elif (not self.avoid_mode) and risk < risk_slow_th and (not urgent_vttc):
            desired = cmd_hold
            reason_codes.append("CLEAR_TRACK")
        else:
            if situation == "DIVERGING" and risk < risk_turn_th and (not urgent_vttc):
                desired = cmd_slow if risk >= risk_slow_th else cmd_hold
                reason_codes.append("DIVERGING_NO_TURN")
            elif situation == "CROSSING_LEFT" and risk < risk_turn_th and (not urgent_vttc):
                # stand-on inspired: avoid overreacting to port-side crossing unless risk becomes high
                desired = cmd_slow if (risk >= risk_slow_th or in_corridor) else cmd_hold
                reason_codes.append("PORT_TARGET_CONSERVATIVE")
            else:
                if risk < risk_turn_slow_th and (not urgent_vttc):
                    desired = cmd_slow
                    reason_codes.append("LOW_SEVERITY_SLOW")
                elif risk < risk_turn_th and (not very_close) and (not urgent_vttc):
                    desired = cmd_trs if direction == "RIGHT" else cmd_tls
                    reason_codes.append(f"TURN_SLOW_{direction}")
                else:
                    desired = cmd_tr if direction == "RIGHT" else cmd_tl
                    reason_codes.append(f"TURN_{direction}")

            if (not in_corridor) and risk < risk_turn_th and (not urgent_vttc):
                desired = cmd_slow
                reason_codes.append("OFF_CORRIDOR_DEESCALATE")

        self._maybe_update_cmd(t, desired, hold_s)
        metrics["cmd"] = self.last_cmd
        metrics["decision"] = {
            "direction": direction,
            "very_close": bool(very_close),
            "urgent_vttc": bool(urgent_vttc),
            "emergency_vttc": bool(emergency_vttc),
            "desired_cmd": desired,
        }
        metrics["reason_codes"] = reason_codes
        metrics["dominant_factor"] = self._dominant_factor(metrics.get("components", {}))
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

    def _dashed_line(
        self, img, p1, p2, bgr, thickness: int, alpha: float, dash: int = 12, gap: int = 10
    ) -> None:
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

    def _put_text(
        self,
        img,
        text: str,
        x: int,
        y: int,
        scale: float,
        thickness: int,
        color=(255, 255, 255),
        shadow: bool = True,
    ) -> Tuple[int, int]:
        font = self._font()
        th = max(1, thickness)
        text = _ascii_safe(text)
        if shadow and bool(self.get_parameter("overlay_text_shadow").value):
            cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)
        (tw, thh), _ = cv2.getTextSize(text, font, scale, th)
        return int(tw), int(thh)

    def _fit_text(self, text: str, max_px: int, scale: float, thickness: int) -> str:
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
        cv2.putText(
            img, label, (chip_x1 + 6, chip_y2 - 4), font, scale, txtc, thickness, cv2.LINE_AA
        )

    def _draw_bearing_ruler(self, img, top: Optional[Track]) -> None:
        if np is None:
            return
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

        font = self._font()
        thickness = max(1, int(self.get_parameter("overlay_thickness").value))
        scale = 0.40 if font != cv2.FONT_HERSHEY_PLAIN else 0.85
        cv2.putText(img, "PORT", (xL + 8, y2 - 10), font, scale, muted, thickness, cv2.LINE_AA)
        tw_stbd = cv2.getTextSize("STBD", font, scale, thickness)[0][0]
        cv2.putText(
            img, "STBD", (xR - 8 - tw_stbd, y2 - 10), font, scale, muted, thickness, cv2.LINE_AA
        )

        cx = W // 2
        self._alpha_line(img, (cx, y1 + 6), (cx, y2 - 8), teal, 1, 0.45)

        half = hfov * 0.5
        deg = -int(half // tick_deg) * tick_deg
        if deg < -half:
            deg += tick_deg

        while deg <= half + 1e-6:
            xr = 0.5 + (deg / hfov)
            x = int(clamp(xr, 0.0, 1.0) * (W - 1))
            major = deg % (tick_deg * 2) == 0
            tlen = 16 if major else 10
            self._alpha_line(img, (x, y2 - 8), (x, y2 - 8 - tlen), grid, 1, 0.40)

            if major:
                lab = f"{deg:+d}"
                tw = cv2.getTextSize(lab, font, scale, thickness)[0][0]
                lx = clampi(x - tw // 2, xL + 6, xR - 6 - tw)
                cv2.putText(img, lab, (lx, y1 + 20), font, scale, txtc, thickness, cv2.LINE_AA)

            deg += tick_deg

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
        vmode = _ascii_safe(str(metrics.get("vision_mode", self.mode)))
        colregs = _ascii_safe(str(metrics.get("colregs", "COLREG: --")))
        dominant_factor = _ascii_safe(str(metrics.get("dominant_factor", "--")))

        det_dt = metrics.get("det_dt_ms", None)
        fps = metrics.get("fps", None)
        det_dt_txt = "--" if det_dt is None else f"{float(det_dt):.0f}ms"
        fps_txt = "--" if fps is None else f"{float(fps):.1f}"
        geom_status = _ascii_safe(str(metrics.get("image_geometry_status", "ok")))

        tg = metrics.get("target", None) if isinstance(metrics.get("target", None), dict) else None
        comp = (
            metrics.get("components", {})
            if isinstance(metrics.get("components", None), dict)
            else {}
        )
        topk = metrics.get("topk", []) if isinstance(metrics.get("topk", None), list) else []
        reason_codes = (
            metrics.get("reason_codes", [])
            if isinstance(metrics.get("reason_codes", None), list)
            else []
        )

        freeze = bool(metrics.get("freeze", False))
        fr = _ascii_safe(str(metrics.get("freeze_reason", "unknown")))
        img_age = float(metrics.get("img_age_s", 0.0))
        risk_stage = _ascii_safe(str(metrics.get("risk_stage", "--")))
        decision_gate = _ascii_safe(str(metrics.get("decision_gate", "--")))

        lines: List[str] = []
        lines.append(f"CMD: {cmd}")
        lines.append(f"MODE: {vmode}   AVOID: {'ON' if avoid else 'OFF'}")
        lines.append(f"VQ: {vq:.2f}   TRK: {ntrk}   DET: {det_dt_txt}   FPS: {fps_txt}")
        lines.append(f"IMG_AGE: {img_age:.2f}s   FREEZE: {str(freeze)}   REASON: {fr}")
        lines.append(f"GEOM: {geom_status}")
        lines.append(f"SITUATION: {situation}   STAGE: {risk_stage}")
        lines.append(f"GATE: {decision_gate}   DOM: {dominant_factor}")
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
            vttc = tg.get("vttc_s", None)
            treason = _ascii_safe(str(tg.get("vttc_reason", "")))

            if vttc is None:
                if treason == "no_approach":
                    vttc_txt = "-- (NO APPROACH)"
                elif treason == "disabled":
                    vttc_txt = "-- (OFF)"
                else:
                    vttc_txt = "--"
            else:
                vttc_txt = f"{float(vttc):.1f}s"

            lines.append(f"TARGET: id={tid} cls={cls} conf={conf:.2f} corridor={corridor}")
            lines.append(f"  x={x:.2f} bot={by:.2f} area={area:.3f}")
            lines.append(f"  brg={b:+.1f}deg  rate={br:+.1f}dps  dlog={dlog:+.2f}")
            lines.append(f"  vTTC: {vttc_txt}")
        else:
            lines.append("TARGET: --")

        if comp:
            prox = float(comp.get("prox", 0.0))
            cen = float(comp.get("center", 0.0))
            app = float(comp.get("approach", 0.0))
            bc = float(comp.get("bconst", 0.0))
            ttc_s = float(comp.get("ttc_score", 0.0))
            lines.append(
                f"COMP: prox={prox:.2f} cen={cen:.2f} app={app:.2f} bc={bc:.2f} vttc={ttc_s:.2f}"
            )

        if reason_codes:
            lines.append("WHY: " + ", ".join([_ascii_safe(str(x)) for x in reason_codes[:4]]))

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

        header_h = (hAg + baseAg + 10) + 14
        bar_h = int(self.get_parameter("overlay_riskbar_h_px").value)
        body_h = (len(lines) * line_h) + 6
        total_h = pad + header_h + 6 + bar_h + 12 + body_h + pad
        total_h = min(total_h, H - 2 * margin)

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

        self._alpha_rect(img, x1 + 4, y1 + 4, x2 + 4, y2 + 4, (0, 0, 0), 0.18)
        self._alpha_rect(img, x1, y1, x2, y2, panel, alpha_bg)
        cv2.rectangle(img, (x1, y1), (x2, y2), accent, border_th)

        strip_h = 6
        cv2.rectangle(img, (x1, y1), (x2, y1 + strip_h), teal, -1)

        hx = x1 + pad
        hy = y1 + pad + (hAg + baseAg + 2)

        title = "SEANO | CA HUD"
        risk_txt = f"RISK {risk:.2f}"
        self._put_text(img, title, hx, hy, s_head, th, txtc, shadow=True)

        (tw_r, _), _ = cv2.getTextSize(risk_txt, font, s_head, th)
        self._put_text(img, risk_txt, x2 - pad - tw_r, hy, s_head, th, txtc, shadow=True)

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

        avail_w = max(10, (x2 - x1) - 2 * pad)
        cy = by2 + 14 + line_h
        for s in lines:
            fitted = self._fit_text(s, avail_w, s_body, th)
            if fitted:
                self._put_text(img, fitted, hx, cy, s_body, th, txtc, shadow=True)
            cy += line_h

    def _publish_debug_overlay(
        self, det_msg: Optional[Detection2DArray], metrics: dict, top: Optional[Track]
    ) -> None:
        if not bool(self.get_parameter("overlay_enabled").value):
            return
        if self.bridge is None or not _HAS_CV:
            return
        if not self.image_buf:
            return

        img_msg: Optional[Image] = None
        if det_msg is not None:
            det_stamp_sec = _stamp_to_sec(det_msg.header.stamp)
            img_msg = (
                self._pick_image_for_stamp(det_stamp_sec)
                if det_stamp_sec > 0
                else (self.image_buf[-1] if self.image_buf else None)
            )
        else:
            img_msg = self.image_buf[-1]

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

    # SEANO_FINAL_AVOID_ACTIVE_GATE_V5
    def _seano_final_avoid_active(self, ctx=None) -> bool:
        """Final authority signal for /ca/avoid_active.

        This must not be raw risk. It must be false when perception/gate
        logic says the system is not allowed to perform avoidance.
        """
        try:
            ctx = ctx or {}

            items = []

            for name in (
                "mode",
                "ca_mode",
                "mode_text",
                "gate",
                "gate_reason",
                "reason",
                "why",
                "cmd",
                "command",
                "safe_cmd",
                "situation",
                "target",
                "status",
                "track_state",
            ):
                if hasattr(self, name):
                    value = getattr(self, name)
                    if isinstance(value, (str, int, float, bool)):
                        items.append(f"{name}={value}")

            for key, value in ctx.items():
                lk = str(key).lower()
                if any(
                    s in lk
                    for s in (
                        "mode",
                        "gate",
                        "reason",
                        "why",
                        "cmd",
                        "command",
                        "situation",
                        "target",
                        "status",
                        "track",
                        "dom",
                    )
                ):
                    if isinstance(value, (str, int, float, bool)):
                        items.append(f"{lk}={value}")

            text = " ".join(items).upper()

            hard_reject = (
                "LOST_PERCEPTION",
                "FORCED_LOST_PERCEPTION",
                "NO_TARGET",
                "TARGET_LOST",
                "STALE",
                "IMAGE_STALE",
                "CAMERA_STALE",
                "FREEZE",
                "FAILSAFE",
                "WATCHDOG",
                "FORCED_STOP",
            )

            if any(token in text for token in hard_reject):
                return False

            positive_names = (
                "avoid_on",
                "avoid_enabled",
                "avoid_authorized",
                "avoid_active_final",
                "final_avoid_active",
                "avoid_decision",
                "avoid_cmd_active",
                "takeover_request",
                "takeover_active",
            )

            for key, value in ctx.items():
                lk = str(key).lower()

                if lk in positive_names:
                    if isinstance(value, bool):
                        return bool(value)

                    if isinstance(value, (int, float)):
                        return float(value) > 0.5

                    if isinstance(value, str):
                        sv = value.strip().upper()
                        if sv in ("ON", "TRUE", "YES", "AVOID", "HOLD_AVOID", "TAKEOVER"):
                            return True
                        if sv in ("OFF", "FALSE", "NO", "MISSION", "REJOIN", "STOP"):
                            return False

            # Fallback only if old code has raw avoid_active. This is still
            # protected by the hard reject gate above.
            for key in ("avoid_active", "raw_avoid_active", "hazard_active", "risk_active"):
                if key in ctx:
                    value = ctx[key]
                    if isinstance(value, bool):
                        return bool(value)
                    if isinstance(value, (int, float)):
                        return float(value) > 0.5
                    if isinstance(value, str):
                        return value.strip().lower() in ("1", "true", "yes", "on", "avoid")

            return False

        except Exception:
            return False

    def _pfloat_safe(self, name: str, default: float) -> float:
        try:
            return float(self.get_parameter(name).value)
        except Exception:
            return float(default)

    def _pbool_safe(self, name: str, default: bool) -> bool:
        try:
            v = self.get_parameter(name).value
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)
        except Exception:
            return bool(default)

    def _govern_avoid_active(self, raw_active: bool, risk_value: float) -> bool:
        """
        Anti-sticky avoid-active governor.

        Design rule:
        - /ca/avoid_active must represent confirmed avoidance intent.
        - Risk spike alone must not keep the system in AVOID forever.
        - If force_from_risk is disabled, risk is not allowed to hold the latch.
        """
        now_s = self.get_clock().now().nanoseconds * 1e-9

        enter_risk = self._pfloat_safe("avoid_active_enter_risk", 0.45)
        exit_risk = self._pfloat_safe("avoid_active_exit_risk", 0.20)
        hold_s = max(0.0, self._pfloat_safe("avoid_active_hold_s", 0.75))
        force_from_risk = self._pbool_safe("avoid_active_force_from_risk", False)

        try:
            risk = max(0.0, min(1.0, float(risk_value)))
        except Exception:
            risk = 0.0

        raw_trigger = bool(raw_active)
        risk_trigger = bool(force_from_risk and risk >= enter_risk)

        trigger = raw_trigger or risk_trigger

        if trigger:
            self._avoid_active_latched = True
            self._avoid_hold_until_sec = max(self._avoid_hold_until_sec, now_s + hold_s)
        else:
            # Important: when force_from_risk is False, risk must not keep the latch alive.
            if now_s >= self._avoid_hold_until_sec:
                self._avoid_active_latched = False
            elif force_from_risk and risk > exit_risk:
                self._avoid_hold_until_sec = max(self._avoid_hold_until_sec, now_s + 0.10)

        return bool(self._avoid_active_latched)


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
