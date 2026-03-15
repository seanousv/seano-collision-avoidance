#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEANO - demo_full_ca.launch.py

Default baseline:
  USB camera minimal CA pipeline (hardware-friendly)

Default active path:
  (optional) camera include launch
  detector -> /camera/detections
  risk_evaluator -> /ca/risk + /ca/command + /ca/mode + /ca/debug_image
  watchdog_failsafe -> /ca/failsafe_active + /ca/failsafe_reason (+ status)

Optional full-pipeline modules (disabled by default):
  waterline_horizon -> /vision/waterline_y + /vision/waterline_debug
  false_positive_guard -> /camera/detections_filtered
  multi_target_fusion -> /camera/detections_fused
  vision_quality -> /vision/quality (+ detail)
  frame_freeze_detector -> /vision/freeze (+ score + reason)

Viewer:
  showimage /ca/debug_image
  showimage /vision/waterline_debug
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _effective_risk_input_topic(
    explicit_topic: LaunchConfiguration,
    use_fusion: LaunchConfiguration,
    use_fp_guard: LaunchConfiguration,
    detections_raw_topic: LaunchConfiguration,
    detections_filtered_topic: LaunchConfiguration,
    detections_fused_topic: LaunchConfiguration,
) -> PythonExpression:
    """
    Pilih input detections untuk risk node secara otomatis:
      1) kalau explicit_topic diisi -> pakai itu
      2) kalau fusion aktif       -> pakai fused
      3) kalau fp_guard aktif     -> pakai filtered
      4) selain itu               -> pakai raw
    """
    return PythonExpression(
        [
            "'",
            explicit_topic,
            "' if '",
            explicit_topic,
            "' != '' else (",
            "'",
            detections_fused_topic,
            "' if '",
            use_fusion,
            "'.lower() == 'true' else (",
            "'",
            detections_filtered_topic,
            "' if '",
            use_fp_guard,
            "'.lower() == 'true' else '",
            detections_raw_topic,
            "'",
            "))",
        ]
    )


def generate_launch_description():
    # -------------------------
    # Toggles
    # -------------------------
    use_camera = LaunchConfiguration("use_camera")
    use_detector = LaunchConfiguration("use_detector")
    use_waterline = LaunchConfiguration("use_waterline")
    use_fp_guard = LaunchConfiguration("use_fp_guard")
    use_fusion = LaunchConfiguration("use_fusion")
    use_vq = LaunchConfiguration("use_vq")
    use_freeze = LaunchConfiguration("use_freeze")
    use_risk = LaunchConfiguration("use_risk")
    use_watchdog = LaunchConfiguration("use_watchdog")

    use_ca_viewer = LaunchConfiguration("use_ca_viewer")
    use_wl_viewer = LaunchConfiguration("use_wl_viewer")

    # -------------------------
    # Camera include
    # -------------------------
    camera_launch = LaunchConfiguration("camera_launch")
    pkg_share = FindPackageShare("seano_vision")

    camera_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", camera_launch])
        ),
        condition=IfCondition(use_camera),
    )

    # -------------------------
    # Topics
    # -------------------------
    image_topic = LaunchConfiguration("image_topic")
    annotated_topic = LaunchConfiguration("annotated_topic")

    detections_raw_topic = LaunchConfiguration("detections_raw_topic")
    detections_filtered_topic = LaunchConfiguration("detections_filtered_topic")
    detections_fused_topic = LaunchConfiguration("detections_fused_topic")

    # Boleh dioverride caller. Kalau kosong, akan dipilih otomatis.
    detections_for_risk_topic = LaunchConfiguration("detections_for_risk_topic")

    waterline_topic = LaunchConfiguration("waterline_topic")
    waterline_debug_topic = LaunchConfiguration("waterline_debug_topic")

    vq_topic = LaunchConfiguration("vq_topic")
    vq_detail_topic = LaunchConfiguration("vq_detail_topic")

    freeze_topic = LaunchConfiguration("freeze_topic")
    freeze_score_topic = LaunchConfiguration("freeze_score_topic")
    freeze_reason_topic = LaunchConfiguration("freeze_reason_topic")

    risk_topic = LaunchConfiguration("risk_topic")
    command_topic = LaunchConfiguration("command_topic")
    mode_topic = LaunchConfiguration("mode_topic")
    metrics_topic = LaunchConfiguration("metrics_topic")
    debug_image_topic = LaunchConfiguration("debug_image_topic")

    # -------------------------
    # Detector QoS
    # -------------------------
    det_sub_reliability = LaunchConfiguration("det_sub_reliability")
    det_pub_reliability = LaunchConfiguration("det_pub_reliability")
    det_qos_depth = LaunchConfiguration("det_qos_depth")

    # -------------------------
    # Detector runtime/config
    # -------------------------
    det_model_path = LaunchConfiguration("det_model_path")
    det_device = LaunchConfiguration("det_device")
    det_imgsz = LaunchConfiguration("det_imgsz")
    det_conf = LaunchConfiguration("det_conf")
    det_iou = LaunchConfiguration("det_iou")
    det_class_ids = LaunchConfiguration("det_class_ids")
    det_max_det = LaunchConfiguration("det_max_det")
    det_agnostic_nms = LaunchConfiguration("det_agnostic_nms")
    det_half = LaunchConfiguration("det_half")
    det_warmup = LaunchConfiguration("det_warmup")
    det_max_fps = LaunchConfiguration("det_max_fps")
    det_publish_annotated = LaunchConfiguration("det_publish_annotated")
    det_publish_detections = LaunchConfiguration("det_publish_detections")
    det_publish_empty_detections = LaunchConfiguration("det_publish_empty_detections")

    # -------------------------
    # FP Guard args
    # -------------------------
    fp_use_waterline = LaunchConfiguration("fp_use_waterline")
    fp_waterline_margin_px = LaunchConfiguration("fp_waterline_margin_px")
    fp_min_score = LaunchConfiguration("fp_min_score")
    fp_min_area_px = LaunchConfiguration("fp_min_area_px")
    fp_window_size = LaunchConfiguration("fp_window_size")
    fp_min_hits = LaunchConfiguration("fp_min_hits")
    fp_iou_match = LaunchConfiguration("fp_iou_match")
    fp_max_miss = LaunchConfiguration("fp_max_miss")

    # -------------------------
    # Fusion args
    # -------------------------
    fusion_enabled = LaunchConfiguration("fusion_enabled")
    fusion_mode = LaunchConfiguration("fusion_mode")
    fusion_top_k = LaunchConfiguration("fusion_top_k")

    # -------------------------
    # Watchdog args
    # -------------------------
    wd_startup_grace_s = LaunchConfiguration("wd_startup_grace_s")
    wd_start_in_failsafe = LaunchConfiguration("wd_start_in_failsafe")

    # -------------------------
    # Effective risk input topic
    # -------------------------
    effective_detections_for_risk_topic = _effective_risk_input_topic(
        explicit_topic=detections_for_risk_topic,
        use_fusion=use_fusion,
        use_fp_guard=use_fp_guard,
        detections_raw_topic=detections_raw_topic,
        detections_filtered_topic=detections_filtered_topic,
        detections_fused_topic=detections_fused_topic,
    )

    # -------------------------
    # Nodes
    # -------------------------

    detector_node = Node(
        package="seano_vision",
        executable="detector_node",
        name="detector_node",
        output="screen",
        condition=IfCondition(use_detector),
        parameters=[
            {
                "sub_image": image_topic,
                "pub_image": annotated_topic,
                "pub_det": detections_raw_topic,
                "publish_annotated": ParameterValue(det_publish_annotated, value_type=bool),
                "publish_detections": ParameterValue(det_publish_detections, value_type=bool),
                "publish_empty_detections": ParameterValue(
                    det_publish_empty_detections, value_type=bool
                ),
                "model_path": det_model_path,
                "device": det_device,
                "imgsz": ParameterValue(det_imgsz, value_type=int),
                "conf": ParameterValue(det_conf, value_type=float),
                "iou": ParameterValue(det_iou, value_type=float),
                "class_ids": det_class_ids,
                "max_det": ParameterValue(det_max_det, value_type=int),
                "agnostic_nms": ParameterValue(det_agnostic_nms, value_type=bool),
                "half": ParameterValue(det_half, value_type=bool),
                "warmup": ParameterValue(det_warmup, value_type=bool),
                "max_fps": ParameterValue(det_max_fps, value_type=float),
                "qos_depth": ParameterValue(det_qos_depth, value_type=int),
                "sub_reliability": det_sub_reliability,
                "pub_det_reliability": det_pub_reliability,
                "pub_image_reliability": det_pub_reliability,
            }
        ],
    )

    waterline_node = Node(
        package="seano_vision",
        executable="waterline_horizon_node",
        name="waterline_horizon_node",
        output="screen",
        condition=IfCondition(use_waterline),
        parameters=[
            {
                "input_topic": image_topic,
                "waterline_topic": waterline_topic,
                "debug_topic": waterline_debug_topic,
                "enable_debug": True,
                "publish_mask": True,
                "default_ratio": 0.35,
                "ema_alpha": 0.25,
                "process_every_n": 1,
                "downscale_width": 480,
            }
        ],
    )

    fp_guard_node = Node(
        package="seano_vision",
        executable="false_positive_guard_node",
        name="false_positive_guard_node",
        output="screen",
        condition=IfCondition(use_fp_guard),
        parameters=[
            {
                "enabled": True,
                "input_topic": detections_raw_topic,
                "output_topic": detections_filtered_topic,
                "use_waterline": ParameterValue(fp_use_waterline, value_type=bool),
                "waterline_topic": waterline_topic,
                "waterline_margin_px": ParameterValue(fp_waterline_margin_px, value_type=int),
                "min_score": ParameterValue(fp_min_score, value_type=float),
                "min_area_px": ParameterValue(fp_min_area_px, value_type=float),
                "window_size": ParameterValue(fp_window_size, value_type=int),
                "min_hits": ParameterValue(fp_min_hits, value_type=int),
                "iou_match": ParameterValue(fp_iou_match, value_type=float),
                "max_miss": ParameterValue(fp_max_miss, value_type=int),
            }
        ],
    )

    fusion_node = Node(
        package="seano_vision",
        executable="multi_target_fusion_node",
        name="multi_target_fusion_node",
        output="screen",
        condition=IfCondition(use_fusion),
        parameters=[
            {
                "enabled": ParameterValue(fusion_enabled, value_type=bool),
                "input_topic": detections_filtered_topic,
                "output_topic": detections_fused_topic,
                "image_topic": image_topic,
                "output_mode": fusion_mode,
                "top_k": ParameterValue(fusion_top_k, value_type=int),
                "w_bottom": 0.55,
                "w_area": 0.25,
                "w_center": 0.20,
                "w_det_score": 0.05,
                "use_tracking": True,
                "iou_match": 0.35,
                "max_miss": 6,
                "bbox_ema_alpha": 0.40,
                "score_ema_alpha": 0.30,
            }
        ],
    )

    vq_node = Node(
        package="seano_vision",
        executable="vision_quality_node",
        name="vision_quality_node",
        output="screen",
        condition=IfCondition(use_vq),
        parameters=[
            {
                "input_topic": image_topic,
                "quality_topic": vq_topic,
                "detail_topic": vq_detail_topic,
                "publish_detail": True,
                "downsample_w": 320,
            }
        ],
    )

    freeze_node = Node(
        package="seano_vision",
        executable="frame_freeze_detector_node",
        name="frame_freeze_detector_node",
        output="screen",
        condition=IfCondition(use_freeze),
        parameters=[
            {
                "input_topic": image_topic,
                "freeze_topic": freeze_topic,
                "score_topic": freeze_score_topic,
                "reason_topic": freeze_reason_topic,
                "diff_threshold": 2.5,
                "consecutive_frames": 15,
                "no_frame_timeout_s": 2.0,
                "timer_hz": 5.0,
            }
        ],
    )

    risk_node = Node(
        package="seano_vision",
        executable="risk_evaluator_node",
        name="risk_evaluator_node",
        output="screen",
        condition=IfCondition(use_risk),
        parameters=[
            {
                "detections_topic": effective_detections_for_risk_topic,
                "image_topic": image_topic,
                "risk_topic": risk_topic,
                "command_topic": command_topic,
                "mode_topic": mode_topic,
                "metrics_topic": metrics_topic,
                "debug_image_topic": debug_image_topic,
                "publish_debug_image": True,
                "use_external_vision_quality": ParameterValue(use_vq, value_type=bool),
                "external_vq_topic": vq_topic,
                "use_freeze_detector": ParameterValue(use_freeze, value_type=bool),
                "freeze_topic": freeze_topic,
                "freeze_reason_topic": freeze_reason_topic,
            }
        ],
    )

    watchdog_node = Node(
        package="seano_vision",
        executable="watchdog_failsafe_node",
        name="watchdog_failsafe_node",
        output="screen",
        condition=IfCondition(use_watchdog),
        parameters=[
            {
                "image_topic": image_topic,
                "detections_topic": detections_raw_topic,
                "risk_topic": risk_topic,
                "command_topic": command_topic,
                "mode_topic": mode_topic,
                "startup_grace_s": ParameterValue(wd_startup_grace_s, value_type=float),
                "start_in_failsafe": ParameterValue(wd_start_in_failsafe, value_type=bool),
            }
        ],
    )

    ca_viewer = Node(
        package="image_tools",
        executable="showimage",
        name="show_ca_debug",
        output="screen",
        condition=IfCondition(use_ca_viewer),
        remappings=[("image", debug_image_topic)],
    )

    wl_viewer = Node(
        package="image_tools",
        executable="showimage",
        name="show_waterline_debug",
        output="screen",
        condition=IfCondition(use_wl_viewer),
        remappings=[("image", waterline_debug_topic)],
    )

    return LaunchDescription(
        [
            # toggles - default sekarang selaras dengan baseline hardware USB
            DeclareLaunchArgument("use_camera", default_value="true"),
            DeclareLaunchArgument("use_detector", default_value="true"),
            DeclareLaunchArgument("use_waterline", default_value="false"),
            DeclareLaunchArgument("use_fp_guard", default_value="false"),
            DeclareLaunchArgument("use_fusion", default_value="false"),
            DeclareLaunchArgument("use_vq", default_value="false"),
            DeclareLaunchArgument("use_freeze", default_value="false"),
            DeclareLaunchArgument("use_risk", default_value="true"),
            DeclareLaunchArgument("use_watchdog", default_value="true"),
            DeclareLaunchArgument("use_ca_viewer", default_value="false"),
            DeclareLaunchArgument("use_wl_viewer", default_value="false"),

            # camera include - default tidak lagi ke kamera HP lama
            DeclareLaunchArgument(
                "camera_launch",
                default_value="phase2_camera_usb_test.launch.py",
            ),

            # image topic - selaras dengan baseline hardware sekarang
            DeclareLaunchArgument(
                "image_topic",
                default_value="/seano/camera/image_raw_reliable",
            ),

            # topics
            DeclareLaunchArgument("annotated_topic", default_value="/camera/image_annotated"),
            DeclareLaunchArgument("detections_raw_topic", default_value="/camera/detections"),
            DeclareLaunchArgument(
                "detections_filtered_topic",
                default_value="/camera/detections_filtered",
            ),
            DeclareLaunchArgument(
                "detections_fused_topic",
                default_value="/camera/detections_fused",
            ),

            # kosong = auto select by enabled pipeline stage
            DeclareLaunchArgument(
                "detections_for_risk_topic",
                default_value="",
            ),

            DeclareLaunchArgument("waterline_topic", default_value="/vision/waterline_y"),
            DeclareLaunchArgument(
                "waterline_debug_topic",
                default_value="/vision/waterline_debug",
            ),
            DeclareLaunchArgument("vq_topic", default_value="/vision/quality"),
            DeclareLaunchArgument("vq_detail_topic", default_value="/vision/quality_detail"),
            DeclareLaunchArgument("freeze_topic", default_value="/vision/freeze"),
            DeclareLaunchArgument("freeze_score_topic", default_value="/vision/freeze_score"),
            DeclareLaunchArgument("freeze_reason_topic", default_value="/vision/freeze_reason"),
            DeclareLaunchArgument("risk_topic", default_value="/ca/risk"),
            DeclareLaunchArgument("command_topic", default_value="/ca/command"),
            DeclareLaunchArgument("mode_topic", default_value="/ca/mode"),
            DeclareLaunchArgument("metrics_topic", default_value="/ca/metrics"),
            DeclareLaunchArgument("debug_image_topic", default_value="/ca/debug_image"),

            # detector QoS
            DeclareLaunchArgument("det_sub_reliability", default_value="reliable"),
            DeclareLaunchArgument("det_pub_reliability", default_value="reliable"),
            DeclareLaunchArgument("det_qos_depth", default_value="10"),

            # detector runtime/config
            DeclareLaunchArgument("det_model_path", default_value="yolov8n.pt"),
            DeclareLaunchArgument(
                "det_device",
                default_value="",
                description="Empty string = auto device selection by Ultralytics",
            ),
            DeclareLaunchArgument("det_imgsz", default_value="416"),
            DeclareLaunchArgument("det_conf", default_value="0.20"),
            DeclareLaunchArgument("det_iou", default_value="0.45"),
            DeclareLaunchArgument("det_class_ids", default_value="ALL"),
            DeclareLaunchArgument("det_max_det", default_value="50"),
            DeclareLaunchArgument("det_agnostic_nms", default_value="false"),
            DeclareLaunchArgument("det_half", default_value="false"),
            DeclareLaunchArgument("det_warmup", default_value="true"),
            DeclareLaunchArgument("det_max_fps", default_value="8.0"),
            DeclareLaunchArgument("det_publish_annotated", default_value="true"),
            DeclareLaunchArgument("det_publish_detections", default_value="true"),
            DeclareLaunchArgument("det_publish_empty_detections", default_value="true"),

            # FP guard args
            DeclareLaunchArgument("fp_use_waterline", default_value="true"),
            DeclareLaunchArgument("fp_waterline_margin_px", default_value="15"),
            DeclareLaunchArgument("fp_min_score", default_value="0.25"),
            DeclareLaunchArgument("fp_min_area_px", default_value="900"),
            DeclareLaunchArgument("fp_window_size", default_value="8"),
            DeclareLaunchArgument("fp_min_hits", default_value="3"),
            DeclareLaunchArgument("fp_iou_match", default_value="0.35"),
            DeclareLaunchArgument("fp_max_miss", default_value="4"),

            # fusion args
            DeclareLaunchArgument("fusion_enabled", default_value="true"),
            DeclareLaunchArgument("fusion_mode", default_value="topk"),
            DeclareLaunchArgument("fusion_top_k", default_value="3"),

            # watchdog args
            DeclareLaunchArgument("wd_startup_grace_s", default_value="3.0"),
            DeclareLaunchArgument("wd_start_in_failsafe", default_value="false"),

            # actions
            camera_include,
            detector_node,
            waterline_node,
            fp_guard_node,
            fusion_node,
            vq_node,
            freeze_node,
            risk_node,
            watchdog_node,
            ca_viewer,
            wl_viewer,
        ]
    )