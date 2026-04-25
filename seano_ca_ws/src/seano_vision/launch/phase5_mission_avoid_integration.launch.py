#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _bool_by_profile(
    profile_lc: LaunchConfiguration, enabled_profiles: list[str]
) -> PythonExpression:
    profile_list = ", ".join([f"'{p}'" for p in enabled_profiles])
    return PythonExpression(
        [
            "'true' if '",
            profile_lc,
            f"' in [{profile_list}] else 'false'",
        ]
    )


def _str_by_profile(
    profile_lc: LaunchConfiguration,
    full_value: str,
    synthetic_value: str,
) -> PythonExpression:
    return PythonExpression(
        [
            f"'{full_value}' if '",
            profile_lc,
            "' == 'full' else '",
            synthetic_value,
            "'",
        ]
    )


def _and_true(*lcs: LaunchConfiguration) -> PythonExpression:
    pieces = []
    for i, lc in enumerate(lcs):
        if i > 0:
            pieces.append(" and ")
        pieces.extend(["'", lc, "'.lower() == 'true'"])
    return PythonExpression(["'true' if (", *pieces, ") else 'false'"])


def generate_launch_description():
    pkg_share = FindPackageShare("seano_vision")

    default_video_demo = PathJoinSubstitution(
        [
            EnvironmentVariable("HOME"),
            "seano-collision-avoidance",
            "seano_ca_ws",
            "test_media",
            "videodemo.mp4",
        ]
    )

    # ------------------------------------------------------------------
    # Common args
    # ------------------------------------------------------------------
    record = LaunchConfiguration("record")
    bag_name = LaunchConfiguration("bag_name")
    master_enable_on_start = LaunchConfiguration("master_enable_on_start")
    failsafe_stale_is_active = LaunchConfiguration("failsafe_stale_is_active")

    input_mode = LaunchConfiguration("input_mode")
    output_mode = LaunchConfiguration("output_mode")
    avoid_mode = LaunchConfiguration("avoid_mode")
    mission_mode_default = LaunchConfiguration("mission_mode_default")
    failsafe_mode = LaunchConfiguration("failsafe_mode")

    use_ca_pipeline = LaunchConfiguration("use_ca_pipeline")
    use_takeover_manager = LaunchConfiguration("use_takeover_manager")
    use_mode_manager = LaunchConfiguration("use_mode_manager")

    # ------------------------------------------------------------------
    # Runtime profile
    # ------------------------------------------------------------------
    ca_runtime_profile = LaunchConfiguration("ca_runtime_profile")

    # ------------------------------------------------------------------
    # Camera args
    # ------------------------------------------------------------------
    ca_camera_launch = LaunchConfiguration("ca_camera_launch")
    ca_image_topic = LaunchConfiguration("ca_image_topic")
    ca_use_camera = LaunchConfiguration("ca_use_camera")

    ca_camera_profile = LaunchConfiguration("ca_camera_profile")
    ca_camera_source = LaunchConfiguration("ca_camera_source")
    ca_camera_backend = LaunchConfiguration("ca_camera_backend")
    ca_camera_url = LaunchConfiguration("ca_camera_url")
    ca_camera_pipeline = LaunchConfiguration("ca_camera_pipeline")

    ca_camera_device_path = LaunchConfiguration("ca_camera_device_path")
    ca_camera_device_index = LaunchConfiguration("ca_camera_device_index")
    ca_camera_device_fourcc = LaunchConfiguration("ca_camera_device_fourcc")
    ca_camera_device_width = LaunchConfiguration("ca_camera_device_width")
    ca_camera_device_height = LaunchConfiguration("ca_camera_device_height")
    ca_camera_device_fps = LaunchConfiguration("ca_camera_device_fps")

    ca_camera_topic_best_effort = LaunchConfiguration("ca_camera_topic_best_effort")
    ca_camera_topic_reliable = LaunchConfiguration("ca_camera_topic_reliable")
    ca_camera_frame_id = LaunchConfiguration("ca_camera_frame_id")
    ca_camera_max_fps = LaunchConfiguration("ca_camera_max_fps")
    ca_camera_max_age_ms = LaunchConfiguration("ca_camera_max_age_ms")

    ca_camera_record = LaunchConfiguration("ca_camera_record")
    ca_camera_bag_base_dir = LaunchConfiguration("ca_camera_bag_base_dir")
    ca_camera_bag_prefix = LaunchConfiguration("ca_camera_bag_prefix")
    ca_camera_duration_s = LaunchConfiguration("ca_camera_duration_s")

    # ------------------------------------------------------------------
    # CA / detector / risk args
    # ------------------------------------------------------------------
    ca_use_detector = LaunchConfiguration("ca_use_detector")
    ca_use_risk = LaunchConfiguration("ca_use_risk")
    ca_use_watchdog = LaunchConfiguration("ca_use_watchdog")
    ca_use_ca_viewer = LaunchConfiguration("ca_use_ca_viewer")

    ca_risk_profile = LaunchConfiguration("ca_risk_profile")
    risk_profile_file = PathJoinSubstitution([pkg_share, "config", ca_risk_profile])

    ca_annotated_topic = LaunchConfiguration("ca_annotated_topic")
    ca_detections_topic = LaunchConfiguration("ca_detections_topic")
    ca_risk_topic = LaunchConfiguration("ca_risk_topic")
    ca_command_topic = LaunchConfiguration("ca_command_topic")
    ca_mode_topic = LaunchConfiguration("ca_mode_topic")
    ca_metrics_topic = LaunchConfiguration("ca_metrics_topic")
    ca_debug_image_topic = LaunchConfiguration("ca_debug_image_topic")
    ca_publish_debug_image = LaunchConfiguration("ca_publish_debug_image")

    ca_det_sub_reliability = LaunchConfiguration("ca_det_sub_reliability")
    ca_det_pub_reliability = LaunchConfiguration("ca_det_pub_reliability")
    ca_det_qos_depth = LaunchConfiguration("ca_det_qos_depth")

    ca_det_model_path = LaunchConfiguration("ca_det_model_path")
    ca_det_device = LaunchConfiguration("ca_det_device")
    ca_det_imgsz = LaunchConfiguration("ca_det_imgsz")
    ca_det_conf = LaunchConfiguration("ca_det_conf")
    ca_det_iou = LaunchConfiguration("ca_det_iou")
    ca_det_class_ids = LaunchConfiguration("ca_det_class_ids")
    ca_det_max_det = LaunchConfiguration("ca_det_max_det")
    ca_det_agnostic_nms = LaunchConfiguration("ca_det_agnostic_nms")
    ca_det_half = LaunchConfiguration("ca_det_half")
    ca_det_warmup = LaunchConfiguration("ca_det_warmup")
    ca_det_max_fps = LaunchConfiguration("ca_det_max_fps")
    ca_det_publish_annotated = LaunchConfiguration("ca_det_publish_annotated")
    ca_det_publish_detections = LaunchConfiguration("ca_det_publish_detections")
    ca_det_publish_empty_detections = LaunchConfiguration("ca_det_publish_empty_detections")

    wd_startup_grace_s = LaunchConfiguration("wd_startup_grace_s")
    wd_start_in_failsafe = LaunchConfiguration("wd_start_in_failsafe")

    # ------------------------------------------------------------------
    # Optional camera include
    # ------------------------------------------------------------------
    camera_include_plain = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", ca_camera_launch])
        ),
        condition=IfCondition(
            PythonExpression(
                [
                    "'true' if ('",
                    ca_use_camera,
                    "'.lower() == 'true' and '",
                    ca_camera_launch,
                    "' != 'phase2_camera_source_test.launch.py') else 'false'",
                ]
            )
        ),
    )

    camera_include_passthrough = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", ca_camera_launch])
        ),
        condition=IfCondition(
            PythonExpression(
                [
                    "'true' if ('",
                    ca_use_camera,
                    "'.lower() == 'true' and '",
                    ca_camera_launch,
                    "' == 'phase2_camera_source_test.launch.py') else 'false'",
                ]
            )
        ),
        launch_arguments={
            "profile": ca_camera_profile,
            "source": ca_camera_source,
            "backend": ca_camera_backend,
            "url": ca_camera_url,
            "pipeline": ca_camera_pipeline,
            "device_path": ca_camera_device_path,
            "device_index": ca_camera_device_index,
            "device_fourcc": ca_camera_device_fourcc,
            "device_width": ca_camera_device_width,
            "device_height": ca_camera_device_height,
            "device_fps": ca_camera_device_fps,
            "topic_best_effort": ca_camera_topic_best_effort,
            "topic_reliable": ca_camera_topic_reliable,
            "frame_id": ca_camera_frame_id,
            "max_fps": ca_camera_max_fps,
            "max_age_ms": ca_camera_max_age_ms,
            "record": ca_camera_record,
            "bag_base_dir": ca_camera_bag_base_dir,
            "bag_prefix": ca_camera_bag_prefix,
            "duration_s": ca_camera_duration_s,
        }.items(),
    )

    # ------------------------------------------------------------------
    # Detector / risk / watchdog
    # ------------------------------------------------------------------
    detector_node = Node(
        package="seano_vision",
        executable="detector_node",
        name="detector_node",
        output="screen",
        condition=IfCondition(_and_true(use_ca_pipeline, ca_use_detector)),
        parameters=[
            {
                "sub_image": ca_image_topic,
                "pub_image": ca_annotated_topic,
                "pub_det": ca_detections_topic,
                "publish_annotated": ParameterValue(ca_det_publish_annotated, value_type=bool),
                "publish_detections": ParameterValue(ca_det_publish_detections, value_type=bool),
                "publish_empty_detections": ParameterValue(
                    ca_det_publish_empty_detections, value_type=bool
                ),
                "model_path": ca_det_model_path,
                "device": ca_det_device,
                "imgsz": ParameterValue(ca_det_imgsz, value_type=int),
                "conf": ParameterValue(ca_det_conf, value_type=float),
                "iou": ParameterValue(ca_det_iou, value_type=float),
                "class_ids": ca_det_class_ids,
                "max_det": ParameterValue(ca_det_max_det, value_type=int),
                "agnostic_nms": ParameterValue(ca_det_agnostic_nms, value_type=bool),
                "half": ParameterValue(ca_det_half, value_type=bool),
                "warmup": ParameterValue(ca_det_warmup, value_type=bool),
                "max_fps": ParameterValue(ca_det_max_fps, value_type=float),
                "qos_depth": ParameterValue(ca_det_qos_depth, value_type=int),
                "sub_reliability": ca_det_sub_reliability,
                "pub_det_reliability": ca_det_pub_reliability,
                "pub_image_reliability": ca_det_pub_reliability,
            }
        ],
    )

    risk_node = Node(
        package="seano_vision",
        executable="risk_evaluator_node",
        name="risk_evaluator_node",
        output="screen",
        condition=IfCondition(_and_true(use_ca_pipeline, ca_use_risk)),
        parameters=[
            risk_profile_file,
            {
                "detections_topic": ca_detections_topic,
                "image_topic": ca_image_topic,
                "risk_topic": ca_risk_topic,
                "command_topic": ca_command_topic,
                "mode_topic": ca_mode_topic,
                "metrics_topic": ca_metrics_topic,
                "debug_image_topic": ca_debug_image_topic,
                "publish_debug_image": ParameterValue(ca_publish_debug_image, value_type=bool),
                "use_external_vision_quality": False,
                "use_freeze_detector": False,
            },
        ],
    )

    watchdog_node = Node(
        package="seano_vision",
        executable="watchdog_failsafe_node",
        name="watchdog_failsafe_node",
        output="screen",
        condition=IfCondition(_and_true(use_ca_pipeline, ca_use_watchdog)),
        parameters=[
            {
                "image_topic": ca_image_topic,
                "detections_topic": ca_detections_topic,
                "risk_topic": ca_risk_topic,
                "command_topic": ca_command_topic,
                "mode_topic": ca_mode_topic,
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
        condition=IfCondition(ca_use_ca_viewer),
        remappings=[("image", ca_debug_image_topic)],
    )

    # ------------------------------------------------------------------
    # mux -> limiter -> bridge
    # ------------------------------------------------------------------
    mux = Node(
        package="seano_vision",
        executable="command_mux_node",
        name="command_mux_node",
        output="screen",
        parameters=[
            {
                "manual_left_topic": "/seano/manual/left_cmd",
                "manual_right_topic": "/seano/manual/right_cmd",
                "auto_left_topic": "/seano/auto/left_cmd",
                "auto_right_topic": "/seano/auto/right_cmd",
                "out_left_topic": "/seano/selected/left_cmd",
                "out_right_topic": "/seano/selected/right_cmd",
                "auto_enable_topic": "/seano/auto_enable",
                "fallback_to_manual": True,
                "command_timeout_s": 0.6,
                "allow_reverse": False,
            }
        ],
    )

    limiter = Node(
        package="seano_vision",
        executable="actuator_safety_limiter_node",
        name="actuator_safety_limiter_node",
        output="screen",
        parameters=[
            {
                "in_left_topic": "/seano/selected/left_cmd",
                "in_right_topic": "/seano/selected/right_cmd",
                "out_left_topic": "/seano/left_cmd",
                "out_right_topic": "/seano/right_cmd",
                "failsafe_active_topic": "/ca/failsafe_active",
                "failsafe_stale_is_active": ParameterValue(
                    failsafe_stale_is_active, value_type=bool
                ),
                "allow_reverse": False,
                "input_timeout_s": 0.6,
                "failsafe_timeout_s": 2.0,
                "loop_hz": 20.0,
                "reason_topic": "/seano/limiter_reason",
            }
        ],
    )

    bridge = Node(
        package="seano_vision",
        executable="mavros_rc_override_bridge_node",
        name="mavros_rc_override_bridge_node",
        output="screen",
        parameters=[
            {
                "input_mode": input_mode,
                "output_mode": output_mode,
                "left_topic": "/seano/left_cmd",
                "right_topic": "/seano/right_cmd",
                "out_topic": "/mavros/rc/override",
                "allow_reverse": False,
                "override_enable_topic": "/seano/rc_override_enable",
                "override_enabled_default": False,
                "publish_release_when_disabled": True,
                "rc_steer_chan": 1,
                "rc_throttle_chan": 3,
                "pwm_neutral": 1500,
                "pwm_fwd_max": 1900,
                "pwm_steer_left": 1100,
                "pwm_steer_right": 1900,
                "pub_hz": 20.0,
                "command_timeout_s": 0.5,
            }
        ],
    )

    # ------------------------------------------------------------------
    # Takeover manager
    # ------------------------------------------------------------------
    takeover = Node(
        package="seano_vision",
        executable="auto_controller_stub_node",
        name="auto_controller_stub_node",
        output="screen",
        condition=IfCondition(use_takeover_manager),
        parameters=[
            {
                "command_topic": ca_command_topic,
                "failsafe_active_topic": "/ca/failsafe_active",
                "out_left_topic": "/seano/auto/left_cmd",
                "out_right_topic": "/seano/auto/right_cmd",
                "auto_enable_topic": "/seano/auto_enable",
                "rc_override_enable_topic": "/seano/rc_override_enable",
                "master_enable_topic": "/seano/auto_master_enable",
                "master_enable_on_start": ParameterValue(master_enable_on_start, value_type=bool),
                "cruise_speed": 0.30,
                "turn_cmd": 0.55,
                "diff_mix_gain": 0.70,
            }
        ],
    )

    # ------------------------------------------------------------------
    # Mission / mode manager
    # ------------------------------------------------------------------
    mode_mgr = Node(
        package="seano_vision",
        executable="mission_mode_manager_node",
        name="mission_mode_manager_node",
        output="screen",
        condition=IfCondition(use_mode_manager),
        parameters=[
            {
                "avoid_mode": avoid_mode,
                "mission_mode_default": mission_mode_default,
                "failsafe_mode": failsafe_mode,
                "switch_to_avoid_on_takeover": True,
                "restore_mode_on_release": True,
                "switch_to_failsafe_on_failsafe": True,
                "restore_after_failsafe_if_clear": True,
                "min_mode_switch_interval_s": 1.0,
            }
        ],
    )

    # ------------------------------------------------------------------
    # rosbag record
    # ------------------------------------------------------------------
    bag_dir = PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"])
    bag_path = PathJoinSubstitution([bag_dir, bag_name])

    topics = [
        ca_image_topic,
        ca_annotated_topic,
        ca_detections_topic,
        ca_risk_topic,
        ca_command_topic,
        "/ca/failsafe_active",
        "/ca/failsafe_reason",
        ca_mode_topic,
        ca_metrics_topic,
        ca_debug_image_topic,
        "/seano/auto_master_enable",
        "/seano/auto_enable",
        "/seano/rc_override_enable",
        "/mavros/state",
        "/mavros/rc/override",
        "/mavros/rc/in",
        "/ca/mode_manager_state",
        "/ca/mode_manager_event",
    ]

    cond_record = IfCondition(PythonExpression(["('", record, "'.lower() == 'true')"]))

    bag_record = ExecuteProcess(
        condition=cond_record,
        cmd=["ros2", "bag", "record", "-o", bag_path, *topics],
        output="screen",
    )

    return LaunchDescription(
        [
            # Common
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("bag_name", default_value="phase5_mission_avoid"),
            DeclareLaunchArgument("master_enable_on_start", default_value="false"),
            DeclareLaunchArgument("failsafe_stale_is_active", default_value="true"),
            # Main toggles
            DeclareLaunchArgument("use_ca_pipeline", default_value="true"),
            DeclareLaunchArgument("use_takeover_manager", default_value="true"),
            DeclareLaunchArgument("use_mode_manager", default_value="true"),
            # Runtime profile
            DeclareLaunchArgument(
                "ca_runtime_profile",
                default_value="synthetic_light",
                description="synthetic_light | synthetic_watchdog | full",
            ),
            # Camera
            DeclareLaunchArgument(
                "ca_camera_launch",
                default_value="phase2_camera_source_test.launch.py",
            ),
            DeclareLaunchArgument(
                "ca_image_topic",
                default_value="/seano/camera/image_raw_reliable",
            ),
            DeclareLaunchArgument(
                "ca_use_camera",
                default_value=_bool_by_profile(
                    ca_runtime_profile, ["synthetic_light", "synthetic_watchdog", "full"]
                ),
            ),
            DeclareLaunchArgument("ca_camera_profile", default_value="custom"),
            DeclareLaunchArgument("ca_camera_source", default_value="url"),
            DeclareLaunchArgument("ca_camera_backend", default_value="opencv"),
            DeclareLaunchArgument("ca_camera_url", default_value=default_video_demo),
            DeclareLaunchArgument("ca_camera_pipeline", default_value=""),
            DeclareLaunchArgument("ca_camera_device_path", default_value="/dev/video0"),
            DeclareLaunchArgument("ca_camera_device_index", default_value="0"),
            DeclareLaunchArgument("ca_camera_device_fourcc", default_value="MJPG"),
            DeclareLaunchArgument("ca_camera_device_width", default_value="640"),
            DeclareLaunchArgument("ca_camera_device_height", default_value="480"),
            DeclareLaunchArgument("ca_camera_device_fps", default_value="30"),
            DeclareLaunchArgument(
                "ca_camera_topic_best_effort",
                default_value="/seano/camera/image_raw",
            ),
            DeclareLaunchArgument(
                "ca_camera_topic_reliable",
                default_value="/seano/camera/image_raw_reliable",
            ),
            DeclareLaunchArgument("ca_camera_frame_id", default_value="camera_link"),
            DeclareLaunchArgument("ca_camera_max_fps", default_value="4.0"),
            DeclareLaunchArgument("ca_camera_max_age_ms", default_value="250"),
            DeclareLaunchArgument("ca_camera_record", default_value="false"),
            DeclareLaunchArgument("ca_camera_bag_base_dir", default_value=""),
            DeclareLaunchArgument("ca_camera_bag_prefix", default_value="phase5_camera"),
            DeclareLaunchArgument("ca_camera_duration_s", default_value="0"),
            # Detector / risk / watchdog
            DeclareLaunchArgument(
                "ca_use_detector",
                default_value=_bool_by_profile(
                    ca_runtime_profile, ["synthetic_light", "synthetic_watchdog", "full"]
                ),
            ),
            DeclareLaunchArgument(
                "ca_use_risk",
                default_value=_bool_by_profile(
                    ca_runtime_profile, ["synthetic_light", "synthetic_watchdog", "full"]
                ),
            ),
            DeclareLaunchArgument(
                "ca_use_watchdog",
                default_value=_bool_by_profile(ca_runtime_profile, ["synthetic_watchdog", "full"]),
            ),
            DeclareLaunchArgument("ca_use_ca_viewer", default_value="false"),
            # IMPORTANT: paksa profile videodemo sebagai default Phase 5
            DeclareLaunchArgument(
                "ca_risk_profile",
                default_value="alfin7_videodemo.yaml",
            ),
            DeclareLaunchArgument(
                "ca_annotated_topic",
                default_value="/camera/image_annotated",
            ),
            DeclareLaunchArgument(
                "ca_detections_topic",
                default_value="/camera/detections",
            ),
            DeclareLaunchArgument("ca_risk_topic", default_value="/ca/risk"),
            DeclareLaunchArgument("ca_command_topic", default_value="/ca/command_safe"),
            DeclareLaunchArgument("ca_mode_topic", default_value="/ca/mode"),
            DeclareLaunchArgument("ca_metrics_topic", default_value="/ca/metrics"),
            DeclareLaunchArgument("ca_debug_image_topic", default_value="/ca/debug_image"),
            DeclareLaunchArgument("ca_publish_debug_image", default_value="true"),
            DeclareLaunchArgument("ca_det_sub_reliability", default_value="reliable"),
            DeclareLaunchArgument("ca_det_pub_reliability", default_value="reliable"),
            DeclareLaunchArgument("ca_det_qos_depth", default_value="1"),
            DeclareLaunchArgument("ca_det_model_path", default_value="yolov8n.pt"),
            DeclareLaunchArgument("ca_det_device", default_value=""),
            DeclareLaunchArgument("ca_det_imgsz", default_value="416"),
            DeclareLaunchArgument("ca_det_conf", default_value="0.20"),
            DeclareLaunchArgument("ca_det_iou", default_value="0.45"),
            DeclareLaunchArgument("ca_det_class_ids", default_value="ALL"),
            DeclareLaunchArgument("ca_det_max_det", default_value="30"),
            DeclareLaunchArgument("ca_det_agnostic_nms", default_value="false"),
            DeclareLaunchArgument("ca_det_half", default_value="false"),
            DeclareLaunchArgument("ca_det_warmup", default_value="true"),
            DeclareLaunchArgument("ca_det_max_fps", default_value="2.0"),
            DeclareLaunchArgument("ca_det_publish_annotated", default_value="true"),
            DeclareLaunchArgument("ca_det_publish_detections", default_value="true"),
            DeclareLaunchArgument("ca_det_publish_empty_detections", default_value="true"),
            DeclareLaunchArgument(
                "wd_startup_grace_s",
                default_value=_str_by_profile(ca_runtime_profile, "3.0", "8.0"),
            ),
            DeclareLaunchArgument("wd_start_in_failsafe", default_value="false"),
            # Bridge / mode policy
            DeclareLaunchArgument("input_mode", default_value="left_right"),
            DeclareLaunchArgument("output_mode", default_value="rc_thr_steer"),
            DeclareLaunchArgument("avoid_mode", default_value="MANUAL"),
            DeclareLaunchArgument("mission_mode_default", default_value="AUTO"),
            DeclareLaunchArgument("failsafe_mode", default_value="MANUAL"),
            # Actions
            camera_include_plain,
            camera_include_passthrough,
            detector_node,
            risk_node,
            watchdog_node,
            ca_viewer,
            mux,
            limiter,
            bridge,
            takeover,
            mode_mgr,
            bag_record,
        ]
    )
