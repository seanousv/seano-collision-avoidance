# -*- coding: utf-8 -*-
# THESIS_ACTIVE_HW_BASELINE
#
# This launch is the active hardware baseline for the thesis.
# It should be read as the primary Jetson + CUAV + USB camera entry point for:
# - hardware bench integration,
# - dockside preparation,
# - controlled field-test preparation,
# - full runtime monitoring with MAVROS and browser-based HUD.
#
# Operational note:
# - keep filename stable while the field baseline is still being validated.
# - treat parameter changes carefully and record them in docs/BASELINE_PARAMETER_LOCK.md.
#

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

DEFAULT_USB_DEVICE_PATH = "/dev/v4l/by-id/usb-FEC_NYK_NEMESIS_202001010001-video-index0"


def _bool_by_profile(profile, enabled_profiles):
    csv_profiles = ",".join(enabled_profiles)
    return PythonExpression(
        [
            "'true' if '",
            profile,
            "' in '",
            csv_profiles,
            "'.split(',') else 'false'",
        ]
    )


def _str_by_profile(profile, non_full_value, full_value):
    return PythonExpression(
        [
            "'",
            full_value,
            "' if '",
            profile,
            "' == 'full' else '",
            non_full_value,
            "'",
        ]
    )


def _all_true(*subs):
    expr = ["'true' if ("]
    for i, sub in enumerate(subs):
        if i > 0:
            expr.append(" and ")
        expr.extend(["'", sub, "'.lower() == 'true'"])
    expr.append(") else 'false'")
    return PythonExpression(expr)


def _camera_launch_is(use_camera, camera_launch, filename):
    return PythonExpression(
        [
            "'true' if ('",
            use_camera,
            "'.lower() == 'true' and '",
            camera_launch,
            "' == '",
            filename,
            "') else 'false'",
        ]
    )


def _camera_launch_other(use_camera, camera_launch):
    return PythonExpression(
        [
            "'true' if ('",
            use_camera,
            "'.lower() == 'true' and '",
            camera_launch,
            "' not in ['phase2_camera_usb_test.launch.py', 'phase2_camera_source_test.launch.py']) else 'false'",
        ]
    )


def generate_launch_description():
    pkg_share = FindPackageShare("seano_vision")
    mavros_share = FindPackageShare("mavros")

    # ------------------------------------------------------------------
    # Core / orchestration
    # ------------------------------------------------------------------
    record = LaunchConfiguration("record")
    bag_name = LaunchConfiguration("bag_name")
    use_event_logger = LaunchConfiguration("use_event_logger")
    event_log_root = LaunchConfiguration("event_log_root")
    event_run_id = LaunchConfiguration("event_run_id")
    event_frame_max_age_s = LaunchConfiguration("event_frame_max_age_s")

    use_mavros = LaunchConfiguration("use_mavros")
    use_ca_pipeline = LaunchConfiguration("use_ca_pipeline")
    use_takeover_manager = LaunchConfiguration("use_takeover_manager")
    use_mode_manager = LaunchConfiguration("use_mode_manager")

    master_enable_on_start = LaunchConfiguration("master_enable_on_start")
    failsafe_stale_is_active = LaunchConfiguration("failsafe_stale_is_active")

    # ------------------------------------------------------------------
    # MAVROS hardware link
    # ------------------------------------------------------------------
    mavros_namespace = LaunchConfiguration("mavros_namespace")
    fcu_url = LaunchConfiguration("fcu_url")
    gcs_url = LaunchConfiguration("gcs_url")
    tgt_system = LaunchConfiguration("tgt_system")
    tgt_component = LaunchConfiguration("tgt_component")
    fcu_protocol = LaunchConfiguration("fcu_protocol")
    respawn_mavros = LaunchConfiguration("respawn_mavros")

    # ------------------------------------------------------------------
    # CA profile
    # ------------------------------------------------------------------
    ca_runtime_profile = LaunchConfiguration("ca_runtime_profile")
    ca_camera_launch = LaunchConfiguration("ca_camera_launch")

    ca_image_topic = LaunchConfiguration("ca_image_topic")
    ca_annotated_topic = LaunchConfiguration("ca_annotated_topic")
    ca_detections_raw_topic = LaunchConfiguration("ca_detections_raw_topic")
    ca_detections_filtered_topic = LaunchConfiguration("ca_detections_filtered_topic")
    ca_detections_fused_topic = LaunchConfiguration("ca_detections_fused_topic")
    ca_detections_for_risk_topic = LaunchConfiguration("ca_detections_for_risk_topic")
    ca_risk_topic = LaunchConfiguration("ca_risk_topic")
    ca_command_topic = LaunchConfiguration("ca_command_topic")
    ca_mode_topic = LaunchConfiguration("ca_mode_topic")
    ca_metrics_topic = LaunchConfiguration("ca_metrics_topic")
    ca_debug_image_topic = LaunchConfiguration("ca_debug_image_topic")

    ca_use_camera = LaunchConfiguration("ca_use_camera")
    ca_use_detector = LaunchConfiguration("ca_use_detector")
    ca_use_waterline = LaunchConfiguration("ca_use_waterline")
    ca_use_fp_guard = LaunchConfiguration("ca_use_fp_guard")
    ca_use_fusion = LaunchConfiguration("ca_use_fusion")
    ca_use_vq = LaunchConfiguration("ca_use_vq")
    ca_use_freeze = LaunchConfiguration("ca_use_freeze")
    ca_use_risk = LaunchConfiguration("ca_use_risk")
    ca_use_watchdog = LaunchConfiguration("ca_use_watchdog")
    ca_use_ca_viewer = LaunchConfiguration("ca_use_ca_viewer")
    ca_use_wl_viewer = LaunchConfiguration("ca_use_wl_viewer")

    ca_det_sub_reliability = LaunchConfiguration("ca_det_sub_reliability")
    ca_det_pub_reliability = LaunchConfiguration("ca_det_pub_reliability")
    ca_det_qos_depth = LaunchConfiguration("ca_det_qos_depth")

    # ------------------------------------------------------------------
    # Direct camera management from phase7
    # ------------------------------------------------------------------
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
    ca_camera_reconnect_sec = LaunchConfiguration("ca_camera_reconnect_sec")
    ca_camera_log_stats_sec = LaunchConfiguration("ca_camera_log_stats_sec")
    ca_camera_record = LaunchConfiguration("ca_camera_record")
    ca_camera_bag_base_dir = LaunchConfiguration("ca_camera_bag_base_dir")
    ca_camera_bag_prefix = LaunchConfiguration("ca_camera_bag_prefix")
    ca_camera_duration_s = LaunchConfiguration("ca_camera_duration_s")

    # ------------------------------------------------------------------
    # Detector runtime/config passthrough
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Watchdog tuning
    # ------------------------------------------------------------------
    wd_startup_grace_s = LaunchConfiguration("wd_startup_grace_s")
    wd_start_in_failsafe = LaunchConfiguration("wd_start_in_failsafe")

    # ------------------------------------------------------------------
    # Mux / limiter
    # ------------------------------------------------------------------
    mux_command_timeout_s = LaunchConfiguration("mux_command_timeout_s")
    limiter_input_timeout_s = LaunchConfiguration("limiter_input_timeout_s")
    limiter_failsafe_timeout_s = LaunchConfiguration("limiter_failsafe_timeout_s")
    limiter_loop_hz = LaunchConfiguration("limiter_loop_hz")

    # ------------------------------------------------------------------
    # Bridge / hardware output
    # ------------------------------------------------------------------
    input_mode = LaunchConfiguration("input_mode")
    output_mode = LaunchConfiguration("output_mode")

    rc_left_chan = LaunchConfiguration("rc_left_chan")
    rc_right_chan = LaunchConfiguration("rc_right_chan")
    rc_steer_chan = LaunchConfiguration("rc_steer_chan")
    rc_throttle_chan = LaunchConfiguration("rc_throttle_chan")

    pwm_neutral = LaunchConfiguration("pwm_neutral")
    pwm_fwd_max = LaunchConfiguration("pwm_fwd_max")
    pwm_rev_min = LaunchConfiguration("pwm_rev_min")
    pwm_steer_left = LaunchConfiguration("pwm_steer_left")
    pwm_steer_right = LaunchConfiguration("pwm_steer_right")
    pwm_output_min = LaunchConfiguration("pwm_output_min")
    pwm_output_max = LaunchConfiguration("pwm_output_max")

    allow_reverse = LaunchConfiguration("allow_reverse")
    bridge_pub_hz = LaunchConfiguration("bridge_pub_hz")
    bridge_command_timeout_s = LaunchConfiguration("bridge_command_timeout_s")
    bridge_pwm_slew_rate_us_per_s = LaunchConfiguration("bridge_pwm_slew_rate_us_per_s")

    # ------------------------------------------------------------------
    # Takeover / mode
    # ------------------------------------------------------------------
    avoid_mode = LaunchConfiguration("avoid_mode")
    mission_mode_default = LaunchConfiguration("mission_mode_default")
    failsafe_mode = LaunchConfiguration("failsafe_mode")

    cruise_speed = LaunchConfiguration("cruise_speed")
    slow_factor = LaunchConfiguration("slow_factor")
    turn_speed_factor = LaunchConfiguration("turn_speed_factor")
    turn_cmd = LaunchConfiguration("turn_cmd")
    diff_mix_gain = LaunchConfiguration("diff_mix_gain")
    speed_max = LaunchConfiguration("speed_max")

    # ------------------------------------------------------------------
    # Includes
    # ------------------------------------------------------------------
    mavros_include = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(PathJoinSubstitution([mavros_share, "launch", "apm.launch"])),
        condition=IfCondition(use_mavros),
        launch_arguments={
            "fcu_url": fcu_url,
            "gcs_url": gcs_url,
            "tgt_system": tgt_system,
            "tgt_component": tgt_component,
            "fcu_protocol": fcu_protocol,
            "respawn_mavros": respawn_mavros,
            "namespace": mavros_namespace,
        }.items(),
    )

    # Direct camera include: USB launch with tunable hardware args
    camera_usb_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "phase2_camera_usb_test.launch.py"])
        ),
        condition=IfCondition(
            _camera_launch_is(ca_use_camera, ca_camera_launch, "phase2_camera_usb_test.launch.py")
        ),
        launch_arguments={
            "device_index": ca_camera_device_index,
            "device_path": ca_camera_device_path,
            "device_fourcc": ca_camera_device_fourcc,
            "device_width": ca_camera_device_width,
            "device_height": ca_camera_device_height,
            "device_fps": ca_camera_device_fps,
            "max_fps": ca_camera_max_fps,
            "max_age_ms": ca_camera_max_age_ms,
            "reconnect_sec": ca_camera_reconnect_sec,
            "log_stats_sec": ca_camera_log_stats_sec,
            "topic_best_effort": ca_camera_topic_best_effort,
            "topic_reliable": ca_camera_topic_reliable,
        }.items(),
    )

    # Direct camera include: generic source test launch
    camera_source_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "phase2_camera_source_test.launch.py"])
        ),
        condition=IfCondition(
            _camera_launch_is(
                ca_use_camera, ca_camera_launch, "phase2_camera_source_test.launch.py"
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

    # Fallback for custom camera launch names
    camera_other_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", ca_camera_launch])
        ),
        condition=IfCondition(_camera_launch_other(ca_use_camera, ca_camera_launch)),
    )

    # Keep detector/risk/watchdog/viewers in demo_full_ca, but disable its camera include
    ca_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "demo_full_ca.launch.py"])
        ),
        condition=IfCondition(use_ca_pipeline),
        launch_arguments={
            "use_camera": "false",
            "use_detector": ca_use_detector,
            "use_waterline": ca_use_waterline,
            "use_fp_guard": ca_use_fp_guard,
            "use_fusion": ca_use_fusion,
            "use_vq": ca_use_vq,
            "use_freeze": ca_use_freeze,
            "use_risk": ca_use_risk,
            "use_watchdog": ca_use_watchdog,
            "use_ca_viewer": ca_use_ca_viewer,
            "use_wl_viewer": ca_use_wl_viewer,
            "camera_launch": ca_camera_launch,
            "image_topic": "/ca/debug_image",
            "annotated_topic": ca_annotated_topic,
            "detections_raw_topic": ca_detections_raw_topic,
            "detections_filtered_topic": ca_detections_filtered_topic,
            "detections_fused_topic": ca_detections_fused_topic,
            "detections_for_risk_topic": ca_detections_for_risk_topic,
            "risk_topic": ca_risk_topic,
            "command_topic": ca_command_topic,
            "mode_topic": ca_mode_topic,
            "metrics_topic": ca_metrics_topic,
            "debug_image_topic": ca_debug_image_topic,
            "det_sub_reliability": ca_det_sub_reliability,
            "det_pub_reliability": ca_det_pub_reliability,
            "det_qos_depth": ca_det_qos_depth,
            "det_model_path": ca_det_model_path,
            "det_device": ca_det_device,
            "det_imgsz": ca_det_imgsz,
            "det_conf": ca_det_conf,
            "det_iou": ca_det_iou,
            "det_class_ids": ca_det_class_ids,
            "det_max_det": ca_det_max_det,
            "det_agnostic_nms": ca_det_agnostic_nms,
            "det_half": ca_det_half,
            "det_warmup": ca_det_warmup,
            "det_max_fps": ca_det_max_fps,
            "det_publish_annotated": ca_det_publish_annotated,
            "det_publish_detections": ca_det_publish_detections,
            "det_publish_empty_detections": ca_det_publish_empty_detections,
            "wd_startup_grace_s": wd_startup_grace_s,
            "wd_start_in_failsafe": wd_start_in_failsafe,
        }.items(),
    )

    # ------------------------------------------------------------------
    # mux -> limiter -> bridge
    # ------------------------------------------------------------------
    mux = Node(
        package="seano_vision",
        executable="command_mux_node",
        name="command_mux_node",
        output="screen",
        condition=IfCondition(use_ca_pipeline),
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
                "command_timeout_s": ParameterValue(mux_command_timeout_s, value_type=float),
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
            }
        ],
    )

    limiter = Node(
        package="seano_vision",
        executable="actuator_safety_limiter_node",
        name="actuator_safety_limiter_node",
        output="screen",
        condition=IfCondition(use_ca_pipeline),
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
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
                "input_timeout_s": ParameterValue(limiter_input_timeout_s, value_type=float),
                "failsafe_timeout_s": ParameterValue(limiter_failsafe_timeout_s, value_type=float),
                "loop_hz": ParameterValue(limiter_loop_hz, value_type=float),
                "reason_topic": "/seano/limiter_reason",
            }
        ],
    )

    bridge = Node(
        package="seano_vision",
        executable="mavros_rc_override_bridge_node",
        name="mavros_rc_override_bridge_node",
        output="screen",
        condition=IfCondition(use_ca_pipeline),
        parameters=[
            {
                "input_mode": input_mode,
                "output_mode": output_mode,
                "left_topic": "/seano/left_cmd",
                "right_topic": "/seano/right_cmd",
                "out_topic": "/mavros/rc/override",
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
                "override_enable_topic": "/seano/rc_override_enable",
                "override_enabled_default": False,
                "publish_release_when_disabled": True,
                "rc_left_chan": ParameterValue(rc_left_chan, value_type=int),
                "rc_right_chan": ParameterValue(rc_right_chan, value_type=int),
                "rc_steer_chan": ParameterValue(rc_steer_chan, value_type=int),
                "rc_throttle_chan": ParameterValue(rc_throttle_chan, value_type=int),
                "pwm_neutral": ParameterValue(pwm_neutral, value_type=int),
                "pwm_fwd_max": ParameterValue(pwm_fwd_max, value_type=int),
                "pwm_rev_min": ParameterValue(pwm_rev_min, value_type=int),
                "pwm_steer_left": ParameterValue(pwm_steer_left, value_type=int),
                "pwm_steer_right": ParameterValue(pwm_steer_right, value_type=int),
                "pwm_output_min": ParameterValue(pwm_output_min, value_type=int),
                "pwm_output_max": ParameterValue(pwm_output_max, value_type=int),
                "pwm_slew_rate_us_per_s": ParameterValue(
                    bridge_pwm_slew_rate_us_per_s, value_type=float
                ),
                "pub_hz": ParameterValue(bridge_pub_hz, value_type=float),
                "command_timeout_s": ParameterValue(bridge_command_timeout_s, value_type=float),
            }
        ],
    )

    # ------------------------------------------------------------------
    # takeover manager
    # ------------------------------------------------------------------
    takeover = Node(
        package="seano_vision",
        executable="auto_controller_stub_node",
        name="auto_controller_stub_node",
        output="screen",
        condition=IfCondition(_all_true(use_ca_pipeline, use_takeover_manager)),
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
                "cruise_speed": ParameterValue(cruise_speed, value_type=float),
                "slow_factor": ParameterValue(slow_factor, value_type=float),
                "turn_speed_factor": ParameterValue(turn_speed_factor, value_type=float),
                "turn_cmd": ParameterValue(turn_cmd, value_type=float),
                "diff_mix_gain": ParameterValue(diff_mix_gain, value_type=float),
                "speed_max": ParameterValue(speed_max, value_type=float),
            }
        ],
    )

    # ------------------------------------------------------------------
    # mission / mode manager
    # ------------------------------------------------------------------
    mode_mgr = Node(
        package="seano_vision",
        executable="mission_mode_manager_node",
        name="mission_mode_manager_node",
        output="screen",
        condition=IfCondition(_all_true(use_ca_pipeline, use_mode_manager)),
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
    # event logger
    # ------------------------------------------------------------------
    event_logger = Node(
        package="seano_vision",
        executable="event_logger_node",
        name="event_logger_node",
        output="screen",
        condition=IfCondition(use_event_logger),
        parameters=[
            {
                "log_root": event_log_root,
                "run_id": event_run_id,
                "image_topic": "/ca/debug_image",
                "avoid_state_topic": "/ca/mode_manager_state",
                "ca_mode_topic": "/ca/mode",
                "mode_event_topic": "/ca/mode_manager_event",
                "command_safe_topic": ca_command_topic,
                "command_raw_topic": "/ca/command",
                "risk_topic": "/ca/risk",
                "failsafe_active_topic": "/ca/failsafe_active",
                "auto_master_enable_topic": "/seano/auto_master_enable",
                "auto_enable_topic": "/seano/auto_enable",
                "rc_override_enable_topic": "/seano/rc_override_enable",
                "manual_left_cmd_topic": "/seano/manual/left_cmd",
                "manual_right_cmd_topic": "/seano/manual/right_cmd",
                "auto_left_cmd_topic": "/seano/auto/left_cmd",
                "auto_right_cmd_topic": "/seano/auto/right_cmd",
                "selected_left_cmd_topic": "/seano/selected/left_cmd",
                "selected_right_cmd_topic": "/seano/selected/right_cmd",
                "left_cmd_topic": "/seano/left_cmd",
                "right_cmd_topic": "/seano/right_cmd",
                "limiter_reason_topic": "/seano/limiter_reason",
                "rc_override_topic": "/mavros/rc/override",
                "frame_max_age_s": ParameterValue(event_frame_max_age_s, value_type=float),
                "save_frames": True,
                "float_epsilon": 0.02,
                "min_event_interval_s": 0.05,
            }
        ],
    )

    # ------------------------------------------------------------------
    # rosbag record
    # ------------------------------------------------------------------
    bag_dir = PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"])
    bag_path = PathJoinSubstitution([bag_dir, bag_name])

    bag_topics = [
        ca_image_topic,
        ca_annotated_topic,
        ca_detections_raw_topic,
        ca_detections_filtered_topic,
        ca_detections_fused_topic,
        ca_risk_topic,
        ca_command_topic,
        "/ca/command_safe",
        "/ca/failsafe_active",
        "/ca/failsafe_reason",
        ca_mode_topic,
        ca_metrics_topic,
        ca_debug_image_topic,
        "/ca/watchdog_status",
        "/vision/freeze",
        "/vision/freeze_reason",
        "/seano/auto_master_enable",
        "/seano/auto_enable",
        "/seano/rc_override_enable",
        "/seano/manual/left_cmd",
        "/seano/manual/right_cmd",
        "/seano/auto/left_cmd",
        "/seano/auto/right_cmd",
        "/seano/selected/left_cmd",
        "/seano/selected/right_cmd",
        "/seano/left_cmd",
        "/seano/right_cmd",
        "/seano/limiter_reason",
        "/mavros/state",
        "/mavros/rc/override",
        "/mavros/rc/in",
        "/mavros/global_position/raw/fix",
        "/mavros/global_position/compass_hdg",
        "/mavros/local_position/pose",
        "/mavros/imu/data",
        "/ca/mode_manager_state",
        "/ca/mode_manager_event",
    ]

    bag_record = ExecuteProcess(
        condition=IfCondition(record),
        cmd=["ros2", "bag", "record", "-o", bag_path, *bag_topics],
        output="screen",
    )

    # ------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------
    args = [
        # Core
        DeclareLaunchArgument("record", default_value="false"),
        DeclareLaunchArgument("bag_name", default_value="phase7_cuav_usb_e2e"),
        DeclareLaunchArgument("use_event_logger", default_value="true"),
        DeclareLaunchArgument("event_log_root", default_value="~/seano_event_logs"),
        DeclareLaunchArgument("event_run_id", default_value=""),
        DeclareLaunchArgument("event_frame_max_age_s", default_value="1.0"),
        DeclareLaunchArgument("master_enable_on_start", default_value="false"),
        DeclareLaunchArgument("failsafe_stale_is_active", default_value="true"),
        DeclareLaunchArgument("use_mavros", default_value="true"),
        DeclareLaunchArgument("use_ca_pipeline", default_value="true"),
        DeclareLaunchArgument("use_takeover_manager", default_value="true"),
        DeclareLaunchArgument("use_mode_manager", default_value="true"),
        # MAVROS
        DeclareLaunchArgument("mavros_namespace", default_value="mavros"),
        DeclareLaunchArgument(
            "fcu_url",
            default_value="/dev/ttyACM0:115200",
            description="Contoh: /dev/ttyACM0:115200 atau /dev/ttyUSB0:115200",
        ),
        DeclareLaunchArgument("gcs_url", default_value=""),
        DeclareLaunchArgument("tgt_system", default_value="1"),
        DeclareLaunchArgument("tgt_component", default_value="1"),
        DeclareLaunchArgument("fcu_protocol", default_value="v2.0"),
        DeclareLaunchArgument("respawn_mavros", default_value="false"),
        # Runtime profile
        DeclareLaunchArgument(
            "ca_runtime_profile",
            default_value="usb_watchdog",
            description="usb_light | usb_watchdog | full",
        ),
        DeclareLaunchArgument(
            "ca_camera_launch",
            default_value="phase2_camera_usb_test.launch.py",
        ),
        # Core CA topics
        DeclareLaunchArgument(
            "ca_image_topic",
            default_value="/seano/camera/image_raw_reliable",
        ),
        DeclareLaunchArgument(
            "ca_annotated_topic",
            default_value="/camera/image_annotated",
        ),
        DeclareLaunchArgument(
            "ca_detections_raw_topic",
            default_value="/camera/detections",
        ),
        DeclareLaunchArgument(
            "ca_detections_filtered_topic",
            default_value="/camera/detections_filtered",
        ),
        DeclareLaunchArgument(
            "ca_detections_fused_topic",
            default_value="/camera/detections_fused",
        ),
        DeclareLaunchArgument(
            "ca_detections_for_risk_topic",
            default_value="",
        ),
        DeclareLaunchArgument("ca_risk_topic", default_value="/ca/risk"),
        DeclareLaunchArgument(
            "ca_command_topic",
            default_value="/ca/command_safe",
            description="Default hardware mengikuti watchdog-conditioned command stream (/ca/command_safe). Override ke /ca/command hanya untuk debugging atau eksperimen terkontrol.",
        ),
        DeclareLaunchArgument("ca_mode_topic", default_value="/ca/mode"),
        DeclareLaunchArgument("ca_metrics_topic", default_value="/ca/metrics"),
        DeclareLaunchArgument("ca_debug_image_topic", default_value="/ca/debug_image"),
        # Profile-based toggles
        DeclareLaunchArgument(
            "ca_use_camera",
            default_value=_bool_by_profile(
                ca_runtime_profile, ["usb_light", "usb_watchdog", "full"]
            ),
        ),
        DeclareLaunchArgument(
            "ca_use_detector",
            default_value=_bool_by_profile(
                ca_runtime_profile, ["usb_light", "usb_watchdog", "full"]
            ),
        ),
        DeclareLaunchArgument(
            "ca_use_waterline",
            default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
        ),
        DeclareLaunchArgument(
            "ca_use_fp_guard",
            default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
        ),
        DeclareLaunchArgument(
            "ca_use_fusion",
            default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
        ),
        DeclareLaunchArgument(
            "ca_use_vq",
            default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
        ),
        DeclareLaunchArgument(
            "ca_use_freeze",
            default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
        ),
        DeclareLaunchArgument(
            "ca_use_risk",
            default_value=_bool_by_profile(
                ca_runtime_profile, ["usb_light", "usb_watchdog", "full"]
            ),
        ),
        DeclareLaunchArgument(
            "ca_use_watchdog",
            default_value=_bool_by_profile(ca_runtime_profile, ["usb_watchdog", "full"]),
        ),
        DeclareLaunchArgument("ca_use_ca_viewer", default_value="false"),
        DeclareLaunchArgument("ca_use_wl_viewer", default_value="false"),
        # Detector QoS
        DeclareLaunchArgument("ca_det_sub_reliability", default_value="reliable"),
        DeclareLaunchArgument("ca_det_pub_reliability", default_value="reliable"),
        DeclareLaunchArgument("ca_det_qos_depth", default_value="10"),
        # Camera passthrough / tuning
        DeclareLaunchArgument("ca_camera_profile", default_value=""),
        DeclareLaunchArgument("ca_camera_source", default_value=""),
        DeclareLaunchArgument("ca_camera_backend", default_value=""),
        DeclareLaunchArgument("ca_camera_url", default_value=""),
        DeclareLaunchArgument("ca_camera_pipeline", default_value=""),
        DeclareLaunchArgument("ca_camera_device_path", default_value=DEFAULT_USB_DEVICE_PATH),
        DeclareLaunchArgument("ca_camera_device_index", default_value="0"),
        DeclareLaunchArgument("ca_camera_device_fourcc", default_value="MJPG"),
        DeclareLaunchArgument("ca_camera_device_width", default_value="640"),
        DeclareLaunchArgument("ca_camera_device_height", default_value="480"),
        DeclareLaunchArgument("ca_camera_device_fps", default_value="30"),
        DeclareLaunchArgument(
            "ca_camera_topic_best_effort", default_value="/seano/camera/image_raw"
        ),
        DeclareLaunchArgument(
            "ca_camera_topic_reliable", default_value="/seano/camera/image_raw_reliable"
        ),
        DeclareLaunchArgument("ca_camera_frame_id", default_value=""),
        DeclareLaunchArgument("ca_camera_max_fps", default_value="15.0"),
        DeclareLaunchArgument("ca_camera_max_age_ms", default_value="120"),
        DeclareLaunchArgument("ca_camera_reconnect_sec", default_value="0.5"),
        DeclareLaunchArgument("ca_camera_log_stats_sec", default_value="2.0"),
        DeclareLaunchArgument("ca_camera_record", default_value="false"),
        DeclareLaunchArgument("ca_camera_bag_base_dir", default_value=""),
        DeclareLaunchArgument("ca_camera_bag_prefix", default_value="phase2_camera"),
        DeclareLaunchArgument("ca_camera_duration_s", default_value="0"),
        # Detector runtime/config passthrough
        DeclareLaunchArgument("ca_det_model_path", default_value="yolov8n.pt"),
        DeclareLaunchArgument(
            "ca_det_device",
            default_value="",
            description="Empty string = auto device selection by Ultralytics",
        ),
        DeclareLaunchArgument("ca_det_imgsz", default_value="416"),
        DeclareLaunchArgument("ca_det_conf", default_value="0.20"),
        DeclareLaunchArgument("ca_det_iou", default_value="0.45"),
        DeclareLaunchArgument("ca_det_class_ids", default_value="ALL"),
        DeclareLaunchArgument("ca_det_max_det", default_value="50"),
        DeclareLaunchArgument("ca_det_agnostic_nms", default_value="false"),
        DeclareLaunchArgument("ca_det_half", default_value="false"),
        DeclareLaunchArgument("ca_det_warmup", default_value="true"),
        DeclareLaunchArgument("ca_det_max_fps", default_value="8.0"),
        DeclareLaunchArgument("ca_det_publish_annotated", default_value="true"),
        DeclareLaunchArgument("ca_det_publish_detections", default_value="true"),
        DeclareLaunchArgument("ca_det_publish_empty_detections", default_value="true"),
        # Watchdog
        DeclareLaunchArgument(
            "wd_startup_grace_s",
            default_value=_str_by_profile(ca_runtime_profile, "4.0", "8.0"),
        ),
        DeclareLaunchArgument("wd_start_in_failsafe", default_value="false"),
        # mux / limiter
        DeclareLaunchArgument("mux_command_timeout_s", default_value="0.6"),
        DeclareLaunchArgument("limiter_input_timeout_s", default_value="0.6"),
        DeclareLaunchArgument("limiter_failsafe_timeout_s", default_value="2.0"),
        DeclareLaunchArgument("limiter_loop_hz", default_value="20.0"),
        # bridge / output
        DeclareLaunchArgument("input_mode", default_value="left_right"),
        DeclareLaunchArgument("output_mode", default_value="rc_left_right"),
        DeclareLaunchArgument("rc_left_chan", default_value="1"),
        DeclareLaunchArgument("rc_right_chan", default_value="3"),
        DeclareLaunchArgument("rc_steer_chan", default_value="1"),
        DeclareLaunchArgument("rc_throttle_chan", default_value="3"),
        DeclareLaunchArgument("pwm_neutral", default_value="1500"),
        DeclareLaunchArgument("pwm_fwd_max", default_value="1900"),
        DeclareLaunchArgument("pwm_rev_min", default_value="1100"),
        DeclareLaunchArgument("pwm_steer_left", default_value="1100"),
        DeclareLaunchArgument("pwm_steer_right", default_value="1900"),
        DeclareLaunchArgument("pwm_output_min", default_value="1000"),
        DeclareLaunchArgument("pwm_output_max", default_value="2000"),
        DeclareLaunchArgument("allow_reverse", default_value="false"),
        DeclareLaunchArgument("bridge_pub_hz", default_value="20.0"),
        DeclareLaunchArgument("bridge_command_timeout_s", default_value="0.5"),
        DeclareLaunchArgument("bridge_pwm_slew_rate_us_per_s", default_value="250.0"),
        # mode policy
        DeclareLaunchArgument("avoid_mode", default_value="MANUAL"),
        DeclareLaunchArgument("mission_mode_default", default_value="AUTO"),
        DeclareLaunchArgument("failsafe_mode", default_value="MANUAL"),
        # takeover tuning
        DeclareLaunchArgument("cruise_speed", default_value="0.30"),
        DeclareLaunchArgument("slow_factor", default_value="0.55"),
        DeclareLaunchArgument("turn_speed_factor", default_value="0.75"),
        DeclareLaunchArgument("turn_cmd", default_value="0.50"),
        DeclareLaunchArgument("diff_mix_gain", default_value="0.65"),
        DeclareLaunchArgument("speed_max", default_value="0.55"),
    ]

    actions = [
        *args,
        mavros_include,
        camera_usb_include,
        camera_source_include,
        camera_other_include,
        ca_include,
        mux,
        limiter,
        bridge,
        takeover,
        mode_mgr,
        event_logger,
        bag_record,
    ]

    return LaunchDescription(actions)
