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


def generate_launch_description():
    # ---------- Launch args ----------
    record = LaunchConfiguration("record")
    record_images = LaunchConfiguration("record_images")
    bag_dir = LaunchConfiguration("bag_dir")
    bag_name = LaunchConfiguration("bag_name")

    master_enable_on_start = LaunchConfiguration("master_enable_on_start")
    failsafe_stale_is_active = LaunchConfiguration("failsafe_stale_is_active")

    # CA pipeline config
    ca_camera_launch = LaunchConfiguration("ca_camera_launch")
    ca_image_topic = LaunchConfiguration("ca_image_topic")
    ca_det_sub_reliability = LaunchConfiguration("ca_det_sub_reliability")
    ca_det_pub_reliability = LaunchConfiguration("ca_det_pub_reliability")

    # Bridge config
    input_mode = LaunchConfiguration("input_mode")
    output_mode = LaunchConfiguration("output_mode")

    # RC mapping (hardware)
    rc_left_chan = LaunchConfiguration("rc_left_chan")
    rc_right_chan = LaunchConfiguration("rc_right_chan")

    # (optional) rc_thr_steer mapping if you switch output_mode:=rc_thr_steer
    rc_steer_chan = LaunchConfiguration("rc_steer_chan")
    rc_throttle_chan = LaunchConfiguration("rc_throttle_chan")

    # Tuning
    cruise_speed = LaunchConfiguration("cruise_speed")
    turn_cmd = LaunchConfiguration("turn_cmd")
    diff_mix_gain = LaunchConfiguration("diff_mix_gain")

    pkg_share = FindPackageShare("seano_vision")

    # ---------- Include full CA pipeline ----------
    ca_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "demo_full_ca.launch.py"])
        ),
        launch_arguments={
            "camera_launch": ca_camera_launch,
            "image_topic": ca_image_topic,
            "det_sub_reliability": ca_det_sub_reliability,
            "det_pub_reliability": ca_det_pub_reliability,
            "use_ca_viewer": "false",
            "use_wl_viewer": "false",
        }.items(),
    )

    # ---------- Command MUX (manual/auto -> selected) ----------
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

    # ---------- Safety limiter (selected -> final) ----------
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

    # ---------- RC override bridge (final -> /mavros/rc/override) ----------
    # Default hardware: output_mode=rc_left_right + rc_left_chan/rc_right_chan.
    bridge = Node(
        package="seano_vision",
        executable="mavros_rc_override_bridge_node",
        name="mavros_rc_override_bridge_node",
        output="screen",
        parameters=[
            {
                "input_mode": input_mode,  # left_right
                "output_mode": output_mode,  # rc_left_right (default)
                "left_topic": "/seano/left_cmd",
                "right_topic": "/seano/right_cmd",
                "out_topic": "/mavros/rc/override",
                "allow_reverse": False,
                # enable/release override
                "override_enable_topic": "/seano/rc_override_enable",
                "override_enabled_default": False,
                "publish_release_when_disabled": True,
                # PWM
                "pwm_neutral": 1500,
                "pwm_fwd_max": 1900,
                "pwm_rev_min": 1100,
                "pwm_output_min": 1000,
                "pwm_output_max": 2000,
                # rc_left_right mapping
                "rc_left_chan": ParameterValue(rc_left_chan, value_type=int),
                "rc_right_chan": ParameterValue(rc_right_chan, value_type=int),
                # rc_thr_steer mapping (if you switch output_mode:=rc_thr_steer)
                "rc_steer_chan": ParameterValue(rc_steer_chan, value_type=int),
                "rc_throttle_chan": ParameterValue(rc_throttle_chan, value_type=int),
                # rates
                "pub_hz": 20.0,
                "command_timeout_s": 0.5,
            }
        ],
    )

    # ---------- Takeover manager ----------
    takeover = Node(
        package="seano_vision",
        executable="auto_controller_stub_node",
        name="auto_controller_stub_node",
        output="screen",
        parameters=[
            {
                "command_topic": "/ca/command_safe",
                "failsafe_active_topic": "/ca/failsafe_active",
                "out_left_topic": "/seano/auto/left_cmd",
                "out_right_topic": "/seano/auto/right_cmd",
                "auto_enable_topic": "/seano/auto_enable",
                "rc_override_enable_topic": "/seano/rc_override_enable",
                "master_enable_topic": "/seano/auto_master_enable",
                "master_enable_on_start": ParameterValue(master_enable_on_start, value_type=bool),
                "cruise_speed": ParameterValue(cruise_speed, value_type=float),
                "turn_cmd": ParameterValue(turn_cmd, value_type=float),
                "diff_mix_gain": ParameterValue(diff_mix_gain, value_type=float),
            }
        ],
    )

    # ---------- Rosbag record (opsional, hemat disk default) ----------
    topics_base = [
        "/ca/command",
        "/ca/command_safe",
        "/ca/failsafe_active",
        "/ca/failsafe_reason",
        "/ca/mode",
        "/ca/risk",
        "/ca/watchdog_status",
        "/vision/freeze",
        "/vision/freeze_reason",
        "/vision/quality",
        "/seano/auto_enable",
        "/seano/auto_master_enable",
        "/seano/rc_override_enable",
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
        "/mavros/vfr_hud",
        "/mavros/local_position/pose",
        "/mavros/global_position/global",
    ]

    topics_with_images = topics_base + [
        "/seano/camera/image_raw_reliable",
        "/seano/camera/image_raw",
        "/camera/image_raw_reliable",
        "/camera/image_raw",
        "/camera/image_annotated",
    ]

    bag_path = PathJoinSubstitution([bag_dir, bag_name])

    cond_record_non_image = IfCondition(
        PythonExpression(
            [
                "('",
                record,
                "'.lower() == 'true') and ('",
                record_images,
                "'.lower() != 'true')",
            ]
        )
    )

    cond_record_with_images = IfCondition(
        PythonExpression(
            [
                "('",
                record,
                "'.lower() == 'true') and ('",
                record_images,
                "'.lower() == 'true')",
            ]
        )
    )

    record_non_image = ExecuteProcess(
        condition=cond_record_non_image,
        cmd=["ros2", "bag", "record", "-o", bag_path, *topics_base],
        output="screen",
    )

    record_with_images = ExecuteProcess(
        condition=cond_record_with_images,
        cmd=["ros2", "bag", "record", "-o", bag_path, *topics_with_images],
        output="screen",
    )

    return LaunchDescription(
        [
            # recording
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("record_images", default_value="false"),
            DeclareLaunchArgument(
                "bag_dir",
                default_value=PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"]),
            ),
            DeclareLaunchArgument("bag_name", default_value="lake_auto_hw"),
            # safety
            DeclareLaunchArgument("master_enable_on_start", default_value="false"),
            DeclareLaunchArgument("failsafe_stale_is_active", default_value="true"),
            # CA settings (default reliable)
            DeclareLaunchArgument(
                "ca_camera_launch", default_value="phase2_camera_source_test.launch.py"
            ),
            DeclareLaunchArgument(
                "ca_image_topic", default_value="/seano/camera/image_raw_reliable"
            ),
            DeclareLaunchArgument("ca_det_sub_reliability", default_value="reliable"),
            DeclareLaunchArgument("ca_det_pub_reliability", default_value="reliable"),
            # bridge defaults for hardware: direct left/right PWM override
            DeclareLaunchArgument("input_mode", default_value="left_right"),
            DeclareLaunchArgument("output_mode", default_value="rc_left_right"),
            # IMPORTANT: set these sesuai mapping RC channel di ArduPilot (Mission Planner)
            # Default saya set CH1 & CH3 karena sering dipakai, tapi WAJIB Anda pastikan.
            DeclareLaunchArgument("rc_left_chan", default_value="1"),
            DeclareLaunchArgument("rc_right_chan", default_value="3"),
            # If you choose output_mode:=rc_thr_steer
            DeclareLaunchArgument("rc_steer_chan", default_value="1"),
            DeclareLaunchArgument("rc_throttle_chan", default_value="3"),
            # tuning (aman untuk danau awal)
            DeclareLaunchArgument("cruise_speed", default_value="0.30"),
            DeclareLaunchArgument("turn_cmd", default_value="0.55"),
            DeclareLaunchArgument("diff_mix_gain", default_value="0.70"),
            ca_include,
            mux,
            limiter,
            bridge,
            takeover,
            record_non_image,
            record_with_images,
        ]
    )
