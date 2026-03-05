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


def _is_true(expr: LaunchConfiguration) -> PythonExpression:
    # Returns PythonExpression that evaluates to True if LaunchConfiguration == "true" (case-insensitive)
    # NOTE: We must quote the substituted string, otherwise Python sees: false/true (NameError).
    return PythonExpression(["('", expr, "'.lower() == 'true')"])


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

    # ---------- Command MUX ----------
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

    # ---------- Safety limiter ----------
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

    # ---------- RC override bridge ----------
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
                "lr_to_steer_gain": 1.0,
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
                "cruise_speed": 0.30,
                "turn_cmd": 0.55,
                "diff_mix_gain": 0.7,
            }
        ],
    )

    # ---------- Rosbag record ----------
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

    # record == true AND record_images == false
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

    # record == true AND record_images == true
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
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("record_images", default_value="false"),
            DeclareLaunchArgument(
                "bag_dir",
                default_value=PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"]),
            ),
            DeclareLaunchArgument("bag_name", default_value="lake_auto_demo"),
            DeclareLaunchArgument("master_enable_on_start", default_value="false"),
            DeclareLaunchArgument("failsafe_stale_is_active", default_value="true"),
            DeclareLaunchArgument(
                "ca_camera_launch", default_value="phase2_camera_source_test.launch.py"
            ),
            DeclareLaunchArgument(
                "ca_image_topic", default_value="/seano/camera/image_raw_reliable"
            ),
            DeclareLaunchArgument("ca_det_sub_reliability", default_value="reliable"),
            DeclareLaunchArgument("ca_det_pub_reliability", default_value="reliable"),
            DeclareLaunchArgument("input_mode", default_value="left_right"),
            DeclareLaunchArgument("output_mode", default_value="rc_thr_steer"),
            ca_include,
            mux,
            limiter,
            bridge,
            takeover,
            record_non_image,
            record_with_images,
        ]
    )
