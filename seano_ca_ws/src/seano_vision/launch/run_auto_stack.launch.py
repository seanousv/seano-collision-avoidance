#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ---- args ----
    auto_enable_on_start = LaunchConfiguration("auto_enable_on_start")

    # Topics (final)
    auto_left = LaunchConfiguration("auto_left_topic")
    auto_right = LaunchConfiguration("auto_right_topic")
    man_left = LaunchConfiguration("manual_left_topic")
    man_right = LaunchConfiguration("manual_right_topic")
    out_left = LaunchConfiguration("out_left_topic")
    out_right = LaunchConfiguration("out_right_topic")

    # MAVROS override topic
    mavros_rc_override = LaunchConfiguration("mavros_rc_override_topic")

    # Bridge RC channels
    rc_left_chan = LaunchConfiguration("rc_left_chan")
    rc_right_chan = LaunchConfiguration("rc_right_chan")

    # Limiter params
    publish_left_right = LaunchConfiguration("publish_left_right")
    publish_thr_steer = LaunchConfiguration("publish_thr_steer")
    diff_mix_gain = LaunchConfiguration("diff_mix_gain")
    allow_reverse = LaunchConfiguration("allow_reverse")

    # ---- Nodes ----
    # 1) Limiter: /ca/command -> /seano/auto/left_cmd & /seano/auto/right_cmd
    limiter = Node(
        package="seano_vision",
        executable="actuator_safety_limiter_node",
        name="actuator_safety_limiter_node",
        output="screen",
        parameters=[
            {
                "out_left_topic": auto_left,
                "out_right_topic": auto_right,
                "publish_left_right": ParameterValue(publish_left_right, value_type=bool),
                "publish_thr_steer": ParameterValue(publish_thr_steer, value_type=bool),
                "diff_mix_gain": ParameterValue(diff_mix_gain, value_type=float),
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
                "auto_enable_on_start": ParameterValue(auto_enable_on_start, value_type=bool),
            }
        ],
    )

    # 2) Teleop: /seano/manual/left_cmd & /seano/manual/right_cmd
    teleop = Node(
        package="seano_vision",
        executable="teleop_diff_thruster_node",
        name="teleop_diff_thruster_node",
        output="screen",
        parameters=[
            {
                "left_topic": man_left,
                "right_topic": man_right,
            }
        ],
    )

    # 3) Mux: choose manual vs auto -> /seano/left_cmd & /seano/right_cmd
    mux = Node(
        package="seano_vision",
        executable="command_mux_node",
        name="command_mux_node",
        output="screen",
        parameters=[
            {
                "auto_left_topic": auto_left,
                "auto_right_topic": auto_right,
                "manual_left_topic": man_left,
                "manual_right_topic": man_right,
                "out_left_topic": out_left,
                "out_right_topic": out_right,
            }
        ],
    )

    # 4) Bridge: /seano/left_cmd & /seano/right_cmd -> /mavros/rc/override
    bridge = Node(
        package="seano_vision",
        executable="mavros_rc_override_bridge_node",
        name="mavros_rc_override_bridge_node",
        output="screen",
        parameters=[
            {
                "input_mode": "left_right",
                "output_mode": "rc_left_right",
                "left_topic": out_left,
                "right_topic": out_right,
                "out_topic": mavros_rc_override,
                "rc_left_chan": ParameterValue(rc_left_chan, value_type=int),
                "rc_right_chan": ParameterValue(rc_right_chan, value_type=int),
                "test_enable": False,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("auto_enable_on_start", default_value="false"),
            DeclareLaunchArgument("auto_left_topic", default_value="/seano/auto/left_cmd"),
            DeclareLaunchArgument("auto_right_topic", default_value="/seano/auto/right_cmd"),
            DeclareLaunchArgument("manual_left_topic", default_value="/seano/manual/left_cmd"),
            DeclareLaunchArgument("manual_right_topic", default_value="/seano/manual/right_cmd"),
            DeclareLaunchArgument("out_left_topic", default_value="/seano/left_cmd"),
            DeclareLaunchArgument("out_right_topic", default_value="/seano/right_cmd"),
            DeclareLaunchArgument("mavros_rc_override_topic", default_value="/mavros/rc/override"),
            DeclareLaunchArgument("rc_left_chan", default_value="1"),
            DeclareLaunchArgument("rc_right_chan", default_value="3"),
            DeclareLaunchArgument("publish_left_right", default_value="true"),
            DeclareLaunchArgument("publish_thr_steer", default_value="false"),
            DeclareLaunchArgument("diff_mix_gain", default_value="0.7"),
            DeclareLaunchArgument("allow_reverse", default_value="false"),
            limiter,
            teleop,
            mux,
            bridge,
        ]
    )
