#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ---- args ----
    manual_left = LaunchConfiguration("manual_left_topic")
    manual_right = LaunchConfiguration("manual_right_topic")
    auto_left = LaunchConfiguration("auto_left_topic")
    auto_right = LaunchConfiguration("auto_right_topic")
    out_left = LaunchConfiguration("out_left_topic")
    out_right = LaunchConfiguration("out_right_topic")
    auto_enable_topic = LaunchConfiguration("auto_enable_topic")

    mavros_rc_override = LaunchConfiguration("mavros_rc_override_topic")
    rc_left_chan = LaunchConfiguration("rc_left_chan")
    rc_right_chan = LaunchConfiguration("rc_right_chan")

    # ---- nodes ----
    # 0) Test maneuver publishes MANUAL left/right and forces auto_enable False
    test_node = Node(
        package="seano_vision",
        executable="test_maneuver_node",
        name="test_maneuver_node",
        output="screen",
        parameters=[
            {
                "manual_left_topic": manual_left,
                "manual_right_topic": manual_right,
                "auto_enable_topic": auto_enable_topic,
                # keep publish rate high to avoid mux timeout
                "pub_hz": 20.0,
                "enable_pub_hz": 2.0,
                # default maneuver values (bisa di-tune nanti via ros2 param / launch args kalau mau)
                "repeat": True,
            }
        ],
    )

    # 1) Mux: MANUAL/AUTO -> OUT
    mux = Node(
        package="seano_vision",
        executable="command_mux_node",
        name="command_mux_node",
        output="screen",
        parameters=[
            {
                "auto_left_topic": auto_left,
                "auto_right_topic": auto_right,
                "manual_left_topic": manual_left,
                "manual_right_topic": manual_right,
                "out_left_topic": out_left,
                "out_right_topic": out_right,
                "auto_enable_topic": auto_enable_topic,
                "rate_hz": 20.0,
                "command_timeout_s": 0.6,
                "fallback_to_manual": True,
                "allow_reverse": False,
                "output_min": 0.0,
                "output_max": 1.0,
            }
        ],
    )

    # 2) Bridge: OUT left/right -> MAVROS rc/override
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

    # Trik kecil supaya mux/bridge start setelah test node mulai publish (mengurangi FAILSAFE awal)
    delayed_mux = TimerAction(period=0.6, actions=[mux])
    delayed_bridge = TimerAction(period=0.8, actions=[bridge])

    return LaunchDescription(
        [
            DeclareLaunchArgument("manual_left_topic", default_value="/seano/manual/left_cmd"),
            DeclareLaunchArgument("manual_right_topic", default_value="/seano/manual/right_cmd"),
            DeclareLaunchArgument("auto_left_topic", default_value="/seano/auto/left_cmd"),
            DeclareLaunchArgument("auto_right_topic", default_value="/seano/auto/right_cmd"),
            DeclareLaunchArgument("out_left_topic", default_value="/seano/left_cmd"),
            DeclareLaunchArgument("out_right_topic", default_value="/seano/right_cmd"),
            DeclareLaunchArgument("auto_enable_topic", default_value="/seano/auto_enable"),
            DeclareLaunchArgument("mavros_rc_override_topic", default_value="/mavros/rc/override"),
            DeclareLaunchArgument("rc_left_chan", default_value="1"),
            DeclareLaunchArgument("rc_right_chan", default_value="3"),
            test_node,
            delayed_mux,
            delayed_bridge,
        ]
    )
