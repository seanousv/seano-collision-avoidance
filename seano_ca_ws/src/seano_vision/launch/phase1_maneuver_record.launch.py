#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare

TOPICS_PHASE1 = [
    "/mavros/state",
    "/mavros/rc/override",
    "/mavros/rc/in",
    "/seano/auto_enable",
    "/seano/manual/left_cmd",
    "/seano/manual/right_cmd",
    "/seano/selected/left_cmd",
    "/seano/selected/right_cmd",
    "/seano/left_cmd",
    "/seano/right_cmd",
]


def _make_record_action(context, *args, **kwargs):
    record_str = context.perform_substitution(LaunchConfiguration("record")).strip().lower()
    if record_str not in ("true", "1", "yes", "y", "on"):
        return []

    base_dir = context.perform_substitution(LaunchConfiguration("bag_base_dir")).strip()
    prefix = context.perform_substitution(LaunchConfiguration("bag_prefix")).strip()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"{ts}_{prefix}")

    topic_args = " ".join(TOPICS_PHASE1)

    cmd = [
        "bash",
        "-lc",
        f'mkdir -p "{base_dir}" && '
        f'echo "Recording rosbag to: {out_dir}" && '
        f'ros2 bag record -o "{out_dir}" {topic_args}',
    ]
    return [ExecuteProcess(cmd=cmd, output="screen")]


def generate_launch_description():
    # Default test launch
    default_test_launch = PathJoinSubstitution(
        [FindPackageShare("seano_vision"), "launch", "phase1_maneuver_test.launch.py"]
    )

    # Default bag dir: ~/bags
    default_bag_dir = PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"])

    # Forward args to the included test launch (ini kunci 1-click)
    include_test = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(LaunchConfiguration("test_launch")),
        launch_arguments={
            "max_cycles": LaunchConfiguration("max_cycles"),
            "lr_to_steer_gain": LaunchConfiguration("lr_to_steer_gain"),
            "base_throttle": LaunchConfiguration("base_throttle"),
            "turn_delta": LaunchConfiguration("turn_delta"),
        }.items(),
    )

    # Start recording slightly after test start (test punya warmup 1s)
    delayed_record = TimerAction(
        period=0.3,
        actions=[OpaqueFunction(function=_make_record_action)],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "test_launch",
                default_value=default_test_launch,
                description="Path to phase1 maneuver test launch file.",
            ),
            DeclareLaunchArgument(
                "record",
                default_value="true",
                description="Enable rosbag recording (true/false).",
            ),
            DeclareLaunchArgument(
                "bag_base_dir",
                default_value=default_bag_dir,
                description="Base output directory for rosbags (default: ~/bags).",
            ),
            DeclareLaunchArgument(
                "bag_prefix",
                default_value="phase1_maneuver",
                description="Prefix for bag folder name.",
            ),
            # Standard test defaults (pengujian terkontrol)
            DeclareLaunchArgument("max_cycles", default_value="5"),
            DeclareLaunchArgument("lr_to_steer_gain", default_value="0.6"),
            DeclareLaunchArgument("base_throttle", default_value="0.45"),
            DeclareLaunchArgument("turn_delta", default_value="0.06"),
            include_test,
            delayed_record,
        ]
    )
