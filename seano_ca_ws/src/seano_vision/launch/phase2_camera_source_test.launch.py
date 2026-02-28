#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

TOPICS_PHASE2_MIN = [
    "/seano/camera/image_raw",
]


def _maybe_record(context, *args, **kwargs):
    record = context.perform_substitution(LaunchConfiguration("record")).strip().lower()
    if record not in ("1", "true", "yes", "y", "on"):
        return []

    base_dir = context.perform_substitution(LaunchConfiguration("bag_base_dir")).strip()
    prefix = context.perform_substitution(LaunchConfiguration("bag_prefix")).strip()
    topic = context.perform_substitution(LaunchConfiguration("topic_best_effort")).strip()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"{ts}_{prefix}")

    cmd = [
        "bash",
        "-lc",
        f'mkdir -p "{base_dir}" && '
        f'echo "Recording rosbag to: {out_dir}" && '
        f'ros2 bag record -o "{out_dir}" "{topic}"',
    ]
    return [ExecuteProcess(cmd=cmd, output="screen")]


def _maybe_autostop(context, *args, **kwargs):
    dur_s_str = context.perform_substitution(LaunchConfiguration("duration_s")).strip()
    try:
        dur_s = float(dur_s_str)
    except Exception:
        dur_s = 0.0

    if dur_s <= 0.0:
        return []

    return [
        LogInfo(msg=f"[phase2] auto shutdown in {dur_s:.1f}s"),
        TimerAction(
            period=dur_s,
            actions=[
                LogInfo(msg="[phase2] duration reached -> shutting down launch"),
                EmitEvent(event=Shutdown(reason="phase2 camera test completed")),
            ],
        ),
    ]


def generate_launch_description():
    # Default pipeline that works even when /dev/video0 doesn't exist (WSL-safe)
    default_pipeline = (
        "videotestsrc is-live=true pattern=smpte ! "
        "video/x-raw,framerate=30/1,width=640,height=480 ! "
        "videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )

    default_bag_dir = PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"])

    camera_node = Node(
        package="seano_vision",
        executable="camera_node",
        name="camera_source",
        output="screen",
        parameters=[
            {
                "source": LaunchConfiguration("source"),  # pipeline | url | device
                "backend": LaunchConfiguration("backend"),  # gstreamer | opencv
                "url": LaunchConfiguration("url"),  # rtsp://... (if source=url)
                "pipeline": LaunchConfiguration(
                    "pipeline"
                ),  # gstreamer pipeline (if source=pipeline)
                "device_path": LaunchConfiguration("device_path"),  # /dev/video0 (if source=device)
                "device_index": ParameterValue(LaunchConfiguration("device_index"), value_type=int),
                "device_fourcc": LaunchConfiguration("device_fourcc"),
                "device_width": ParameterValue(LaunchConfiguration("device_width"), value_type=int),
                "device_height": ParameterValue(
                    LaunchConfiguration("device_height"), value_type=int
                ),
                "device_fps": ParameterValue(LaunchConfiguration("device_fps"), value_type=int),
                "topic_best_effort": LaunchConfiguration("topic_best_effort"),
                "topic_reliable": LaunchConfiguration("topic_reliable"),
                "frame_id": LaunchConfiguration("frame_id"),
                "max_fps": ParameterValue(LaunchConfiguration("max_fps"), value_type=float),
                "max_age_ms": ParameterValue(LaunchConfiguration("max_age_ms"), value_type=int),
            }
        ],
    )

    return LaunchDescription(
        [
            # source selection
            DeclareLaunchArgument(
                "source", default_value="pipeline", description="pipeline | url | device"
            ),
            DeclareLaunchArgument(
                "backend", default_value="gstreamer", description="gstreamer | opencv"
            ),
            DeclareLaunchArgument(
                "url", default_value="", description="RTSP/HTTP URL (if source=url)"
            ),
            DeclareLaunchArgument(
                "pipeline", default_value=default_pipeline, description="GStreamer appsink pipeline"
            ),
            DeclareLaunchArgument(
                "device_path", default_value="/dev/video0", description="V4L2 device path"
            ),
            DeclareLaunchArgument(
                "device_index", default_value="0", description="Device index (legacy)"
            ),
            # device tuning (if source=device)
            DeclareLaunchArgument("device_fourcc", default_value="MJPG"),
            DeclareLaunchArgument("device_width", default_value="1280"),
            DeclareLaunchArgument("device_height", default_value="720"),
            DeclareLaunchArgument("device_fps", default_value="30"),
            # output topics
            DeclareLaunchArgument("topic_best_effort", default_value="/seano/camera/image_raw"),
            DeclareLaunchArgument(
                "topic_reliable", default_value="/seano/camera/image_raw_reliable"
            ),
            DeclareLaunchArgument("frame_id", default_value="camera"),
            # publish behavior
            DeclareLaunchArgument("max_fps", default_value="15.0"),
            DeclareLaunchArgument("max_age_ms", default_value="120"),
            # record (optional)
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("bag_base_dir", default_value=default_bag_dir),
            DeclareLaunchArgument("bag_prefix", default_value="phase2_camera"),
            # auto stop (optional)
            DeclareLaunchArgument(
                "duration_s",
                default_value="0",
                description="Auto stop after N seconds (0=disabled)",
            ),
            camera_node,
            # start record slightly after node start
            TimerAction(period=0.5, actions=[OpaqueFunction(function=_maybe_record)]),
            OpaqueFunction(function=_maybe_autostop),
        ]
    )
