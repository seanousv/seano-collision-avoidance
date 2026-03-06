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


def _profile_defaults(profile: str) -> dict:
    profile = str(profile or "").strip().lower()

    dummy_light_pipeline = (
        "videotestsrc is-live=true pattern=smpte ! "
        "video/x-raw,framerate=10/1,width=320,height=240 ! "
        "videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )

    dummy_full_pipeline = (
        "videotestsrc is-live=true pattern=smpte ! "
        "video/x-raw,framerate=30/1,width=640,height=480 ! "
        "videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )

    if profile == "dummy_full":
        return {
            "source": "pipeline",
            "backend": "gstreamer",
            "url": "",
            "pipeline": dummy_full_pipeline,
            "device_path": "/dev/video0",
            "device_index": "0",
            "device_fourcc": "MJPG",
            "device_width": "640",
            "device_height": "480",
            "device_fps": "30",
            "frame_id": "camera_dummy",
            "max_fps": "15.0",
            "max_age_ms": "120",
        }

    if profile == "usb":
        return {
            "source": "device",
            "backend": "opencv",
            "url": "",
            "pipeline": "",
            "device_path": "/dev/video0",
            "device_index": "0",
            "device_fourcc": "MJPG",
            "device_width": "1280",
            "device_height": "720",
            "device_fps": "30",
            "frame_id": "camera_usb",
            "max_fps": "15.0",
            "max_age_ms": "120",
        }

    if profile == "rtsp":
        return {
            "source": "url",
            "backend": "gstreamer",
            "url": "rtsp://127.0.0.1:8554/stream",
            "pipeline": "",
            "device_path": "/dev/video0",
            "device_index": "0",
            "device_fourcc": "MJPG",
            "device_width": "1280",
            "device_height": "720",
            "device_fps": "30",
            "frame_id": "camera_rtsp",
            "max_fps": "15.0",
            "max_age_ms": "120",
        }

    if profile == "custom":
        return {
            "source": "",
            "backend": "",
            "url": "",
            "pipeline": "",
            "device_path": "/dev/video0",
            "device_index": "0",
            "device_fourcc": "MJPG",
            "device_width": "1280",
            "device_height": "720",
            "device_fps": "30",
            "frame_id": "camera",
            "max_fps": "15.0",
            "max_age_ms": "120",
        }

    # default = dummy_light
    return {
        "source": "pipeline",
        "backend": "gstreamer",
        "url": "",
        "pipeline": dummy_light_pipeline,
        "device_path": "/dev/video0",
        "device_index": "0",
        "device_fourcc": "MJPG",
        "device_width": "320",
        "device_height": "240",
        "device_fps": "10",
        "frame_id": "camera_dummy",
        "max_fps": "10.0",
        "max_age_ms": "200",
    }


def _pick(context, key: str, defaults: dict) -> str:
    value = context.perform_substitution(LaunchConfiguration(key)).strip()
    if value != "":
        return value
    return str(defaults[key])


def _to_int(value: str, fallback: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(fallback)


def _to_float(value: str, fallback: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float(fallback)


def _build_camera_node(context, *args, **kwargs):
    profile = context.perform_substitution(LaunchConfiguration("profile")).strip().lower()
    defaults = _profile_defaults(profile)

    source = _pick(context, "source", defaults)
    backend = _pick(context, "backend", defaults)
    url = _pick(context, "url", defaults)
    pipeline = _pick(context, "pipeline", defaults)
    device_path = _pick(context, "device_path", defaults)
    device_index = _to_int(_pick(context, "device_index", defaults), 0)
    device_fourcc = _pick(context, "device_fourcc", defaults)
    device_width = _to_int(_pick(context, "device_width", defaults), 640)
    device_height = _to_int(_pick(context, "device_height", defaults), 480)
    device_fps = _to_int(_pick(context, "device_fps", defaults), 30)
    frame_id = _pick(context, "frame_id", defaults)
    max_fps = _to_float(_pick(context, "max_fps", defaults), 15.0)
    max_age_ms = _to_int(_pick(context, "max_age_ms", defaults), 120)

    topic_best_effort = context.perform_substitution(
        LaunchConfiguration("topic_best_effort")
    ).strip()
    topic_reliable = context.perform_substitution(LaunchConfiguration("topic_reliable")).strip()

    node = Node(
        package="seano_vision",
        executable="camera_node",
        name="camera_source",
        output="screen",
        parameters=[
            {
                "source": source,  # pipeline | url | device
                "backend": backend,  # gstreamer | opencv
                "url": url,
                "pipeline": pipeline,
                "device_path": device_path,
                "device_index": device_index,
                "device_fourcc": device_fourcc,
                "device_width": device_width,
                "device_height": device_height,
                "device_fps": device_fps,
                "topic_best_effort": topic_best_effort,
                "topic_reliable": topic_reliable,
                "frame_id": frame_id,
                "max_fps": max_fps,
                "max_age_ms": max_age_ms,
            }
        ],
    )

    return [
        LogInfo(
            msg=(
                "[phase2_camera_source_test] profile="
                + profile
                + " source="
                + source
                + " backend="
                + backend
                + " topic="
                + topic_reliable
            )
        ),
        node,
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
    default_bag_dir = PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                default_value="dummy_light",
                description="dummy_light | dummy_full | usb | rtsp | custom",
            ),
            DeclareLaunchArgument(
                "source", default_value="", description="pipeline | url | device"
            ),
            DeclareLaunchArgument("backend", default_value="", description="gstreamer | opencv"),
            DeclareLaunchArgument(
                "url", default_value="", description="RTSP/HTTP URL (if source=url)"
            ),
            DeclareLaunchArgument(
                "pipeline", default_value="", description="GStreamer appsink pipeline"
            ),
            DeclareLaunchArgument("device_path", default_value="", description="V4L2 device path"),
            DeclareLaunchArgument(
                "device_index", default_value="", description="Device index (legacy)"
            ),
            DeclareLaunchArgument("device_fourcc", default_value=""),
            DeclareLaunchArgument("device_width", default_value=""),
            DeclareLaunchArgument("device_height", default_value=""),
            DeclareLaunchArgument("device_fps", default_value=""),
            DeclareLaunchArgument("topic_best_effort", default_value="/seano/camera/image_raw"),
            DeclareLaunchArgument(
                "topic_reliable",
                default_value="/seano/camera/image_raw_reliable",
            ),
            DeclareLaunchArgument("frame_id", default_value=""),
            DeclareLaunchArgument("max_fps", default_value=""),
            DeclareLaunchArgument("max_age_ms", default_value=""),
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("bag_base_dir", default_value=default_bag_dir),
            DeclareLaunchArgument("bag_prefix", default_value="phase2_camera"),
            DeclareLaunchArgument(
                "duration_s",
                default_value="0",
                description="Auto stop after N seconds (0=disabled)",
            ),
            OpaqueFunction(function=_build_camera_node),
            TimerAction(period=0.5, actions=[OpaqueFunction(function=_maybe_record)]),
            OpaqueFunction(function=_maybe_autostop),
        ]
    )
