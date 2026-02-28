#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os

from ament_index_python.packages import get_package_share_directory
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


def _maybe_record(context, *args, **kwargs):
    record = context.perform_substitution(LaunchConfiguration("record")).strip().lower()
    if record not in ("1", "true", "yes", "y", "on"):
        return []

    record_images = (
        context.perform_substitution(LaunchConfiguration("record_images")).strip().lower()
    )
    record_images = record_images in ("1", "true", "yes", "y", "on")

    base_dir = context.perform_substitution(LaunchConfiguration("bag_base_dir")).strip()
    prefix = context.perform_substitution(LaunchConfiguration("bag_prefix")).strip()

    image_be = context.perform_substitution(LaunchConfiguration("image_best_effort_topic")).strip()
    image_rel = context.perform_substitution(LaunchConfiguration("image_reliable_topic")).strip()
    det_topic = context.perform_substitution(LaunchConfiguration("detections_topic")).strip()

    # Default hemat disk: record non-image topics.
    topics = []
    for t in [det_topic, "/ca/failsafe_active", "/ca/failsafe_reason", "/ca/watchdog_status"]:
        if t and t not in topics:
            topics.append(t)

    # Optional: record image topics (besar!)
    if record_images:
        for t in [image_be, image_rel]:
            if t and t not in topics:
                topics.append(t)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"{ts}_{prefix}")

    topic_args = " ".join([f'"{t}"' for t in topics])

    cmd = [
        "bash",
        "-lc",
        f'mkdir -p "{base_dir}" && '
        f'echo "Recording rosbag to: {out_dir}" && '
        f'echo "record_images={str(record_images).lower()} topics={topics}" && '
        f'ros2 bag record -o "{out_dir}" {topic_args}',
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
                EmitEvent(event=Shutdown(reason="phase2 cam+det+watchdog completed")),
            ],
        ),
    ]


def _make_watchdog(context, *args, **kwargs):
    image_be = context.perform_substitution(LaunchConfiguration("image_best_effort_topic")).strip()
    image_rel = context.perform_substitution(LaunchConfiguration("image_reliable_topic")).strip()

    topics = []
    for t in [image_rel, image_be, "/ca/debug_image"]:
        if t and t not in topics:
            topics.append(t)

    params = {
        "image_topics": topics,
        "sub_reliability": context.perform_substitution(
            LaunchConfiguration("watchdog_sub_reliability")
        ).strip(),
        "image_timeout_s": float(
            context.perform_substitution(LaunchConfiguration("image_timeout_s")).strip()
        ),
        "startup_grace_s": float(
            context.perform_substitution(LaunchConfiguration("startup_grace_s")).strip()
        ),
        "lost_if_risk_stale": False,
        "lost_if_mode_lost": False,
        "lost_if_mode_stale": False,
        "start_in_failsafe": False,
    }

    return [
        Node(
            package="seano_vision",
            executable="watchdog_failsafe_node",
            name="watchdog",
            output="screen",
            emulate_tty=True,
            parameters=[params],
        )
    ]


def generate_launch_description():
    default_pipeline = (
        "videotestsrc is-live=true pattern=smpte ! "
        "video/x-raw,framerate=30/1,width=640,height=480 ! "
        "videoconvert ! appsink drop=true max-buffers=1 sync=false"
    )
    default_bag_dir = PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"])

    pkg_share = get_package_share_directory("seano_vision")
    model_default = os.path.join(pkg_share, "models", "yolov8n.pt")

    camera_node = Node(
        package="seano_vision",
        executable="camera_node",
        name="camera_source",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "source": LaunchConfiguration("source"),
                "backend": LaunchConfiguration("backend"),
                "url": LaunchConfiguration("url"),
                "pipeline": LaunchConfiguration("pipeline"),
                "device_path": LaunchConfiguration("device_path"),
                "device_index": ParameterValue(LaunchConfiguration("device_index"), value_type=int),
                "device_fourcc": LaunchConfiguration("device_fourcc"),
                "device_width": ParameterValue(LaunchConfiguration("device_width"), value_type=int),
                "device_height": ParameterValue(
                    LaunchConfiguration("device_height"), value_type=int
                ),
                "device_fps": ParameterValue(LaunchConfiguration("device_fps"), value_type=int),
                "publish_best_effort": ParameterValue(
                    LaunchConfiguration("publish_best_effort"), value_type=bool
                ),
                "publish_reliable": ParameterValue(
                    LaunchConfiguration("publish_reliable"), value_type=bool
                ),
                "topic_best_effort": LaunchConfiguration("image_best_effort_topic"),
                "topic_reliable": LaunchConfiguration("image_reliable_topic"),
                "frame_id": LaunchConfiguration("frame_id"),
                "max_fps": ParameterValue(LaunchConfiguration("camera_max_fps"), value_type=float),
                "max_age_ms": ParameterValue(
                    LaunchConfiguration("camera_max_age_ms"), value_type=int
                ),
                "publish_in_reader": ParameterValue(
                    LaunchConfiguration("publish_in_reader"), value_type=bool
                ),
                "log_stats_sec": ParameterValue(
                    LaunchConfiguration("log_stats_sec"), value_type=float
                ),
            }
        ],
    )

    detector_node = Node(
        package="seano_vision",
        executable="detector_node",
        name="detector_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "sub_image": LaunchConfiguration("image_best_effort_topic"),
                "sub_reliability": "best_effort",
                "pub_det": LaunchConfiguration("detections_topic"),
                "pub_det_reliability": "best_effort",
                "publish_annotated": ParameterValue(
                    LaunchConfiguration("publish_annotated"), value_type=bool
                ),
                "publish_detections": True,
                "publish_empty_detections": True,
                "model_path": LaunchConfiguration("model_path"),
                "device": LaunchConfiguration("device"),
                "imgsz": ParameterValue(LaunchConfiguration("imgsz"), value_type=int),
                "conf": ParameterValue(LaunchConfiguration("conf"), value_type=float),
                "iou": ParameterValue(LaunchConfiguration("iou"), value_type=float),
                "class_ids": LaunchConfiguration("class_ids"),
                "max_det": ParameterValue(LaunchConfiguration("max_det"), value_type=int),
                "max_fps": ParameterValue(
                    LaunchConfiguration("detector_max_fps"), value_type=float
                ),
                "qos_depth": ParameterValue(LaunchConfiguration("qos_depth"), value_type=int),
                "warmup": ParameterValue(LaunchConfiguration("warmup"), value_type=bool),
                "stats_period": ParameterValue(
                    LaunchConfiguration("stats_period"), value_type=float
                ),
            }
        ],
    )

    delayed_detector = TimerAction(period=1.0, actions=[detector_node])
    delayed_watchdog = TimerAction(period=1.2, actions=[OpaqueFunction(function=_make_watchdog)])

    return LaunchDescription(
        [
            DeclareLaunchArgument("source", default_value="pipeline"),
            DeclareLaunchArgument("backend", default_value="gstreamer"),
            DeclareLaunchArgument("url", default_value=""),
            DeclareLaunchArgument("pipeline", default_value=default_pipeline),
            DeclareLaunchArgument("device_path", default_value="/dev/video0"),
            DeclareLaunchArgument("device_index", default_value="0"),
            DeclareLaunchArgument("device_fourcc", default_value="MJPG"),
            DeclareLaunchArgument("device_width", default_value="640"),
            DeclareLaunchArgument("device_height", default_value="480"),
            DeclareLaunchArgument("device_fps", default_value="30"),
            DeclareLaunchArgument("publish_best_effort", default_value="true"),
            DeclareLaunchArgument("publish_reliable", default_value="false"),
            DeclareLaunchArgument(
                "image_best_effort_topic", default_value="/seano/camera/image_raw"
            ),
            DeclareLaunchArgument(
                "image_reliable_topic", default_value="/seano/camera/image_raw_reliable"
            ),
            DeclareLaunchArgument("frame_id", default_value="camera"),
            DeclareLaunchArgument("camera_max_fps", default_value="15.0"),
            DeclareLaunchArgument("camera_max_age_ms", default_value="120"),
            DeclareLaunchArgument("publish_in_reader", default_value="true"),
            DeclareLaunchArgument("log_stats_sec", default_value="2.0"),
            DeclareLaunchArgument("detections_topic", default_value="/camera/detections"),
            DeclareLaunchArgument("model_path", default_value=model_default),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("imgsz", default_value="416"),
            DeclareLaunchArgument("conf", default_value="0.25"),
            DeclareLaunchArgument("iou", default_value="0.45"),
            DeclareLaunchArgument("class_ids", default_value="ALL"),
            DeclareLaunchArgument("max_det", default_value="50"),
            DeclareLaunchArgument("detector_max_fps", default_value="8.0"),
            DeclareLaunchArgument("qos_depth", default_value="1"),
            DeclareLaunchArgument("warmup", default_value="true"),
            DeclareLaunchArgument("stats_period", default_value="1.0"),
            DeclareLaunchArgument("publish_annotated", default_value="false"),
            DeclareLaunchArgument("watchdog_sub_reliability", default_value="best_effort"),
            DeclareLaunchArgument("image_timeout_s", default_value="2.0"),
            DeclareLaunchArgument("startup_grace_s", default_value="2.0"),
            # Record options
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("record_images", default_value="false"),  # DEFAULT HEMAT DISK
            DeclareLaunchArgument("bag_base_dir", default_value=default_bag_dir),
            DeclareLaunchArgument("bag_prefix", default_value="phase2_cam_det_watchdog"),
            DeclareLaunchArgument("duration_s", default_value="0"),
            camera_node,
            delayed_detector,
            delayed_watchdog,
            TimerAction(period=0.5, actions=[OpaqueFunction(function=_maybe_record)]),
            OpaqueFunction(function=_maybe_autostop),
        ]
    )
