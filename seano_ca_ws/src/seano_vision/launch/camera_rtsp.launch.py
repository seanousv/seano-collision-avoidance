#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("seano_vision")
    default_cfg = os.path.join(pkg_share, "config", "camera_hp_rtsp.yaml")

    return LaunchDescription([
        # config base (tetap pakai YAML kamu)
        DeclareLaunchArgument(
            "cfg",
            default_value=default_cfg,
            description="Base YAML config for camera node",
        ),

        # yang sering berubah kita jadiin launch-arg
        DeclareLaunchArgument(
            "url",
            default_value="rtsp://192.168.1.3:8080/h264.sdp",
            description="RTSP URL",
        ),
        DeclareLaunchArgument("swap_rb", default_value="true", description="Swap R/B channels"),
        DeclareLaunchArgument("gstreamer_latency_ms", default_value="120", description="GStreamer latency (ms)"),
        DeclareLaunchArgument("rtsp_tcp", default_value="true", description="RTSP over TCP"),
        DeclareLaunchArgument("max_fps", default_value="15.0", description="Publish FPS limit"),
        DeclareLaunchArgument("max_age_ms", default_value="120", description="Drop frames older than this"),
        DeclareLaunchArgument("publish_best_effort", default_value="true"),
        DeclareLaunchArgument("publish_reliable", default_value="true"),
        DeclareLaunchArgument("node_name", default_value="camera_hp"),

        Node(
            package="seano_vision",
            executable="camera_node",
            name=LaunchConfiguration("node_name"),
            output="screen",
            emulate_tty=True,
            parameters=[
                LaunchConfiguration("cfg"),
                {
                    "url": LaunchConfiguration("url"),
                    "swap_rb": ParameterValue(LaunchConfiguration("swap_rb"), value_type=bool),
                    "gstreamer_latency_ms": ParameterValue(LaunchConfiguration("gstreamer_latency_ms"), value_type=int),
                    "rtsp_tcp": ParameterValue(LaunchConfiguration("rtsp_tcp"), value_type=bool),
                    "max_fps": ParameterValue(LaunchConfiguration("max_fps"), value_type=float),
                    "max_age_ms": ParameterValue(LaunchConfiguration("max_age_ms"), value_type=int),
                    "publish_best_effort": ParameterValue(LaunchConfiguration("publish_best_effort"), value_type=bool),
                    "publish_reliable": ParameterValue(LaunchConfiguration("publish_reliable"), value_type=bool),
                },
            ],
        ),
    ])
