#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory("seano_vision")
    cfg = os.path.join(pkg_share, "config", "camera_usb.yaml")

    # Default stabil untuk kamera USB yang sudah terdeteksi di Jetson:
    # /dev/v4l/by-id/usb-FEC_NYK_NEMESIS_202001010001-video-index0 -> ../../video0
    #
    # Jika suatu saat symlink by-id berubah, override saat launch:
    #   device_path:=/dev/v4l/by-id/<nama-baru>-video-index0
    default_device_path = (
        "/dev/v4l/by-id/usb-FEC_NYK_NEMESIS_202001010001-video-index0"
    )

    device_index = LaunchConfiguration("device_index")
    device_path = LaunchConfiguration("device_path")
    device_fourcc = LaunchConfiguration("device_fourcc")
    device_width = LaunchConfiguration("device_width")
    device_height = LaunchConfiguration("device_height")
    device_fps = LaunchConfiguration("device_fps")

    max_fps = LaunchConfiguration("max_fps")
    max_age_ms = LaunchConfiguration("max_age_ms")
    reconnect_sec = LaunchConfiguration("reconnect_sec")
    log_stats_sec = LaunchConfiguration("log_stats_sec")

    topic_best_effort = LaunchConfiguration("topic_best_effort")
    topic_reliable = LaunchConfiguration("topic_reliable")

    overrides = {
        # Source
        "source": "device",
        "backend": "opencv",
        # Prioritas sekarang: buka kamera via device_path yang stabil (by-id),
        # bukan lagi mengandalkan index 0.
        # device_index tetap disediakan hanya sebagai fallback manual.
        "device_index": ParameterValue(device_index, value_type=int),
        "device_path": device_path,
        # Format device
        "device_fourcc": device_fourcc,
        "device_width": ParameterValue(device_width, value_type=int),
        "device_height": ParameterValue(device_height, value_type=int),
        "device_fps": ParameterValue(device_fps, value_type=int),
        # Publish throttling
        "max_fps": ParameterValue(max_fps, value_type=float),
        "max_age_ms": ParameterValue(max_age_ms, value_type=int),
        "grab_skip": 0,
        # Stabilitas runtime
        "publish_in_reader": False,
        "output_encoding": "bgr8",
        "swap_rb": False,
        "reconnect_sec": ParameterValue(reconnect_sec, value_type=float),
        "log_stats_sec": ParameterValue(log_stats_sec, value_type=float),
        # Publish dua jalur supaya tetap kompatibel dengan baseline hardware phase7
        "publish_best_effort": True,
        "publish_reliable": True,
        "topic_best_effort": topic_best_effort,
        "topic_reliable": topic_reliable,
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device_index",
                default_value="0",
                description=(
                    "Fallback index kamera. Normalnya diabaikan karena device_path "
                    "sudah diisi dengan symlink by-id yang stabil."
                ),
            ),
            DeclareLaunchArgument(
                "device_path",
                default_value=default_device_path,
                description=(
                    "Path kamera stabil berbasis /dev/v4l/by-id. "
                    "Kosongkan hanya jika ingin pakai device_index secara manual."
                ),
            ),
            DeclareLaunchArgument("device_fourcc", default_value="MJPG"),
            DeclareLaunchArgument("device_width", default_value="640"),
            DeclareLaunchArgument("device_height", default_value="480"),
            DeclareLaunchArgument("device_fps", default_value="30"),
            DeclareLaunchArgument("max_fps", default_value="15.0"),
            DeclareLaunchArgument("max_age_ms", default_value="120"),
            DeclareLaunchArgument("reconnect_sec", default_value="0.5"),
            DeclareLaunchArgument("log_stats_sec", default_value="2.0"),
            DeclareLaunchArgument(
                "topic_best_effort",
                default_value="/seano/camera/image_raw",
            ),
            DeclareLaunchArgument(
                "topic_reliable",
                default_value="/seano/camera/image_raw_reliable",
            ),
            Node(
                package="seano_vision",
                executable="camera_node",
                name="camera_hp",  # jangan diubah dulu; harus match key YAML aktif
                output="screen",
                emulate_tty=True,
                parameters=[cfg, overrides],
            ),
        ]
    )