#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("seano_vision")
    cfg = os.path.join(pkg_share, "config", "camera_usb.yaml")

    # OVERRIDE supaya hasilnya PASTI:
    # - hanya publish 1 topic (best-effort) => /camera/image_raw harus bisa ~15 Hz
    # - publish_in_reader False => shutdown aman, tidak publish saat context sudah mati
    # - kunci beban rendah (MJPG 640x480)
    overrides = {
        "source": "device",
        "backend": "opencv",
        "device_index": 0,
        "device_path": "/dev/video0",
        "device_fourcc": "MJPG",
        "device_width": 640,
        "device_height": 480,
        "device_fps": 30,
        # Target publish
        "max_fps": 15.0,
        "max_age_ms": 120,
        "grab_skip": 0,
        # Hindari error shutdown + kurangi beban
        "publish_in_reader": False,
        "output_encoding": "bgr8",
        "swap_rb": False,
        # KUNCI: hanya satu output topic dulu
        "publish_best_effort": True,
        "publish_reliable": False,
        "topic_best_effort": "/camera/image_raw",
        "topic_reliable": "/camera/image_raw_reliable",
        "reconnect_sec": 0.5,
        "log_stats_sec": 2.0,
    }

    return LaunchDescription(
        [
            Node(
                package="seano_vision",
                executable="camera_node",
                name="camera_hp",  # harus match key YAML (camera_hp:)
                output="screen",
                emulate_tty=True,
                parameters=[cfg, overrides],
            )
        ]
    )
