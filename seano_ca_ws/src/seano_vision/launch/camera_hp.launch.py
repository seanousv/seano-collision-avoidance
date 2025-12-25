#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("seano_vision")
    cfg = os.path.join(pkg_share, "config", "camera_hp_rtsp.yaml")

    return LaunchDescription([
        Node(
            package="seano_vision",
            executable="camera_node",
            name="camera_hp",
            output="screen",
            emulate_tty=True,
            parameters=[cfg],
        ),
    ])
