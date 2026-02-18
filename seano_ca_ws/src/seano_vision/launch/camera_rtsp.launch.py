#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("seano_vision")

    cfg_in = LaunchConfiguration("cfg").perform(context).strip()
    node_name = LaunchConfiguration("node_name").perform(context).strip()

    # Resolve cfg path:
    # - if absolute and exists -> use it
    # - else try: <pkg_share>/config/<cfg_in>
    if cfg_in and os.path.isabs(cfg_in) and os.path.exists(cfg_in):
        cfg_path = cfg_in
    else:
        cfg_try = os.path.join(pkg_share, "config", cfg_in) if cfg_in else ""
        cfg_path = cfg_try if cfg_try and os.path.exists(cfg_try) else cfg_in

    # Optional overrides: only applied if user provides non-empty argument.
    overrides = {}

    def add_if(name, cast=None):
        val = LaunchConfiguration(name).perform(context)
        if val is None:
            return
        s = str(val).strip()
        if s == "":
            return
        if cast is not None:
            try:
                overrides[name] = cast(s)
            except Exception:
                overrides[name] = s
        else:
            overrides[name] = s

    # Common params (match camera_node.py)
    add_if("source")
    add_if("backend")
    add_if("device_index", int)
    add_if("device_path")
    add_if("device_fourcc")
    add_if("device_width", int)
    add_if("device_height", int)
    add_if("device_fps", int)
    add_if("url")
    add_if("pipeline")

    params = []
    if cfg_path:
        params.append(cfg_path)
    if overrides:
        params.append(overrides)

    return [
        Node(
            package="seano_vision",
            executable="camera_node",
            name=node_name if node_name else "camera_hp",
            output="screen",
            parameters=params,
        )
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory("seano_vision")
    default_cfg = os.path.join(pkg_share, "config", "camera_hp_rtsp.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "cfg",
                default_value=default_cfg,
                description="YAML param file path (absolute) or filename in seano_vision/config/",
            ),
            DeclareLaunchArgument("node_name", default_value="camera_hp"),

            # Optional overrides (prove-empty by default so YAML is not overridden)
            DeclareLaunchArgument("source", default_value=""),
            DeclareLaunchArgument("backend", default_value=""),
            DeclareLaunchArgument("device_index", default_value=""),
            DeclareLaunchArgument("device_path", default_value=""),
            DeclareLaunchArgument("device_fourcc", default_value=""),
            DeclareLaunchArgument("device_width", default_value=""),
            DeclareLaunchArgument("device_height", default_value=""),
            DeclareLaunchArgument("device_fps", default_value=""),
            DeclareLaunchArgument("url", default_value=""),
            DeclareLaunchArgument("pipeline", default_value=""),

            OpaqueFunction(function=_launch_setup),
        ]
    )
