#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("seano_vision")

    # ---------- Args ----------
    # camera include
    use_camera = LaunchConfiguration("use_camera")
    camera_launch = LaunchConfiguration("camera_launch")

    # viewer
    view_annot = LaunchConfiguration("view_annot")
    view_raw = LaunchConfiguration("view_raw")
    start_delay = LaunchConfiguration("start_delay")

    # detector topics
    sub_image = LaunchConfiguration("sub_image")
    pub_image = LaunchConfiguration("pub_image")
    pub_det = LaunchConfiguration("pub_det")

    # detector params
    model_path = LaunchConfiguration("model_path")
    device = LaunchConfiguration("device")
    imgsz = LaunchConfiguration("imgsz")
    conf = LaunchConfiguration("conf")
    iou = LaunchConfiguration("iou")
    class_ids = LaunchConfiguration("class_ids")
    max_fps = LaunchConfiguration("max_fps")
    qos_depth = LaunchConfiguration("qos_depth")

    # QoS strings: "reliable" / "best_effort"
    sub_reliability = LaunchConfiguration("sub_reliability")
    pub_image_reliability = LaunchConfiguration("pub_image_reliability")
    pub_det_reliability = LaunchConfiguration("pub_det_reliability")

    # ---------- Camera launch ----------
    cam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", camera_launch])
        ),
        condition=IfCondition(use_camera),
    )

    # ---------- Detector node ----------
    detector = Node(
        package="seano_vision",
        executable="detector_node",
        name="detector_node",
        output="screen",
        parameters=[
            {
                "sub_image": sub_image,
                "pub_image": pub_image,
                "pub_det": pub_det,
                "model_path": model_path,
                "device": device,
                "imgsz": ParameterValue(imgsz, value_type=int),
                "conf": ParameterValue(conf, value_type=float),
                "iou": ParameterValue(iou, value_type=float),
                "class_ids": class_ids,
                "max_fps": ParameterValue(max_fps, value_type=float),
                "qos_depth": ParameterValue(qos_depth, value_type=int),
                "sub_reliability": sub_reliability,
                "pub_image_reliability": pub_image_reliability,
                "pub_det_reliability": pub_det_reliability,
            }
        ],
    )

    start_detector = TimerAction(
        period=start_delay,
        actions=[detector],
    )

    # ---------- Viewer (opsional) ----------
    viewer_annot = Node(
        package="image_tools",
        executable="showimage",
        name="show_annotated",
        output="screen",
        remappings=[("image", pub_image)],
        condition=IfCondition(view_annot),
    )

    viewer_raw = Node(
        package="image_tools",
        executable="showimage",
        name="show_raw",
        output="screen",
        remappings=[("image", sub_image)],
        condition=IfCondition(view_raw),
    )

    start_viewers = TimerAction(
        period=start_delay,
        actions=[viewer_annot, viewer_raw],
    )

    return LaunchDescription(
        [
            # args
            DeclareLaunchArgument("use_camera", default_value="true"),
            DeclareLaunchArgument(
                "camera_launch",
                default_value="phase2_camera_usb_test.launch.py",
            ),
            DeclareLaunchArgument("view_annot", default_value="false"),
            DeclareLaunchArgument("view_raw", default_value="false"),
            DeclareLaunchArgument("start_delay", default_value="0.6"),
            DeclareLaunchArgument(
                "sub_image",
                default_value="/seano/camera/image_raw_reliable",
            ),
            DeclareLaunchArgument(
                "pub_image",
                default_value="/camera/image_annotated",
            ),
            DeclareLaunchArgument(
                "pub_det",
                default_value="/camera/detections",
            ),
            DeclareLaunchArgument("model_path", default_value="yolov8n.pt"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument("imgsz", default_value="416"),
            DeclareLaunchArgument("conf", default_value="0.25"),
            DeclareLaunchArgument("iou", default_value="0.45"),
            DeclareLaunchArgument("class_ids", default_value="ALL"),
            DeclareLaunchArgument("max_fps", default_value="10.0"),
            DeclareLaunchArgument("qos_depth", default_value="1"),
            DeclareLaunchArgument("sub_reliability", default_value="reliable"),
            DeclareLaunchArgument("pub_image_reliability", default_value="reliable"),
            DeclareLaunchArgument("pub_det_reliability", default_value="reliable"),
            # actions
            cam_launch,
            start_detector,
            start_viewers,
        ]
    )