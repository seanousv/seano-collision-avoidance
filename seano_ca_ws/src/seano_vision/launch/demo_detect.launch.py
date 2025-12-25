#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("seano_vision")

    # ---------- Args ----------
    # viewer
    arg_view_annot = DeclareLaunchArgument("view_annot", default_value="true")
    arg_view_raw = DeclareLaunchArgument("view_raw", default_value="false")
    arg_start_delay = DeclareLaunchArgument("start_delay", default_value="0.6")

    # detector topics
    arg_sub_image = DeclareLaunchArgument("sub_image", default_value="/camera/image_raw_reliable")
    arg_pub_image = DeclareLaunchArgument("pub_image", default_value="/camera/image_annotated")
    arg_pub_det = DeclareLaunchArgument("pub_det", default_value="/camera/detections")

    # detector params
    arg_model = DeclareLaunchArgument("model_path", default_value="yolov8n.pt")
    arg_device = DeclareLaunchArgument("device", default_value="cpu")
    arg_imgsz = DeclareLaunchArgument("imgsz", default_value="416")
    arg_conf = DeclareLaunchArgument("conf", default_value="0.25")
    arg_iou = DeclareLaunchArgument("iou", default_value="0.45")
    arg_class_ids = DeclareLaunchArgument("class_ids", default_value="ALL")
    arg_max_fps = DeclareLaunchArgument("max_fps", default_value="10.0")
    arg_qos_depth = DeclareLaunchArgument("qos_depth", default_value="1")

    # QoS strings: "reliable" / "best_effort"
    arg_sub_rel = DeclareLaunchArgument("sub_reliability", default_value="reliable")
    arg_pub_image_rel = DeclareLaunchArgument("pub_image_reliability", default_value="reliable")
    arg_pub_det_rel = DeclareLaunchArgument("pub_det_reliability", default_value="reliable")

    # ---------- Camera launch (reuse yang sudah kamu punya) ----------
    cam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", "camera_hp.launch.py"))
    )

    # ---------- Detector node ----------
    detector = Node(
        package="seano_vision",
        executable="detector_node",
        name="detector_node",
        output="screen",
        parameters=[{
            "sub_image": LaunchConfiguration("sub_image"),
            "pub_image": LaunchConfiguration("pub_image"),
            "pub_det": LaunchConfiguration("pub_det"),

            "model_path": LaunchConfiguration("model_path"),
            "device": LaunchConfiguration("device"),
            "imgsz": LaunchConfiguration("imgsz"),
            "conf": LaunchConfiguration("conf"),
            "iou": LaunchConfiguration("iou"),
            "class_ids": LaunchConfiguration("class_ids"),

            "max_fps": LaunchConfiguration("max_fps"),
            "qos_depth": LaunchConfiguration("qos_depth"),

            "sub_reliability": LaunchConfiguration("sub_reliability"),
            "pub_image_reliability": LaunchConfiguration("pub_image_reliability"),
            "pub_det_reliability": LaunchConfiguration("pub_det_reliability"),
        }]
    )

    start_detector = TimerAction(
        period=LaunchConfiguration("start_delay"),
        actions=[detector]
    )

    # ---------- Viewer (image_tools showimage) ----------
    viewer_annot = Node(
        package="image_tools",
        executable="showimage",
        name="show_annotated",
        output="screen",
        arguments=[
            "--ros-args",
            "-r", ["image:=", LaunchConfiguration("pub_image")]
        ],
        condition=IfCondition(LaunchConfiguration("view_annot"))
    )

    viewer_raw = Node(
        package="image_tools",
        executable="showimage",
        name="show_raw",
        output="screen",
        arguments=[
            "--ros-args",
            "-r", ["image:=", LaunchConfiguration("sub_image")]
        ],
        condition=IfCondition(LaunchConfiguration("view_raw"))
    )

    start_viewers = TimerAction(
        period=LaunchConfiguration("start_delay"),
        actions=[viewer_annot, viewer_raw]
    )

    return LaunchDescription([
        # args
        arg_view_annot, arg_view_raw, arg_start_delay,
        arg_sub_image, arg_pub_image, arg_pub_det,
        arg_model, arg_device, arg_imgsz, arg_conf, arg_iou, arg_class_ids,
        arg_max_fps, arg_qos_depth,
        arg_sub_rel, arg_pub_image_rel, arg_pub_det_rel,

        # actions
        cam_launch,
        start_detector,
        start_viewers,
    ])
