#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # --------------------
    # Launch arguments
    # --------------------
    use_camera = LaunchConfiguration("use_camera")
    camera_launch = LaunchConfiguration("camera_launch")

    use_detector = LaunchConfiguration("use_detector")
    use_risk = LaunchConfiguration("use_risk")
    use_viewer = LaunchConfiguration("use_viewer")

    image_topic = LaunchConfiguration("image_topic")
    detections_topic = LaunchConfiguration("detections_topic")
    annotated_topic = LaunchConfiguration("annotated_topic")
    debug_image_topic = LaunchConfiguration("debug_image_topic")
    mode_topic = LaunchConfiguration("mode_topic")
    metrics_topic = LaunchConfiguration("metrics_topic")

    # detector QoS knobs
    det_sub_reliability = LaunchConfiguration("det_sub_reliability")
    det_pub_reliability = LaunchConfiguration("det_pub_reliability")
    det_qos_depth = LaunchConfiguration("det_qos_depth")

    # --------------------
    # Include camera launch
    # --------------------
    camera_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("seano_vision"), "launch", camera_launch]
            )
        ),
        condition=IfCondition(use_camera),
    )

    # --------------------
    # Detector node
    # --------------------
    detector_node = Node(
        package="seano_vision",
        executable="detector_node",
        name="seano_detector",
        output="screen",
        condition=IfCondition(use_detector),
        parameters=[
            {
                "sub_image": image_topic,
                "pub_det": detections_topic,
                "pub_image": annotated_topic,
                "publish_annotated": True,
                "publish_detections": True,
                "qos_depth": ParameterValue(det_qos_depth, value_type=int),
                "sub_reliability": det_sub_reliability,
                "pub_det_reliability": det_pub_reliability,
                "pub_image_reliability": det_pub_reliability,
            }
        ],
    )

    # --------------------
    # Risk evaluator node
    # --------------------
    risk_node = Node(
        package="seano_vision",
        executable="risk_evaluator_node",
        name="seano_risk_eval",
        output="screen",
        condition=IfCondition(use_risk),
        parameters=[
            {
                "detections_topic": detections_topic,
                "image_topic": image_topic,
                "risk_topic": "/ca/risk",
                "command_topic": "/ca/command",
                "mode_topic": mode_topic,
                "metrics_topic": metrics_topic,
                "debug_image_topic": debug_image_topic,
                "publish_debug_image": True,
            }
        ],
    )

    # --------------------
    # Viewer node (opsional)
    # Default dimatikan karena baseline monitoring sekarang via browser/web_video_server
    # --------------------
    viewer_node = Node(
        package="image_tools",
        executable="showimage",
        name="seano_viewer",
        output="screen",
        condition=IfCondition(use_viewer),
        remappings=[
            ("image", debug_image_topic),
        ],
    )

    return LaunchDescription(
        [
            # toggles
            DeclareLaunchArgument("use_camera", default_value="true"),
            DeclareLaunchArgument("use_detector", default_value="true"),
            DeclareLaunchArgument("use_risk", default_value="true"),
            DeclareLaunchArgument("use_viewer", default_value="false"),
            # camera launch selector
            DeclareLaunchArgument(
                "camera_launch",
                default_value="phase2_camera_usb_test.launch.py",
            ),
            # topics
            DeclareLaunchArgument(
                "image_topic",
                default_value="/seano/camera/image_raw_reliable",
            ),
            DeclareLaunchArgument(
                "detections_topic",
                default_value="/camera/detections",
            ),
            DeclareLaunchArgument(
                "annotated_topic",
                default_value="/camera/image_annotated",
            ),
            DeclareLaunchArgument(
                "debug_image_topic",
                default_value="/ca/debug_image",
            ),
            DeclareLaunchArgument(
                "mode_topic",
                default_value="/ca/mode",
            ),
            DeclareLaunchArgument(
                "metrics_topic",
                default_value="/ca/metrics",
            ),
            # detector QoS
            DeclareLaunchArgument(
                "det_sub_reliability",
                default_value="reliable",
            ),
            DeclareLaunchArgument(
                "det_pub_reliability",
                default_value="reliable",
            ),
            DeclareLaunchArgument(
                "det_qos_depth",
                default_value="10",
            ),
            # actions
            camera_include,
            detector_node,
            risk_node,
            viewer_node,
        ]
    )