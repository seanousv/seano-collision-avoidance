#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("seano_vision")
    camera_cfg = os.path.join(pkg_share, "config", "camera_usb.yaml")

    # Model path (kalau file ada di share/models)
    model_path = os.path.join(pkg_share, "models", "yolov8n.pt")

    # Kamera: pakai setting yang sudah terbukti stabil ~15 Hz (best effort only)
    camera_overrides = {
        "source": "device",
        "backend": "opencv",
        "device_index": 0,
        "device_path": "/dev/video0",
        "device_fourcc": "MJPG",
        "device_width": 640,
        "device_height": 480,
        "device_fps": 30,
        "max_fps": 15.0,
        "max_age_ms": 120,
        "grab_skip": 0,
        "publish_in_reader": False,
        "output_encoding": "bgr8",
        "swap_rb": False,
        "publish_best_effort": True,
        "publish_reliable": False,
        "topic_best_effort": "/camera/image_raw",
        "topic_reliable": "/camera/image_raw_reliable",
        "reconnect_sec": 0.5,
        "log_stats_sec": 2.0,
    }

    camera_node = Node(
        package="seano_vision",
        executable="camera_node",
        name="camera_hp",
        output="screen",
        emulate_tty=True,
        parameters=[camera_cfg, camera_overrides],
    )

    # Detector: paksa subscribe ke /camera/image_raw (BEST_EFFORT) + output deteksi stabil
    detector_overrides = {
        "sub_image": "/camera/image_raw",
        "sub_reliability": "best_effort",
        "pub_det": "/camera/detections",
        "pub_det_reliability": "best_effort",
        # untuk hemat CPU di WSL: matikan annotated dulu
        "publish_annotated": False,
        "publish_detections": True,
        "publish_empty_detections": True,
        # model + performa
        "model_path": model_path,  # kalau tidak ada, node akan fallback resolve internal
        "device": "cpu",
        "imgsz": 416,
        "conf": 0.25,
        "iou": 0.45,
        "class_ids": "ALL",
        "max_det": 50,
        "max_fps": 8.0,  # aman di WSL; nanti bisa dinaikkan
        "qos_depth": 1,
        "warmup": True,
        "stats_period": 1.0,
    }

    detector_node = Node(
        package="seano_vision",
        executable="detector_node",
        name="detector_node",
        output="screen",
        emulate_tty=True,
        parameters=[detector_overrides],
    )

    # Start detektor setelah kamera publish dulu
    delayed_detector = TimerAction(period=1.0, actions=[detector_node])

    return LaunchDescription([camera_node, delayed_detector])
