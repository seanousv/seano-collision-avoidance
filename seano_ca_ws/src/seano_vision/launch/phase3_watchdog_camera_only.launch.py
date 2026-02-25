#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Konfigurasi watchdog untuk FASE 3 (uji camera/detector):
    # - hanya monitor /camera/image_raw
    # - jangan wajibkan /ca/risk dan /ca/mode
    params = {
        # monitor kamera saja
        "image_topics": ["/camera/image_raw"],
        "sub_reliability": "best_effort",
        "image_timeout_s": 2.0,  # cepat terdeteksi kalau kamera mati
        "startup_grace_s": 2.0,  # kasih waktu startup
        # jangan wajib risk/mode di fase ini
        "lost_if_risk_stale": False,
        "lost_if_mode_lost": False,
        "lost_if_mode_stale": False,
        # start normal (biar /ca/failsafe_active bisa False saat sehat)
        "start_in_failsafe": False,
        # output topics (biarin default repo kamu)
        # failsafe_active_topic: /ca/failsafe_active
        # failsafe_reason_topic: /ca/failsafe_reason
    }

    return LaunchDescription(
        [
            Node(
                package="seano_vision",
                executable="watchdog_failsafe_node",
                name="watchdog",
                output="screen",
                emulate_tty=True,
                parameters=[params],
            )
        ]
    )
