#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO - Test Maneuver Node (FASE 1)

Tujuan:
- Menghasilkan manuver repeatable untuk validasi kontrol rover-skid / differential thrust.

Publishes:
- /seano/manual/left_cmd   (std_msgs/Float32)
- /seano/manual/right_cmd  (std_msgs/Float32)
- /seano/auto_enable       (std_msgs/Bool)  -> dipaksa False (agar MUX tetap MANUAL)

Sequence default (repeat):
1) FORWARD   3.0 s : left=0.55 right=0.55
2) TURN_LEFT 2.0 s : left=0.35 right=0.60
3) TURN_RIGHT 2.0 s: left=0.60 right=0.35
4) STOP      2.0 s : left=0.00 right=0.00

Catatan penting:
- Jangan jalankan teleop/manual publisher lain bersamaan, supaya tidak bentrok (publisher dobel).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Stage:
    name: str
    duration_s: float
    left: float
    right: float


class TestManeuverNode(Node):
    def __init__(self) -> None:
        super().__init__("test_maneuver_node")

        # ---- topics ----
        self.declare_parameter("manual_left_topic", "/seano/manual/left_cmd")
        self.declare_parameter("manual_right_topic", "/seano/manual/right_cmd")
        self.declare_parameter("auto_enable_topic", "/seano/auto_enable")

        # ---- timing ----
        self.declare_parameter("pub_hz", 20.0)  # harus > command_timeout_s mux (default 0.6s)
        self.declare_parameter("enable_pub_hz", 2.0)  # publish auto_enable False (force MANUAL)

        # ---- output clamp ----
        self.declare_parameter("out_min", 0.0)
        self.declare_parameter("out_max", 1.0)

        # ---- stage durations (s) ----
        self.declare_parameter("t_forward", 3.0)
        self.declare_parameter("t_turn_left", 2.0)
        self.declare_parameter("t_turn_right", 2.0)
        self.declare_parameter("t_stop", 2.0)
        self.declare_parameter("repeat", True)

        # ---- stage values (0..1) ----
        self.declare_parameter("forward_left", 0.55)
        self.declare_parameter("forward_right", 0.55)
        self.declare_parameter("turn_left_left", 0.35)
        self.declare_parameter("turn_left_right", 0.60)
        self.declare_parameter("turn_right_left", 0.60)
        self.declare_parameter("turn_right_right", 0.35)

        # publishers
        self.pub_left = self.create_publisher(
            Float32, self.get_parameter("manual_left_topic").value, 10
        )
        self.pub_right = self.create_publisher(
            Float32, self.get_parameter("manual_right_topic").value, 10
        )
        self.pub_auto_enable = self.create_publisher(
            Bool, self.get_parameter("auto_enable_topic").value, 10
        )

        # build sequence
        self.stages: List[Stage] = self._build_stages()
        self.stage_idx = 0
        self.stage_start = time.monotonic()
        self._last_print = 0.0

        pub_hz = float(self.get_parameter("pub_hz").value)
        if pub_hz <= 0:
            pub_hz = 20.0
        self.create_timer(1.0 / pub_hz, self._tick)

        en_hz = float(self.get_parameter("enable_pub_hz").value)
        if en_hz > 0:
            self.create_timer(1.0 / en_hz, self._publish_force_manual)

        self.get_logger().info("Test maneuver node STARTED.")
        self.get_logger().info("Publishing /seano/manual/* and forcing /seano/auto_enable=False.")

    def _build_stages(self) -> List[Stage]:
        return [
            Stage(
                "FORWARD",
                float(self.get_parameter("t_forward").value),
                float(self.get_parameter("forward_left").value),
                float(self.get_parameter("forward_right").value),
            ),
            Stage(
                "TURN_LEFT",
                float(self.get_parameter("t_turn_left").value),
                float(self.get_parameter("turn_left_left").value),
                float(self.get_parameter("turn_left_right").value),
            ),
            Stage(
                "TURN_RIGHT",
                float(self.get_parameter("t_turn_right").value),
                float(self.get_parameter("turn_right_left").value),
                float(self.get_parameter("turn_right_right").value),
            ),
            Stage(
                "STOP",
                float(self.get_parameter("t_stop").value),
                0.0,
                0.0,
            ),
        ]

    def _publish_force_manual(self) -> None:
        # Pastikan mux tetap MANUAL (auto_enable=false)
        self.pub_auto_enable.publish(Bool(data=False))

    def _tick(self) -> None:
        now = time.monotonic()

        if not self.stages:
            return

        stage = self.stages[self.stage_idx]
        if (now - self.stage_start) >= stage.duration_s:
            self.stage_idx += 1
            if self.stage_idx >= len(self.stages):
                if bool(self.get_parameter("repeat").value):
                    self.stage_idx = 0
                else:
                    self._publish_cmd(0.0, 0.0)
                    self.get_logger().info("Sequence finished (repeat=false). Output set to STOP.")
                    return
            self.stage_start = now
            stage = self.stages[self.stage_idx]
            self.get_logger().info(f"Stage -> {stage.name} ({stage.duration_s:.1f}s)")

        out_min = float(self.get_parameter("out_min").value)
        out_max = float(self.get_parameter("out_max").value)

        left = clamp(stage.left, out_min, out_max)
        right = clamp(stage.right, out_min, out_max)

        self._publish_cmd(left, right)

        # print ringan tiap ~1s
        if (now - self._last_print) > 1.0:
            self._last_print = now
            self.get_logger().info(f"{stage.name}: left={left:.2f} right={right:.2f}")

    def _publish_cmd(self, left: float, right: float) -> None:
        self.pub_left.publish(Float32(data=float(left)))
        self.pub_right.publish(Float32(data=float(right)))

    def stop(self) -> None:
        try:
            self.pub_auto_enable.publish(Bool(data=False))
            self._publish_cmd(0.0, 0.0)
        except Exception:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TestManeuverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
