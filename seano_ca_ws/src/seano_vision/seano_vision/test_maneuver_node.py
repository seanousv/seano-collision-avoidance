#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SEANO - Test Maneuver Node (FASE 1)

Tujuan:
- Menghasilkan manuver repeatable untuk validasi kontrol rover-skid / differential thrust.
- Dipakai sebagai "maneuver standar" untuk pengujian + bukti TA.

Publishes:
- /seano/manual/left_cmd   (std_msgs/Float32)
- /seano/manual/right_cmd  (std_msgs/Float32)
- /seano/auto_enable       (std_msgs/Bool)  -> dipaksa False (agar MUX tetap MANUAL)

Sequence default (repeat):
1) WARMUP (opsional) : STOP selama warmup_s
2) FORWARD   t_forward   : left=base right=base
3) TURN_LEFT t_turn_left : left=base-delta right=base+delta
4) TURN_RIGHT t_turn_right: left=base+delta right=base-delta
5) STOP      t_stop      : left=0 right=0

Mode tuning:
A) Mode "base+delta" (recommended): use_base_delta=True
   - base_throttle (0..1)
   - turn_delta    (0..1)
B) Mode legacy: use_base_delta=False
   - forward_left/right, turn_left_left/right, turn_right_left/right

Catatan:
- Jangan jalankan publisher manual lain bersamaan (teleop / topic pub) agar tidak bentrok.
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
        self.declare_parameter("pub_hz", 20.0)  # harus > command_timeout_s mux
        self.declare_parameter("enable_pub_hz", 2.0)  # publish auto_enable False (force MANUAL)
        self.declare_parameter("print_hz", 1.0)  # log ringan
        self.declare_parameter("warmup_s", 0.8)  # publish STOP dulu sebelum mulai stage

        # ---- output clamp ----
        self.declare_parameter("out_min", 0.0)
        self.declare_parameter("out_max", 1.0)
        self.declare_parameter("allow_reverse", False)

        # ---- stage durations (s) ----
        self.declare_parameter("t_forward", 3.0)
        self.declare_parameter("t_turn_left", 2.0)
        self.declare_parameter("t_turn_right", 2.0)
        self.declare_parameter("t_stop", 2.0)
        self.declare_parameter("repeat", True)
        self.declare_parameter("max_cycles", 0)  # 0 = infinite

        # ---- recommended tuning (base + delta) ----
        self.declare_parameter("use_base_delta", True)
        self.declare_parameter("base_throttle", 0.50)  # lebih halus dari 0.55
        self.declare_parameter("turn_delta", 0.05)  # beda kiri-kanan kecil agar tidak "bunga"

        # ---- legacy stage values (0..1) ----
        self.declare_parameter("forward_left", 0.55)
        self.declare_parameter("forward_right", 0.55)
        self.declare_parameter("turn_left_left", 0.35)
        self.declare_parameter("turn_left_right", 0.60)
        self.declare_parameter("turn_right_left", 0.60)
        self.declare_parameter("turn_right_right", 0.35)

        # publishers
        left_topic = str(self.get_parameter("manual_left_topic").value)
        right_topic = str(self.get_parameter("manual_right_topic").value)
        enable_topic = str(self.get_parameter("auto_enable_topic").value)

        self.pub_left = self.create_publisher(Float32, left_topic, 10)
        self.pub_right = self.create_publisher(Float32, right_topic, 10)
        self.pub_auto_enable = self.create_publisher(Bool, enable_topic, 10)

        # state
        self._cycle_count = 0
        self._last_print_t = time.monotonic()
        self._print_period = 1.0 / max(0.1, float(self.get_parameter("print_hz").value))

        # stage plan
        self.stages: List[Stage] = self._build_stages()
        self.stage_idx = 0
        self.stage_start = time.monotonic()

        pub_hz = float(self.get_parameter("pub_hz").value)
        if pub_hz <= 0:
            pub_hz = 20.0
        self.create_timer(1.0 / pub_hz, self._tick)

        en_hz = float(self.get_parameter("enable_pub_hz").value)
        if en_hz > 0:
            self.create_timer(1.0 / en_hz, self._publish_force_manual)

        self.get_logger().info("Test maneuver node STARTED.")
        self.get_logger().info(
            f"Publishing: {left_topic}, {right_topic} | forcing {enable_topic}=False"
        )

        # publish initial force manual + STOP
        self._publish_force_manual()
        self._publish_cmd(0.0, 0.0)

    def _build_stages(self) -> List[Stage]:
        t_forward = float(self.get_parameter("t_forward").value)
        t_left = float(self.get_parameter("t_turn_left").value)
        t_right = float(self.get_parameter("t_turn_right").value)
        t_stop = float(self.get_parameter("t_stop").value)
        warmup = float(self.get_parameter("warmup_s").value)

        use_base_delta = bool(self.get_parameter("use_base_delta").value)

        if use_base_delta:
            base = float(self.get_parameter("base_throttle").value)
            delta = float(self.get_parameter("turn_delta").value)

            forward_l = base
            forward_r = base
            left_l = base - delta
            left_r = base + delta
            right_l = base + delta
            right_r = base - delta
        else:
            forward_l = float(self.get_parameter("forward_left").value)
            forward_r = float(self.get_parameter("forward_right").value)
            left_l = float(self.get_parameter("turn_left_left").value)
            left_r = float(self.get_parameter("turn_left_right").value)
            right_l = float(self.get_parameter("turn_right_left").value)
            right_r = float(self.get_parameter("turn_right_right").value)

        stages: List[Stage] = []
        if warmup > 0.0:
            stages.append(Stage("WARMUP_STOP", warmup, 0.0, 0.0))

        stages.extend(
            [
                Stage("FORWARD", t_forward, forward_l, forward_r),
                Stage("TURN_LEFT", t_left, left_l, left_r),
                Stage("TURN_RIGHT", t_right, right_l, right_r),
                Stage("STOP", t_stop, 0.0, 0.0),
            ]
        )
        return stages

    def _publish_force_manual(self) -> None:
        self.pub_auto_enable.publish(Bool(data=False))

    def _tick(self) -> None:
        now = time.monotonic()

        if not self.stages:
            return

        stage = self.stages[self.stage_idx]
        if (now - self.stage_start) >= stage.duration_s:
            # stage transition
            self.stage_idx += 1
            if self.stage_idx >= len(self.stages):
                # one full cycle completed
                self._cycle_count += 1
                max_cycles = int(self.get_parameter("max_cycles").value)
                if max_cycles > 0 and self._cycle_count >= max_cycles:
                    self._publish_force_manual()
                    self._publish_cmd(0.0, 0.0)
                    self.get_logger().info(
                        f"Completed {self._cycle_count} cycles. Stopping (max_cycles reached)."
                    )
                    rclpy.shutdown()
                    return

                if bool(self.get_parameter("repeat").value):
                    self.stage_idx = 0
                else:
                    self._publish_force_manual()
                    self._publish_cmd(0.0, 0.0)
                    self.get_logger().info("Sequence finished (repeat=false). Output set to STOP.")
                    return

            self.stage_start = now
            stage = self.stages[self.stage_idx]
            self.get_logger().info(f"Stage -> {stage.name} ({stage.duration_s:.1f}s)")

        # clamp outputs
        allow_reverse = bool(self.get_parameter("allow_reverse").value)
        out_min = float(self.get_parameter("out_min").value)
        out_max = float(self.get_parameter("out_max").value)
        if allow_reverse:
            out_min = min(out_min, -1.0)

        left = clamp(float(stage.left), out_min, out_max)
        right = clamp(float(stage.right), out_min, out_max)

        # force manual continuously (safety)
        self._publish_force_manual()
        self._publish_cmd(left, right)

        # light log
        if (now - self._last_print_t) >= self._print_period:
            self._last_print_t = now
            self.get_logger().info(f"{stage.name}: left={left:.2f} right={right:.2f}")

    def _publish_cmd(self, left: float, right: float) -> None:
        self.pub_left.publish(Float32(data=float(left)))
        self.pub_right.publish(Float32(data=float(right)))

    def stop(self) -> None:
        try:
            self._publish_force_manual()
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
