#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist


@dataclass
class ActuatorCmd:
    throttle: float  # 0..1
    rudder: float    # -1..1


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def slew_limit(current: float, target: float, rate_per_s: float, dt: float) -> float:
    if rate_per_s <= 0.0 or dt <= 0.0:
        return target
    max_delta = rate_per_s * dt
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


class ActuatorSafetyLimiterNode(Node):
    def __init__(self) -> None:
        super().__init__("actuator_safety_limiter_node")

        # Topics
        self.declare_parameter("command_topic", "/ca/command")
        self.declare_parameter("failsafe_active_topic", "/ca/failsafe_active")

        self.declare_parameter("out_throttle_topic", "/seano/throttle_cmd")
        self.declare_parameter("out_rudder_topic", "/seano/rudder_cmd")
        self.declare_parameter("out_twist_topic", "/seano/cmd_vel")
        self.declare_parameter("out_command_final_topic", "/seano/command_final")

        # Mapping
        self.declare_parameter("throttle_hold", 0.35)
        self.declare_parameter("throttle_slow", 0.20)
        self.declare_parameter("throttle_stop", 0.0)

        self.declare_parameter("rudder_hold", 0.0)
        self.declare_parameter("rudder_turn_slow", 0.35)
        self.declare_parameter("rudder_turn", 0.55)

        # Safety
        self.declare_parameter("command_timeout_s", 1.0)
        self.declare_parameter("failsafe_timeout_s", 2.0)

        # NEW: kalau failsafe topic tidak ada / stale, mau dianggap active atau tidak?
        # Default true (lebih aman). Untuk testing set false.
        self.declare_parameter("failsafe_stale_is_active", True)

        self.declare_parameter("loop_hz", 20.0)
        self.declare_parameter("slew_throttle_per_s", 0.8)
        self.declare_parameter("slew_rudder_per_s", 1.5)

        self.declare_parameter("publish_twist", True)
        self.declare_parameter("twist_linear_x_scale", 1.0)
        self.declare_parameter("twist_angular_z_scale", 1.0)

        self.declare_parameter("invert_rudder", False)
        self.declare_parameter("unknown_command_policy", "HOLD")  # HOLD / STOP

        # State
        self.last_cmd_str: str = "HOLD_COURSE"
        self.last_cmd_time: float = 0.0

        self.last_failsafe_active: bool = False
        self.last_failsafe_time: float = 0.0

        self.throttle_cmd: float = 0.0
        self.rudder_cmd: float = 0.0
        self._last_tick = time.time()
        self._last_reason_print = 0.0

        # I/O
        cmd_topic = self.get_parameter("command_topic").value
        fs_topic = self.get_parameter("failsafe_active_topic").value

        self.create_subscription(String, cmd_topic, self._on_command, 10)
        self.create_subscription(Bool, fs_topic, self._on_failsafe, 10)

        self.pub_thr = self.create_publisher(Float32, self.get_parameter("out_throttle_topic").value, 10)
        self.pub_rud = self.create_publisher(Float32, self.get_parameter("out_rudder_topic").value, 10)
        self.pub_twist = self.create_publisher(Twist, self.get_parameter("out_twist_topic").value, 10)
        self.pub_final = self.create_publisher(String, self.get_parameter("out_command_final_topic").value, 10)

        loop_hz = float(self.get_parameter("loop_hz").value)
        loop_hz = 20.0 if loop_hz <= 0 else loop_hz
        self.create_timer(1.0 / loop_hz, self._on_tick)

        self.get_logger().info(
            f"Started | cmd={cmd_topic} failsafe={fs_topic} | loop={loop_hz:.1f}Hz"
        )

    def _on_command(self, msg: String) -> None:
        self.last_cmd_str = (msg.data or "").strip()
        self.last_cmd_time = time.time()

    def _on_failsafe(self, msg: Bool) -> None:
        self.last_failsafe_active = bool(msg.data)
        self.last_failsafe_time = time.time()

    def _map_command_to_target(self, cmd: str) -> ActuatorCmd:
        thr_hold = float(self.get_parameter("throttle_hold").value)
        thr_slow = float(self.get_parameter("throttle_slow").value)
        thr_stop = float(self.get_parameter("throttle_stop").value)

        rud_hold = float(self.get_parameter("rudder_hold").value)
        rud_turn_slow = float(self.get_parameter("rudder_turn_slow").value)
        rud_turn = float(self.get_parameter("rudder_turn").value)

        c = cmd.upper().strip()

        if c in ("HOLD", "HOLD_COURSE", "KEEP", "KEEP_COURSE"):
            return ActuatorCmd(thr_hold, rud_hold)
        if c in ("SLOW", "SLOW_DOWN", "DECELERATE"):
            return ActuatorCmd(thr_slow, rud_hold)
        if c in ("STOP", "BRAKE", "EMERGENCY_STOP"):
            return ActuatorCmd(thr_stop, rud_hold)

        if c in ("TURN_LEFT_SLOW", "LEFT_SLOW"):
            return ActuatorCmd(thr_slow, -abs(rud_turn_slow))
        if c in ("TURN_RIGHT_SLOW", "RIGHT_SLOW"):
            return ActuatorCmd(thr_slow, abs(rud_turn_slow))
        if c in ("TURN_LEFT", "LEFT"):
            return ActuatorCmd(thr_hold, -abs(rud_turn))
        if c in ("TURN_RIGHT", "RIGHT"):
            return ActuatorCmd(thr_hold, abs(rud_turn))

        policy = str(self.get_parameter("unknown_command_policy").value).upper()
        if policy == "STOP":
            return ActuatorCmd(thr_stop, rud_hold)
        return ActuatorCmd(thr_hold, rud_hold)

    def _on_tick(self) -> None:
        now = time.time()
        dt = max(1e-3, now - self._last_tick)
        self._last_tick = now

        cmd_timeout = float(self.get_parameter("command_timeout_s").value)
        fs_timeout = float(self.get_parameter("failsafe_timeout_s").value)
        stale_is_active = bool(self.get_parameter("failsafe_stale_is_active").value)

        # Ages
        cmd_age = now - self.last_cmd_time if self.last_cmd_time > 0 else 1e9
        fs_age = now - self.last_failsafe_time if self.last_failsafe_time > 0 else 1e9

        cmd_ok = cmd_age <= cmd_timeout
        fs_stale = fs_age > fs_timeout

        # Failsafe decision
        failsafe_active = self.last_failsafe_active or (fs_stale and stale_is_active)

        # Final cmd decision
        reason = "ok"
        final_cmd = self.last_cmd_str if cmd_ok else "STOP"
        if not cmd_ok:
            reason = f"cmd_timeout({cmd_age:.2f}s)"
        if failsafe_active:
            final_cmd = "STOP"
            reason = "failsafe_true" if self.last_failsafe_active else f"failsafe_stale({fs_age:.2f}s)"

        # Print reason occasionally (biar kamu gampang debug)
        if now - self._last_reason_print > 1.0:
            self._last_reason_print = now
            self.get_logger().info(f"final={final_cmd} reason={reason} cmd_age={cmd_age:.2f}s fs_age={fs_age:.2f}s")

        target = self._map_command_to_target(final_cmd)

        if bool(self.get_parameter("invert_rudder").value):
            target = ActuatorCmd(target.throttle, -target.rudder)

        self.throttle_cmd = slew_limit(self.throttle_cmd, target.throttle, float(self.get_parameter("slew_throttle_per_s").value), dt)
        self.rudder_cmd = slew_limit(self.rudder_cmd, target.rudder, float(self.get_parameter("slew_rudder_per_s").value), dt)

        self.throttle_cmd = clamp(self.throttle_cmd, 0.0, 1.0)
        self.rudder_cmd = clamp(self.rudder_cmd, -1.0, 1.0)

        self.pub_thr.publish(Float32(data=float(self.throttle_cmd)))
        self.pub_rud.publish(Float32(data=float(self.rudder_cmd)))
        self.pub_final.publish(String(data=str(final_cmd)))

        if bool(self.get_parameter("publish_twist").value):
            tw = Twist()
            tw.linear.x = float(self.throttle_cmd) * float(self.get_parameter("twist_linear_x_scale").value)
            tw.angular.z = float(self.rudder_cmd) * float(self.get_parameter("twist_angular_z_scale").value)
            self.pub_twist.publish(tw)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActuatorSafetyLimiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
