#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Keyboard teleop untuk differential thruster (USV style).
Publish:
  - /seano/left_cmd  (std_msgs/Float32)
  - /seano/right_cmd (std_msgs/Float32)

Tombol:
  w : tambah throttle
  x : kurang throttle
  a : belok kiri (momentary)
  d : belok kanan (momentary)
  s / space : stop (throttle=0, steer=0)
  q : keluar

Catatan:
- Belok bersifat "momentary": kalau tidak ada input a/d dalam steer_hold_s, steer kembali 0 (lurus).
- Ada deadman_timeout_s: kalau tidak ada input keyboard sama sekali dalam waktu itu, throttle otomatis jadi 0 (aman).
  Set 0 untuk mematikan deadman.
"""

import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class _TerminalRaw:
    def __init__(self):
        self._old = None

    def __enter__(self):
        self._old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


class TeleopDiffThruster(Node):
    def __init__(self):
        super().__init__("teleop_diff_thruster_node")

        # Topics
        self.declare_parameter("left_topic", "/seano/left_cmd")
        self.declare_parameter("right_topic", "/seano/right_cmd")

        # Output rate
        self.declare_parameter("rate_hz", 20.0)

        # Throttle config
        self.declare_parameter("throttle_step", 0.05)  # naik/turun per tekan
        self.declare_parameter("throttle_max", 0.60)  # batas aman test
        self.declare_parameter("throttle_min", 0.0)

        # Steering config (momentary)
        self.declare_parameter("steer_step", 0.20)  # besar steer tiap tekan a/d (0..1)
        self.declare_parameter("steer_max", 1.0)
        self.declare_parameter("steer_hold_s", 0.20)  # kalau > ini tidak ada a/d => steer=0

        # Mixer
        self.declare_parameter("diff_mix_gain", 0.7)  # left=thr+gain*steer, right=thr-gain*steer

        # Deadman safety (0 = off)
        self.declare_parameter("deadman_timeout_s", 2.0)

        # State
        self.thr = 0.0  # 0..1
        self.steer = 0.0  # -1..1
        self.last_steer_time = time.time()
        self.last_key_time = time.time()

        left_topic = self.get_parameter("left_topic").value
        right_topic = self.get_parameter("right_topic").value
        self.pub_left = self.create_publisher(Float32, left_topic, 10)
        self.pub_right = self.create_publisher(Float32, right_topic, 10)

        hz = float(self.get_parameter("rate_hz").value)
        if hz <= 0:
            hz = 20.0
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self._tick)

        self.get_logger().info("Teleop diff thruster ready.")
        self.get_logger().info(
            "Keys: w/x throttle up/down | a/d turn (momentary) | s/SPACE stop | q quit"
        )
        self.get_logger().info(
            "Tip: untuk mode paddle/left-right di Rover, pastikan PILOT_STEER_TYPE=1."
        )

    def _read_key_nonblock(self):
        # Non-blocking read 1 char if available
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            ch = sys.stdin.read(1)
            return ch
        return None

    def _apply_key(self, ch: str):
        now = time.time()
        self.last_key_time = now

        thr_step = float(self.get_parameter("throttle_step").value)
        thr_max = float(self.get_parameter("throttle_max").value)
        thr_min = float(self.get_parameter("throttle_min").value)

        steer_step = float(self.get_parameter("steer_step").value)
        steer_max = float(self.get_parameter("steer_max").value)

        if ch in ("q", "Q"):
            raise KeyboardInterrupt

        if ch in ("s", "S", " "):
            self.thr = 0.0
            self.steer = 0.0
            self.last_steer_time = now
            return

        if ch in ("w", "W"):
            self.thr = clamp(self.thr + thr_step, thr_min, thr_max)
            return

        if ch in ("x", "X"):
            self.thr = clamp(self.thr - thr_step, thr_min, thr_max)
            return

        # momentary steer: tiap tekan a/d update steer_time
        if ch in ("a", "A"):
            self.steer = clamp(self.steer + steer_step * (-1.0), -steer_max, steer_max)
            self.last_steer_time = now
            return

        if ch in ("d", "D"):
            self.steer = clamp(self.steer + steer_step * (1.0), -steer_max, steer_max)
            self.last_steer_time = now
            return

    def _tick(self):
        # read all available keys quickly
        try:
            while True:
                ch = self._read_key_nonblock()
                if ch is None:
                    break
                self._apply_key(ch)
        except KeyboardInterrupt:
            raise

        now = time.time()

        # deadman: kalau user tidak input apa pun terlalu lama -> stop
        deadman = float(self.get_parameter("deadman_timeout_s").value)
        if deadman > 0.0 and (now - self.last_key_time) > deadman:
            self.thr = 0.0
            self.steer = 0.0

        # momentary steer: kalau tidak ada a/d cukup lama -> steer balik 0
        hold = float(self.get_parameter("steer_hold_s").value)
        if (now - self.last_steer_time) > hold:
            self.steer = 0.0

        gain = float(self.get_parameter("diff_mix_gain").value)
        left = self.thr + gain * self.steer
        right = self.thr - gain * self.steer

        # clamp output (0..1)
        left = clamp(left, 0.0, 1.0)
        right = clamp(right, 0.0, 1.0)

        self.pub_left.publish(Float32(data=float(left)))
        self.pub_right.publish(Float32(data=float(right)))

        # log periodik ringan (tiap ~0.5s)
        if int(now * 2) != int((now - self.dt) * 2):
            self.get_logger().info(
                f"thr={self.thr:.2f} steer={self.steer:.2f} -> left={left:.2f} right={right:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TeleopDiffThruster()
    try:
        with _TerminalRaw():
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # stop output on exit
        try:
            node.pub_left.publish(Float32(data=0.0))
            node.pub_right.publish(Float32(data=0.0))
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
