#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyboard teleop untuk differential thruster (USV style).

Tujuan desain:
- Default publish ke TOPIC MANUAL agar TIDAK bypass safety pipeline (mux + limiter).
- Opsional: saat teleop aktif, paksa /seano/auto_enable = false (manual override).

Publish (default):
- /seano/manual/left_cmd  (std_msgs/Float32, 0..1)
- /seano/manual/right_cmd (std_msgs/Float32, 0..1)

Tombol:
- w : tambah throttle
- x : kurang throttle
- a : belok kiri (momentary)
- d : belok kanan (momentary)
- s / space : stop (throttle=0, steer=0)
- q : keluar

Catatan:
- Belok "momentary": jika tidak ada input a/d dalam steer_hold_s, steer kembali 0 (lurus).
- Deadman: jika tidak ada input keyboard dalam deadman_timeout_s, throttle otomatis jadi 0 (aman).
"""

import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class _TerminalRaw:
    def __init__(self) -> None:
        self._old = None

    def __enter__(self):
        self._old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


class TeleopDiffThruster(Node):
    def __init__(self) -> None:
        super().__init__("teleop_diff_thruster_node")

        # ===== Topics (DEFAULT: manual topics, sesuai arsitektur repo) =====
        self.declare_parameter("left_topic", "/seano/manual/left_cmd")
        self.declare_parameter("right_topic", "/seano/manual/right_cmd")

        # Opsional: paksa auto_enable=false saat teleop aktif
        self.declare_parameter("auto_enable_topic", "/seano/auto_enable")
        self.declare_parameter("force_auto_disable", True)
        self.declare_parameter("auto_disable_rate_hz", 2.0)

        # ===== Output rate =====
        self.declare_parameter("rate_hz", 20.0)

        # ===== Throttle config =====
        self.declare_parameter("throttle_step", 0.05)  # naik/turun per tekan
        self.declare_parameter("throttle_max", 0.60)  # batas aman test
        self.declare_parameter("throttle_min", 0.0)

        # ===== Steering config (momentary) =====
        self.declare_parameter("steer_step", 0.20)  # besar steer tiap tekan a/d (0..1)
        self.declare_parameter("steer_max", 1.0)
        self.declare_parameter("steer_hold_s", 0.20)  # kalau > ini tidak ada a/d => steer=0

        # ===== Mixer =====
        self.declare_parameter("diff_mix_gain", 0.7)  # left=thr+gain*steer, right=thr-gain*steer

        # ===== Deadman safety (0 = off) =====
        self.declare_parameter("deadman_timeout_s", 2.0)

        # ===== State =====
        self.thr = 0.0  # 0..1
        self.steer = 0.0  # -1..1
        self.last_steer_time = time.time()
        self.last_key_time = time.time()

        left_topic = str(self.get_parameter("left_topic").value)
        right_topic = str(self.get_parameter("right_topic").value)

        self.pub_left = self.create_publisher(Float32, left_topic, 10)
        self.pub_right = self.create_publisher(Float32, right_topic, 10)

        # auto_enable publisher (opsional)
        self._force_auto_disable = bool(self.get_parameter("force_auto_disable").value)
        self._auto_enable_topic = str(self.get_parameter("auto_enable_topic").value)
        self.pub_auto_enable = self.create_publisher(Bool, self._auto_enable_topic, 10)

        hz = float(self.get_parameter("rate_hz").value)
        if hz <= 0.0:
            hz = 20.0
        self.dt = 1.0 / hz
        self.create_timer(self.dt, self._tick)

        # Timer untuk memaksa auto_disable (kalau enable)
        if self._force_auto_disable:
            rate = float(self.get_parameter("auto_disable_rate_hz").value)
            if rate <= 0.0:
                rate = 2.0
            self.create_timer(1.0 / rate, self._tick_force_auto_disable)

        # Publish auto_disable sekali di awal (biar langsung masuk manual)
        if self._force_auto_disable:
            self.pub_auto_enable.publish(Bool(data=False))

        self.get_logger().info("Teleop diff thruster ready.")
        self.get_logger().info(
            "Keys: w/x throttle up/down | a/d turn (momentary) | s/SPACE stop | q quit"
        )
        self.get_logger().info(f"Publishing: left={left_topic} right={right_topic}")
        if self._force_auto_disable:
            self.get_logger().info(
                f"force_auto_disable=TRUE -> publishing {self._auto_enable_topic}=false"
            )

    def _read_key_nonblock(self):
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None

    def _apply_key(self, ch: str) -> None:
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

        # momentary steer
        if ch in ("a", "A"):
            self.steer = clamp(self.steer - steer_step, -steer_max, steer_max)
            self.last_steer_time = now
            return

        if ch in ("d", "D"):
            self.steer = clamp(self.steer + steer_step, -steer_max, steer_max)
            self.last_steer_time = now
            return

    def _tick_force_auto_disable(self) -> None:
        if self._force_auto_disable:
            self.pub_auto_enable.publish(Bool(data=False))

    def _tick(self) -> None:
        # baca semua input yang tersedia
        while True:
            ch = self._read_key_nonblock()
            if ch is None:
                break
            self._apply_key(ch)

        now = time.time()

        # deadman: tidak ada input -> stop
        deadman = float(self.get_parameter("deadman_timeout_s").value)
        if deadman > 0.0 and (now - self.last_key_time) > deadman:
            self.thr = 0.0
            self.steer = 0.0

        # momentary steer decay
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

        # log ringan tiap ~0.5s
        if int(now * 2) != int((now - self.dt) * 2):
            self.get_logger().info(
                f"thr={self.thr:.2f} steer={self.steer:.2f} -> left={left:.2f} right={right:.2f}"
            )


def main(args=None) -> None:
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
            if node._force_auto_disable:
                node.pub_auto_enable.publish(Bool(data=False))
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
