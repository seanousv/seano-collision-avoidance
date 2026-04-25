#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust camera source node for SEANO.

Main fix:
- Local video file input is treated as a looped simulation source.
- EOF or transient read failure on a video file immediately seeks back to frame 0.
- Device camera failures still trigger reopen logic.
- Publishes both:
  /seano/camera/image_raw_reliable
  /seano/camera/image_raw

This avoids LOST_PERCEPTION caused by MP4 EOF/reopen gaps in Phase 5 simulation.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Optional, Tuple

import cv2
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image


class CameraSource(Node):
    def __init__(self) -> None:
        super().__init__("camera_source")

        self._declare_params()
        self.bridge = CvBridge()

        self.source = str(self.get_parameter("source").value).strip().lower()
        self.backend = str(self.get_parameter("backend").value).strip().lower()
        self.url = os.path.expanduser(str(self.get_parameter("url").value).strip())
        self.pipeline = str(self.get_parameter("pipeline").value)
        self.device_path = str(self.get_parameter("device_path").value).strip()
        self.device_index = int(self.get_parameter("device_index").value)
        self.device_width = int(self.get_parameter("device_width").value)
        self.device_height = int(self.get_parameter("device_height").value)
        self.device_fps = float(self.get_parameter("device_fps").value)
        self.device_fourcc = str(self.get_parameter("device_fourcc").value).strip()
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.max_age_ms = float(self.get_parameter("max_age_ms").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.swap_rb = bool(self.get_parameter("swap_rb").value)
        self.resize_width = int(self.get_parameter("resize_width").value)
        self.resize_height = int(self.get_parameter("resize_height").value)

        self.raw_topic = str(self.get_parameter("raw_topic").value)
        self.raw_reliable_topic = str(self.get_parameter("raw_reliable_topic").value)

        if self.max_fps <= 0.0:
            self.max_fps = 4.0

        self.is_local_file = self.source == "url" and self.url and Path(self.url).is_file()
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame_wall = 0.0
        self.last_stats_wall = time.time()
        self.stats_pub_count = 0
        self.stats_read_count = 0
        self.total_pub_count = 0
        self.total_read_count = 0
        self.fail_count = 0
        self.loop_count = 0

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub_raw = self.create_publisher(Image, self.raw_topic, qos_profile_sensor_data)
        self.pub_raw_reliable = self.create_publisher(Image, self.raw_reliable_topic, reliable_qos)

        self._open_capture(initial=True)

        period = 1.0 / self.max_fps
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            "Started camera source | "
            f"source={self.source} backend={self.backend} "
            f"url={self.url if self.url else '--'} "
            f"device={self._device_label()} "
            f"is_local_file={self.is_local_file} "
            f"max_fps={self.max_fps:.2f} "
            f"pub_raw={self.raw_topic} pub_reliable={self.raw_reliable_topic}"
        )

    def _declare_params(self) -> None:
        dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("source", "url", dyn)
        self.declare_parameter("backend", "opencv", dyn)
        self.declare_parameter("url", "", dyn)
        self.declare_parameter("pipeline", "", dyn)
        self.declare_parameter("device_path", "/dev/video0", dyn)
        self.declare_parameter("device_index", 0, dyn)
        self.declare_parameter("device_width", 640, dyn)
        self.declare_parameter("device_height", 480, dyn)
        self.declare_parameter("device_fps", 30.0, dyn)
        self.declare_parameter("device_fourcc", "", dyn)
        self.declare_parameter("max_fps", 4.0, dyn)
        self.declare_parameter("max_age_ms", 2000.0, dyn)
        self.declare_parameter("frame_id", "seano_camera", dyn)
        self.declare_parameter("swap_rb", False, dyn)
        self.declare_parameter("resize_width", 0, dyn)
        self.declare_parameter("resize_height", 0, dyn)

        self.declare_parameter("raw_topic", "/seano/camera/image_raw", dyn)
        self.declare_parameter("raw_reliable_topic", "/seano/camera/image_raw_reliable", dyn)

        # Kept for launch compatibility with older files.
        self.declare_parameter("publish_in_reader", True, dyn)
        self.declare_parameter("rtsp_tcp", True, dyn)
        self.declare_parameter("gst_latency_ms", 80, dyn)

    def _device_label(self) -> str:
        if self.device_path:
            return self.device_path
        return f"index:{self.device_index}"

    def _open_capture(self, initial: bool = False) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        cap = None

        if self.source == "pipeline":
            if not self.pipeline.strip():
                self.get_logger().error("source=pipeline but pipeline parameter is empty")
                return False
            cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)

        elif self.source == "url":
            if not self.url:
                self.get_logger().error("source=url but url parameter is empty")
                return False

            if self.backend in ("gstreamer", "gst"):
                cap = cv2.VideoCapture(self.url, cv2.CAP_GSTREAMER)
            else:
                cap = cv2.VideoCapture(self.url)

        elif self.source == "device":
            dev = self.device_path if self.device_path else self.device_index
            cap = cv2.VideoCapture(dev)

            if cap is not None and cap.isOpened():
                if self.device_width > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.device_width))
                if self.device_height > 0:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.device_height))
                if self.device_fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, float(self.device_fps))
                if self.device_fourcc:
                    fourcc = cv2.VideoWriter_fourcc(*self.device_fourcc[:4])
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        else:
            self.get_logger().error(f"Unknown source={self.source}; valid: url, device, pipeline")
            return False

        if cap is None or not cap.isOpened():
            self.get_logger().error(
                "Capture open failed | "
                f"source={self.source} backend={self.backend} "
                f"url={self.url if self.url else '--'} device={self._device_label()}"
            )
            return False

        self.cap = cap
        self.fail_count = 0

        if self.is_local_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.get_logger().info(
                "Capture opened | "
                f"file={self.url} video_fps={fps:.2f} frames={frames:.0f} "
                f"publish_fps={self.max_fps:.2f}"
            )
        else:
            if initial:
                self.get_logger().info(
                    "Capture opened | "
                    f"source={self.source} backend={self.backend} "
                    f"url={self.url if self.url else '--'} device={self._device_label()}"
                )
            else:
                self.get_logger().info("Capture reopened")

        return True

    def _read_frame(self) -> Tuple[bool, Optional[object]]:
        if self.cap is None or not self.cap.isOpened():
            ok = self._open_capture(initial=False)
            if not ok:
                return False, None

        ok, frame = self.cap.read()

        if ok and frame is not None:
            self.fail_count = 0
            return True, frame

        # Critical Phase 5 fix:
        # Local MP4 EOF must loop immediately, not wait/reopen like a USB camera.
        if self.is_local_file:
            self.loop_count += 1
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None

            if ok and frame is not None:
                self.fail_count = 0
                if self.loop_count == 1 or self.loop_count % 20 == 0:
                    self.get_logger().info(
                        f"Video looped | file={self.url} loop_count={self.loop_count}"
                    )
                return True, frame

            # If seeking failed, reopen once and try again.
            self._open_capture(initial=False)
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self.fail_count = 0
                    return True, frame

            self.fail_count += 1
            if self.fail_count in (1, 5, 10) or self.fail_count % 30 == 0:
                self.get_logger().warn(
                    "Video file read failed after loop/reopen | "
                    f"cnt={self.fail_count} file={self.url}"
                )
            return False, None

        # Device or stream failure path.
        self.fail_count += 1
        if self.fail_count in (1, 5, 10) or self.fail_count % 30 == 0:
            self.get_logger().warn(
                "cap.read() failed | "
                f"cnt={self.fail_count} source={self.source} "
                f"url={self.url if self.url else '--'} device={self._device_label()}"
            )

        if self.fail_count >= 10:
            self._open_capture(initial=False)

        return False, None

    def _preprocess(self, frame):
        if self.swap_rb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.resize_width > 0 and self.resize_height > 0:
            frame = cv2.resize(
                frame, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA
            )

        return frame

    def _tick(self) -> None:
        ok, frame = self._read_frame()
        now = time.time()

        if not ok or frame is None:
            self._maybe_log_stats(now)
            return

        frame = self._preprocess(frame)

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        self.pub_raw.publish(msg)
        self.pub_raw_reliable.publish(msg)

        self.last_frame_wall = now
        self.total_read_count += 1
        self.total_pub_count += 1
        self.stats_read_count += 1
        self.stats_pub_count += 1

        self._maybe_log_stats(now)

    def _maybe_log_stats(self, now: float) -> None:
        dt = now - self.last_stats_wall
        if dt < 2.0:
            return

        pub_fps = self.stats_pub_count / dt if dt > 0.0 else 0.0
        cap_fps = self.stats_read_count / dt if dt > 0.0 else 0.0
        age_ms = (now - self.last_frame_wall) * 1000.0 if self.last_frame_wall > 0.0 else 1.0e9

        src_label = self.url if self.source == "url" else self._device_label()

        self.get_logger().info(
            "stats | "
            f"source={self.source} input={src_label} "
            f"cap_fps={cap_fps:.1f} pub_fps={pub_fps:.1f} "
            f"age={age_ms:.0f}ms enc=bgr8 max_fps={self.max_fps:.1f} "
            f"loops={self.loop_count}"
        )

        self.last_stats_wall = now
        self.stats_pub_count = 0
        self.stats_read_count = 0

    def destroy_node(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraSource()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
