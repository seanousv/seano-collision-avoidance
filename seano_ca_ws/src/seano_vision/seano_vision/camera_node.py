#!/usr/bin/env python3
"""
camera_node.py (ROS 2 Humble) - SEANO Vision

Fokus: stream HP (IP Webcam) -> ROS Image topic dengan latency rendah & warna konsisten.

Fitur utama:
- Input:
  - source=device: device_index (0/1/2...)
  - source=url: url http mjpeg / rtsp
  - source=pipeline: gstreamer pipeline custom (override semuanya)
- Backend:
  - backend=gstreamer (disarankan untuk RTSP/MJPEG)
  - backend=opencv (fallback)
- Low latency:
  - GStreamer appsink: drop=true max-buffers=1 sync=false
  - reader thread terpisah
  - drop frame tua (max_age_ms)
  - optional publish_in_reader (publish secepat capture, dibatasi max_fps)
- QoS:
  - publish_best_effort: /camera/image_raw
  - publish_reliable:   /camera/image_raw_reliable
- Warna:
  - output_encoding: bgr8 / rgb8
  - swap_rb: tukar R<->B (toggle LIVE)

Run (RTSP IP Webcam):
  ros2 run seano_vision camera_node --ros-args \
    -r __node:=camera_hp \
    -p source:=url \
    -p url:="rtsp://192.168.1.7:8080/h264.sdp" \
    -p backend:=gstreamer \
    -p rtsp_tcp:=true \
    -p gstreamer_latency_ms:=0 \
    -p publish_in_reader:=true \
    -p max_fps:=15.0 \
    -p max_age_ms:=120 \
    -p output_encoding:="bgr8" \
    -p swap_rb:=true

Lihat gambar (Reliable, anti QoS error):
  ros2 run image_tools showimage --ros-args -r image:=/camera/image_raw_reliable

Toggle warna TANPA restart:
  ros2 node list
  ros2 param set /camera_hp swap_rb true
  ros2 param set /camera_hp swap_rb false
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def _rotate(frame: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


@dataclass
class FramePacket:
    frame: np.ndarray
    stamp_ros: rclpy.time.Time
    t_wall: float


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")

        # ---------------- Params ----------------
        self.declare_parameter("source", "url")          # url | device | pipeline
        self.declare_parameter("backend", "gstreamer")   # gstreamer | opencv

        self.declare_parameter("url", "0")               # rtsp://... | http://.../video | "0"
        self.declare_parameter("device_index", 0)
        self.declare_parameter("pipeline", "")

        # Output topics
        self.declare_parameter("publish_best_effort", True)
        self.declare_parameter("publish_reliable", True)
        self.declare_parameter("topic_best_effort", "/camera/image_raw")
        self.declare_parameter("topic_reliable", "/camera/image_raw_reliable")
        self.declare_parameter("frame_id", "camera")

        # Rate / latency controls
        self.declare_parameter("publish_in_reader", True)  # true = publish langsung di reader thread (paling low latency)
        self.declare_parameter("max_fps", 15.0)            # limit publish/capture rate
        self.declare_parameter("max_age_ms", 120)          # drop frame jika lebih tua dari ini (anti delay numpuk)
        self.declare_parameter("grab_skip", 0)             # buang N frame sebelum read (untuk opencv backend)

        # Transform
        self.declare_parameter("rotate", 0)                # 0/90/180/270
        self.declare_parameter("flip_h", False)
        self.declare_parameter("flip_v", False)
        self.declare_parameter("resize_width", 0)
        self.declare_parameter("resize_height", 0)

        # Color / encoding
        self.declare_parameter("output_encoding", "bgr8")  # bgr8 | rgb8
        self.declare_parameter("swap_rb", False)

        # Reconnect
        self.declare_parameter("reconnect_sec", 1.0)

        # RTSP tuning
        self.declare_parameter("gstreamer_latency_ms", 0)
        self.declare_parameter("rtsp_tcp", True)
        self.declare_parameter("prefer_h264_pipeline", True)  # explicit depay/parse/decode (lebih stabil)

        # Logs
        self.declare_parameter("log_stats_sec", 2.0)

        # ---------------- Load params ----------------
        self._load_params(first=True)

        # ---------------- QoS profiles ----------------
        self.qos_best_effort = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.qos_reliable = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.bridge = CvBridge()

        self.pub_be: Optional[rclpy.publisher.Publisher] = None
        self.pub_rel: Optional[rclpy.publisher.Publisher] = None

        if self.publish_best_effort:
            self.pub_be = self.create_publisher(Image, self.topic_best_effort, self.qos_best_effort)
        if self.publish_reliable:
            self.pub_rel = self.create_publisher(Image, self.topic_reliable, self.qos_reliable)

        # ---------------- Runtime state ----------------
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()

        self._latest: Optional[FramePacket] = None
        self._latest_lock = threading.Lock()

        self._stop = threading.Event()
        self._need_reopen = threading.Event()

        # stats
        self._t_last_log = time.time()
        self._cnt_cap = 0
        self._cnt_pub = 0
        self._t0 = time.time()

        self.add_on_set_parameters_callback(self._on_params)

        self.get_logger().info(
            f"camera_node start | source={self.source} backend={self.backend} "
            f"| url={self.url if self.source=='url' else ''} dev={self.device_index if self.source=='device' else ''} "
            f"| publish_in_reader={self.publish_in_reader} max_fps={self.max_fps} max_age_ms={self.max_age_ms} "
            f"| encoding={self.output_encoding} swap_rb={self.swap_rb} "
            f"| topics: BE={self.topic_best_effort if self.publish_best_effort else '-'} "
            f"REL={self.topic_reliable if self.publish_reliable else '-'}"
        )

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        # kalau tidak publish_in_reader, pakai timer publish
        self._pub_timer = None
        if not self.publish_in_reader:
            period = 1.0 / max(1.0, float(self.max_fps))
            self._pub_timer = self.create_timer(period, self._publish_tick)

    # ---------------- Parameter helpers ----------------
    def _load_params(self, first: bool = False) -> None:
        self.source = str(self.get_parameter("source").value).strip().lower()
        self.backend = str(self.get_parameter("backend").value).strip().lower()

        self.url = str(self.get_parameter("url").value).strip()
        self.device_index = int(self.get_parameter("device_index").value)
        self.pipeline = str(self.get_parameter("pipeline").value).strip()

        self.publish_best_effort = bool(self.get_parameter("publish_best_effort").value)
        self.publish_reliable = bool(self.get_parameter("publish_reliable").value)
        self.topic_best_effort = str(self.get_parameter("topic_best_effort").value)
        self.topic_reliable = str(self.get_parameter("topic_reliable").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.publish_in_reader = bool(self.get_parameter("publish_in_reader").value)
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.max_age_ms = int(self.get_parameter("max_age_ms").value)
        self.grab_skip = int(self.get_parameter("grab_skip").value)

        self.rotate_deg = int(self.get_parameter("rotate").value)
        if self.rotate_deg not in (0, 90, 180, 270):
            self.rotate_deg = 0
        self.flip_h = bool(self.get_parameter("flip_h").value)
        self.flip_v = bool(self.get_parameter("flip_v").value)
        self.resize_w = int(self.get_parameter("resize_width").value)
        self.resize_h = int(self.get_parameter("resize_height").value)

        self.output_encoding = str(self.get_parameter("output_encoding").value).strip().lower()
        if self.output_encoding not in ("bgr8", "rgb8"):
            self.output_encoding = "bgr8"
        self.swap_rb = bool(self.get_parameter("swap_rb").value)

        self.reconnect_sec = float(self.get_parameter("reconnect_sec").value)

        self.gst_latency_ms = int(self.get_parameter("gstreamer_latency_ms").value)
        self.rtsp_tcp = bool(self.get_parameter("rtsp_tcp").value)
        self.prefer_h264_pipeline = bool(self.get_parameter("prefer_h264_pipeline").value)

        self.log_stats_sec = float(self.get_parameter("log_stats_sec").value)

        if first:
            return

    def _on_params(self, params: list[Parameter]) -> SetParametersResult:
        reopen_needed = False

        for p in params:
            name = p.name
            if name == "swap_rb":
                self.swap_rb = bool(p.value)
            elif name == "output_encoding":
                v = str(p.value).strip().lower()
                if v in ("bgr8", "rgb8"):
                    self.output_encoding = v
            elif name in ("rotate", "flip_h", "flip_v", "resize_width", "resize_height"):
                # reload simple transform params
                self.rotate_deg = int(self.get_parameter("rotate").value)
                if self.rotate_deg not in (0, 90, 180, 270):
                    self.rotate_deg = 0
                self.flip_h = bool(self.get_parameter("flip_h").value)
                self.flip_v = bool(self.get_parameter("flip_v").value)
                self.resize_w = int(self.get_parameter("resize_width").value)
                self.resize_h = int(self.get_parameter("resize_height").value)
            elif name in ("max_age_ms", "max_fps", "grab_skip"):
                self.max_age_ms = int(self.get_parameter("max_age_ms").value)
                self.max_fps = float(self.get_parameter("max_fps").value)
                self.grab_skip = int(self.get_parameter("grab_skip").value)
            elif name in ("url", "device_index", "pipeline", "source", "backend", "gstreamer_latency_ms", "rtsp_tcp", "prefer_h264_pipeline"):
                reopen_needed = True

        if reopen_needed:
            self._need_reopen.set()

        return SetParametersResult(successful=True)

    # ---------------- GStreamer pipelines ----------------
    def _gst_http_mjpeg(self, url: str) -> str:
        return (
            f"souphttpsrc location={url} is-live=true do-timestamp=true ! "
            f"queue leaky=downstream max-size-buffers=2 ! "
            f"multipartdemux ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    def _gst_rtsp(self, url: str) -> str:
        proto = "tcp" if self.rtsp_tcp else "udp"
        lat = max(0, int(self.gst_latency_ms))

        # explicit H264 pipeline (lebih predictable)
        if self.prefer_h264_pipeline:
            return (
                f"rtspsrc location={url} protocols={proto} latency={lat} drop-on-latency=true ! "
                f"rtpjitterbuffer drop-on-latency=true latency={lat} ! "
                f"rtph264depay ! h264parse ! decodebin ! "
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink drop=true max-buffers=1 sync=false"
            )

        # fallback generic decodebin
        return (
            f"rtspsrc location={url} protocols={proto} latency={lat} drop-on-latency=true ! "
            f"rtpjitterbuffer drop-on-latency=true latency={lat} ! "
            f"decodebin ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    # ---------------- Capture open/close ----------------
    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        # pipeline override
        if self.source == "pipeline" and self.pipeline:
            if self.backend == "gstreamer":
                cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
            else:
                cap = cv2.VideoCapture(self.pipeline)
            return cap if cap.isOpened() else None

        # device
        if self.source == "device":
            cap = cv2.VideoCapture(int(self.device_index))
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
            return None

        # url
        if self.source == "url":
            u = self.url
            # support: url="0" as quick device
            if _is_int(u):
                cap = cv2.VideoCapture(int(u))
                if cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    return cap
                return None

            if self.backend == "gstreamer":
                if u.startswith("rtsp://"):
                    pipe = self._gst_rtsp(u)
                    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                    return cap if cap.isOpened() else None
                if u.startswith("http://") or u.startswith("https://"):
                    pipe = self._gst_http_mjpeg(u)
                    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                    return cap if cap.isOpened() else None

            # fallback OpenCV
            cap = cv2.VideoCapture(u)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap

        return None

    def _close_capture(self) -> None:
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = None

    # ---------------- Frame processing ----------------
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        # normalize channels
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # rotate
        frame = _rotate(frame, self.rotate_deg)

        # flip
        if self.flip_h and self.flip_v:
            frame = cv2.flip(frame, -1)
        elif self.flip_h:
            frame = cv2.flip(frame, 1)
        elif self.flip_v:
            frame = cv2.flip(frame, 0)

        # resize
        if self.resize_w > 0 and self.resize_h > 0:
            frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        # optional swap rb (buat kasus "biru/ungu")
        if self.swap_rb:
            frame = frame[:, :, ::-1]

        # output encoding conversion
        if self.output_encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    # ---------------- Publish ----------------
    def _publish_frame(self, pkt: FramePacket) -> None:
        age_ms = (time.time() - pkt.t_wall) * 1000.0
        if self.max_age_ms > 0 and age_ms > float(self.max_age_ms):
            return  # drop frame tua

        header = Header()
        header.stamp = pkt.stamp_ros.to_msg()
        header.frame_id = self.frame_id

        msg = self.bridge.cv2_to_imgmsg(pkt.frame, encoding=self.output_encoding)
        msg.header = header

        if self.pub_be is not None:
            self.pub_be.publish(msg)
        if self.pub_rel is not None:
            self.pub_rel.publish(msg)

        self._cnt_pub += 1

    # ---------------- Reader loop ----------------
    def _reader_loop(self) -> None:
        last_open_warn = 0.0
        last_pub_t = 0.0

        while not self._stop.is_set():
            if self._need_reopen.is_set():
                self._need_reopen.clear()
                self._load_params()
                self._close_capture()

            with self._cap_lock:
                cap = self._cap

            if cap is None or not cap.isOpened():
                cap_try = self._open_capture()
                with self._cap_lock:
                    self._cap = cap_try
                    cap = cap_try

                if cap is None or not cap.isOpened():
                    now = time.time()
                    if now - last_open_warn > 2.0:
                        self.get_logger().warn(f"Gagal buka stream. retry {self.reconnect_sec:.1f}s | source={self.source} url={self.url}")
                        last_open_warn = now
                    time.sleep(max(0.1, self.reconnect_sec))
                    continue
                else:
                    self.get_logger().info("Capture opened")

            # optional grab_skip (berguna utk backend opencv)
            for _ in range(max(0, int(self.grab_skip))):
                try:
                    cap.grab()
                except Exception:
                    break

            ok, frame = cap.read()
            if not ok or frame is None:
                self._close_capture()
                time.sleep(max(0.05, self.reconnect_sec))
                continue

            self._cnt_cap += 1

            # process
            out = self._process_frame(frame)
            pkt = FramePacket(frame=out, stamp_ros=self.get_clock().now(), t_wall=time.time())

            with self._latest_lock:
                self._latest = pkt

            # publish in reader (low latency) with max_fps limiter
            if self.publish_in_reader:
                now = time.time()
                min_dt = 1.0 / max(1.0, float(self.max_fps))
                if now - last_pub_t >= min_dt:
                    self._publish_frame(pkt)
                    last_pub_t = now

            self._log_stats()

        self._close_capture()

    def _publish_tick(self) -> None:
        with self._latest_lock:
            pkt = self._latest
        if pkt is None:
            self._log_stats()
            return
        self._publish_frame(pkt)
        self._log_stats(pkt)

    # ---------------- Logging ----------------
    def _log_stats(self, pkt: Optional[FramePacket] = None) -> None:
        now = time.time()
        if now - self._t_last_log < max(0.5, float(self.log_stats_sec)):
            return
        self._t_last_log = now

        dt = now - self._t0
        cap_fps = (self._cnt_cap / dt) if dt > 0 else 0.0
        pub_fps = (self._cnt_pub / dt) if dt > 0 else 0.0

        age_ms = 0.0
        if pkt is not None:
            age_ms = (now - pkt.t_wall) * 1000.0

        self.get_logger().info(
            f"stats | cap_fps={cap_fps:.1f} pub_fps={pub_fps:.1f} "
            f"| age={age_ms:.0f}ms | enc={self.output_encoding} swap_rb={self.swap_rb} "
            f"| rtsp_tcp={self.rtsp_tcp} gst_lat={self.gst_latency_ms}ms"
        )

        # reset window
        self._t0 = now
        self._cnt_cap = 0
        self._cnt_pub = 0

    # ---------------- Shutdown ----------------
    def destroy_node(self) -> bool:
        self._stop.set()
        try:
            if self._reader.is_alive():
                self._reader.join(timeout=1.0)
        except Exception:
            pass
        self._close_capture()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
