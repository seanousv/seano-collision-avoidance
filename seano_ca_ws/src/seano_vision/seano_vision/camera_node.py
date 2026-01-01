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
from rcl_interfaces.msg import SetParametersResult

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
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
    t_mono: float  # monotonic timestamp (stable for age/rate)


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
        self.declare_parameter("max_fps", 15.0)            # limit publish rate; <=0 berarti unlimited
        self.declare_parameter("max_age_ms", 120)          # drop frame jika lebih tua dari ini (anti delay numpuk); <=0 disable
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

        self.pub_be = self.create_publisher(Image, self.topic_best_effort, self.qos_best_effort) if self.publish_best_effort else None
        self.pub_rel = self.create_publisher(Image, self.topic_reliable, self.qos_reliable) if self.publish_reliable else None

        # ---------------- Runtime state ----------------
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()

        self._latest: Optional[FramePacket] = None
        self._latest_lock = threading.Lock()

        self._stop = threading.Event()
        self._need_reopen = threading.Event()
        self._need_timer_reconfig = threading.Event()

        # stats (monotonic)
        self._t0 = time.monotonic()
        self._t_last_log = self._t0
        self._cnt_cap = 0
        self._cnt_pub = 0

        # anti duplicate publish (timer mode)
        self._last_sent_pkt_tmono: float = -1.0

        self.add_on_set_parameters_callback(self._on_params)

        self.get_logger().info(
            "camera_node start | "
            f"source={self.source} backend={self.backend} "
            f"| url={self.url if self.source=='url' else ''} dev={self.device_index if self.source=='device' else ''} "
            f"| publish_in_reader={self.publish_in_reader} max_fps={self.max_fps} max_age_ms={self.max_age_ms} "
            f"| encoding={self.output_encoding} swap_rb={self.swap_rb} "
            f"| topics: BE={self.topic_best_effort if self.publish_best_effort else '-'} "
            f"REL={self.topic_reliable if self.publish_reliable else '-'}"
        )

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        # publish timer (jika tidak publish_in_reader)
        self._pub_timer = None
        self._setup_publish_timer()

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
        timer_reconfig_needed = False

        for p in params:
            name = p.name

            # Live params (no reopen)
            if name == "swap_rb":
                self.swap_rb = bool(p.value)

            elif name == "output_encoding":
                v = str(p.value).strip().lower()
                if v in ("bgr8", "rgb8"):
                    self.output_encoding = v

            elif name == "rotate":
                v = int(p.value)
                self.rotate_deg = v if v in (0, 90, 180, 270) else 0
            elif name == "flip_h":
                self.flip_h = bool(p.value)
            elif name == "flip_v":
                self.flip_v = bool(p.value)
            elif name == "resize_width":
                self.resize_w = int(p.value)
            elif name == "resize_height":
                self.resize_h = int(p.value)

            elif name == "max_age_ms":
                self.max_age_ms = int(p.value)

            elif name == "max_fps":
                self.max_fps = float(p.value)
                timer_reconfig_needed = True

            elif name == "grab_skip":
                self.grab_skip = int(p.value)

            elif name == "publish_in_reader":
                self.publish_in_reader = bool(p.value)
                timer_reconfig_needed = True

            # Capture reopen params
            elif name in (
                "url", "device_index", "pipeline", "source", "backend",
                "gstreamer_latency_ms", "rtsp_tcp", "prefer_h264_pipeline",
            ):
                reopen_needed = True

            elif name == "log_stats_sec":
                self.log_stats_sec = float(p.value)

            elif name in ("publish_best_effort", "publish_reliable", "topic_best_effort", "topic_reliable"):
                # ini terkait publisher; untuk aman, minta restart node bila berubah.
                # (kita set reopen_needed supaya kamu sadar perlu restart untuk update publisher)
                reopen_needed = True

        if reopen_needed:
            self._need_reopen.set()

        if timer_reconfig_needed:
            self._need_timer_reconfig.set()

        return SetParametersResult(successful=True)

    def _setup_publish_timer(self) -> None:
        # Timer dipakai hanya jika publish_in_reader == False
        if self._pub_timer is not None:
            try:
                self._pub_timer.cancel()
            except Exception:
                pass
            self._pub_timer = None

        if self.publish_in_reader:
            return

        # Period: jika max_fps > 0 => 1/max_fps
        # jika max_fps <= 0 => polling aman (50Hz) tapi publish hanya kalau ada frame baru
        if self.max_fps and self.max_fps > 0.0:
            period = 1.0 / max(0.01, float(self.max_fps))
        else:
            period = 0.02  # 50Hz polling
        period = max(0.001, period)

        self._pub_timer = self.create_timer(period, self._publish_tick)

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

        if self.prefer_h264_pipeline:
            return (
                f"rtspsrc location={url} protocols={proto} latency={lat} drop-on-latency=true ! "
                f"rtpjitterbuffer drop-on-latency=true latency={lat} ! "
                f"rtph264depay ! h264parse ! decodebin ! "
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink drop=true max-buffers=1 sync=false"
            )

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
        # drop frame tua
        if self.max_age_ms and self.max_age_ms > 0:
            age_ms = (time.monotonic() - pkt.t_mono) * 1000.0
            if age_ms > float(self.max_age_ms):
                return

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
        self._last_sent_pkt_tmono = pkt.t_mono

    # ---------------- Reader loop ----------------
    def _reader_loop(self) -> None:
        last_open_warn_mono = 0.0
        last_pub_mono = 0.0

        while not self._stop.is_set():
            # Reconfigure timer if needed (changes to publish_in_reader/max_fps)
            if self._need_timer_reconfig.is_set():
                self._need_timer_reconfig.clear()
                try:
                    # timer reconfig must run in node context; calling here is ok,
                    # but we keep it minimal and safe.
                    self._setup_publish_timer()
                except Exception:
                    pass

            # Reopen capture if needed
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
                    now_mono = time.monotonic()
                    if now_mono - last_open_warn_mono > 2.0:
                        self.get_logger().warn(
                            f"Gagal buka stream. retry {self.reconnect_sec:.1f}s | source={self.source} url={self.url}"
                        )
                        last_open_warn_mono = now_mono
                    time.sleep(max(0.1, float(self.reconnect_sec)))
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
                time.sleep(max(0.05, float(self.reconnect_sec)))
                continue

            self._cnt_cap += 1

            # process
            out = self._process_frame(frame)
            pkt = FramePacket(frame=out, stamp_ros=self.get_clock().now(), t_mono=time.monotonic())

            with self._latest_lock:
                self._latest = pkt

            # publish in reader (low latency) with max_fps limiter
            if self.publish_in_reader:
                max_fps = float(self.max_fps)
                if max_fps > 0.0:
                    min_dt = 1.0 / max(0.01, max_fps)
                else:
                    min_dt = 0.0  # unlimited

                now_mono = time.monotonic()
                if (min_dt <= 0.0) or ((now_mono - last_pub_mono) >= min_dt):
                    self._publish_frame(pkt)
                    last_pub_mono = now_mono

            self._log_stats(pkt)

        self._close_capture()

    def _publish_tick(self) -> None:
        # timer mode: publish only if ada frame baru (anti spam frame yang sama)
        with self._latest_lock:
            pkt = self._latest

        if pkt is None:
            self._log_stats(None)
            return

        if pkt.t_mono == self._last_sent_pkt_tmono:
            self._log_stats(pkt)
            return

        self._publish_frame(pkt)
        self._log_stats(pkt)

    # ---------------- Logging ----------------
    def _log_stats(self, pkt: Optional[FramePacket]) -> None:
        now_mono = time.monotonic()
        if now_mono - self._t_last_log < max(0.5, float(self.log_stats_sec)):
            return
        self._t_last_log = now_mono

        dt = now_mono - self._t0
        cap_fps = (self._cnt_cap / dt) if dt > 1e-9 else 0.0
        pub_fps = (self._cnt_pub / dt) if dt > 1e-9 else 0.0

        age_ms = 0.0
        if pkt is not None:
            age_ms = (now_mono - pkt.t_mono) * 1000.0

        self.get_logger().info(
            f"stats | cap_fps={cap_fps:.1f} pub_fps={pub_fps:.1f} "
            f"| age={age_ms:.0f}ms | enc={self.output_encoding} swap_rb={self.swap_rb} "
            f"| publish_in_reader={self.publish_in_reader} max_fps={self.max_fps} "
            f"| rtsp_tcp={self.rtsp_tcp} gst_lat={self.gst_latency_ms}ms"
        )

        # reset window
        self._t0 = now_mono
        self._cnt_cap = 0
        self._cnt_pub = 0

    # ---------------- Shutdown ----------------
    def destroy_node(self) -> bool:
        self._stop.set()
        try:
            if self._pub_timer is not None:
                self._pub_timer.cancel()
        except Exception:
            pass

        try:
            if self._reader.is_alive():
                self._reader.join(timeout=1.2)
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