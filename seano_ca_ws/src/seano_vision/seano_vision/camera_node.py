#!/usr/bin/env python3
"""
camera_node.py (ROS 2 Humble) - SEANO Vision (USB/RTSP ready)

Perbaikan utama untuk USB webcam (WSL/Jetson):
- Device mode memakai cv2.CAP_V4L2 (lebih stabil daripada auto/gstreamer di banyak kasus)
- Bisa set FOURCC (default MJPG), width/height/fps
- Ada warning jelas kalau cap.read() gagal terus (sebelumnya “silent retry”)
- Tetap kompatibel dengan parameter lama (source/backend/url/device_index/pipeline/...)
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header


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


def _fourcc_from_str(s: str) -> Optional[int]:
    s = (s or "").strip().upper()
    if len(s) != 4:
        return None
    try:
        return cv2.VideoWriter_fourcc(s[0], s[1], s[2], s[3])
    except Exception:
        return None


@dataclass
class FramePacket:
    frame: np.ndarray
    stamp_ros: rclpy.time.Time
    t_mono: float


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")

        # ---------------- Params (lama) ----------------
        self.declare_parameter("source", "url")  # url | device | pipeline
        self.declare_parameter("backend", "gstreamer")  # gstreamer | opencv
        self.declare_parameter("url", "0")  # rtsp://... | http://.../video | "0"
        self.declare_parameter("device_index", 0)
        self.declare_parameter("pipeline", "")

        # Output topics
        self.declare_parameter("publish_best_effort", True)
        self.declare_parameter("publish_reliable", True)
        self.declare_parameter("topic_best_effort", "/camera/image_raw")
        self.declare_parameter("topic_reliable", "/camera/image_raw_reliable")
        self.declare_parameter("frame_id", "camera")

        # Latency / rate
        self.declare_parameter("publish_in_reader", True)
        self.declare_parameter("max_fps", 15.0)
        self.declare_parameter("max_age_ms", 120)
        self.declare_parameter("grab_skip", 0)

        # Transforms
        self.declare_parameter("rotate", 0)  # 0/90/180/270
        self.declare_parameter("flip_h", False)
        self.declare_parameter("flip_v", False)
        self.declare_parameter("resize_width", 0)
        self.declare_parameter("resize_height", 0)
        self.declare_parameter("output_encoding", "bgr8")  # bgr8|rgb8
        self.declare_parameter("swap_rb", False)

        # Reconnect
        self.declare_parameter("reconnect_sec", 0.5)

        # GStreamer params (lama)
        self.declare_parameter("gstreamer_latency_ms", 80)
        self.declare_parameter("rtsp_tcp", True)
        self.declare_parameter("prefer_h264_pipeline", True)

        # Logs
        self.declare_parameter("log_stats_sec", 2.0)

        # ---------------- Params (baru, untuk USB webcam) ----------------
        # Opsional: pakai salah satu: device_path ATAU device_index
        self.declare_parameter("device_path", "")  # contoh: "/dev/video0"
        self.declare_parameter("device_width", 0)  # contoh: 1280
        self.declare_parameter("device_height", 0)  # contoh: 720
        self.declare_parameter("device_fps", 0)  # contoh: 30
        self.declare_parameter("device_fourcc", "MJPG")  # MJPG/YUYV/...

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

        self.pub_be = (
            self.create_publisher(Image, self.topic_best_effort, self.qos_best_effort)
            if self.publish_best_effort
            else None
        )
        self.pub_rel = (
            self.create_publisher(Image, self.topic_reliable, self.qos_reliable)
            if self.publish_reliable
            else None
        )

        # ---------------- State ----------------
        self._cap_lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None

        self._latest_lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._last_sent_pkt_tmono: float = -1.0

        self._stop = threading.Event()
        self._need_reopen = threading.Event()

        self._pub_timer = None

        # stats window
        self._t0 = time.monotonic()
        self._cnt_cap = 0
        self._cnt_pub = 0
        self._last_stats_log = time.monotonic()

        # read-fail diagnostics
        self._consecutive_read_fail = 0
        self._last_read_fail_warn = time.monotonic()

        self.add_on_set_parameters_callback(self._on_params)

        self.get_logger().info(
            "camera_node start | "
            f"source={self.source} backend={self.backend} "
            f"| url={self.url if self.source=='url' else ''}"
            f" dev={self.device_index if self.source=='device' else ''} "
            f"| publish_in_reader={self.publish_in_reader} "
            f"max_fps={self.max_fps} max_age_ms={self.max_age_ms} "
            f"| encoding={self.output_encoding} swap_rb={self.swap_rb} "
            f"| topics: BE={self.topic_best_effort if self.publish_best_effort else '-'} "
            f"REL={self.topic_reliable if self.publish_reliable else '-'}"
        )

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        # timer publish kalau publish_in_reader False
        self._setup_publish_timer()

        # trigger open pertama
        self._need_reopen.set()

    # ---------------- Param handling ----------------
    def _load_params(self, first: bool = False) -> None:
        self.source = str(self.get_parameter("source").value).strip().lower()
        if self.source not in ("url", "device", "pipeline"):
            self.source = "url"

        self.backend = str(self.get_parameter("backend").value).strip().lower()
        if self.backend not in ("gstreamer", "opencv"):
            self.backend = "gstreamer"

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

        # USB extras
        self.device_path = str(self.get_parameter("device_path").value).strip()
        self.device_width = int(self.get_parameter("device_width").value)
        self.device_height = int(self.get_parameter("device_height").value)
        self.device_fps = int(self.get_parameter("device_fps").value)
        self.device_fourcc = str(self.get_parameter("device_fourcc").value).strip().upper()

        if first:
            return

    def _on_params(self, params) -> SetParametersResult:
        reopen_needed = False
        timer_reconfig_needed = False

        for p in params:
            name = p.name
            if name in (
                "max_fps",
                "publish_in_reader",
            ):
                timer_reconfig_needed = True

            if name in (
                "url",
                "device_index",
                "device_path",
                "device_width",
                "device_height",
                "device_fps",
                "device_fourcc",
                "pipeline",
                "source",
                "backend",
                "gstreamer_latency_ms",
                "rtsp_tcp",
                "prefer_h264_pipeline",
            ):
                reopen_needed = True

        # Apply values
        for p in params:
            try:
                self.set_parameters([Parameter(p.name, p.type_, p.value)])
            except Exception:
                pass

        # Reload from node parameters
        self._load_params()

        if timer_reconfig_needed:
            try:
                self._setup_publish_timer()
            except Exception:
                pass

        if reopen_needed:
            self._need_reopen.set()

        return SetParametersResult(successful=True)

    # ---------------- Timer (publish mode) ----------------
    def _setup_publish_timer(self) -> None:
        if self._pub_timer is not None:
            try:
                self._pub_timer.cancel()
            except Exception:
                pass
            self._pub_timer = None

        if self.publish_in_reader:
            return

        if self.max_fps and self.max_fps > 0.0:
            period = 1.0 / max(0.01, float(self.max_fps))
        else:
            period = 0.02  # 50Hz polling

        period = max(0.001, period)
        self._pub_timer = self.create_timer(period, self._publish_tick)

    # ---------------- GStreamer pipelines ----------------
    def _gst_http_mjpeg(self, url: str) -> str:
        return (
            f"souphttpsrc location={url} is-live=true ! "
            f"multipartdemux ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    def _gst_rtsp(self, url: str) -> str:
        lat = max(0, int(self.gst_latency_ms))
        proto = "tcp" if self.rtsp_tcp else "udp"
        if self.prefer_h264_pipeline:
            return (
                f"rtspsrc location={url} protocols={proto} latency={lat} drop-on-latency=true ! "
                f"rtpjitterbuffer drop-on-latency=true latency={lat} ! "
                f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
                f"appsink drop=true max-buffers=1 sync=false"
            )
        return (
            f"rtspsrc location={url} protocols={proto} latency={lat} drop-on-latency=true ! "
            f"rtpjitterbuffer drop-on-latency=true latency={lat} ! "
            f"decodebin ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    def _gst_v4l2_device(self, dev: str) -> str:
        # MJPG is common for USB webcams; decode to BGR
        w = int(self.device_width) if self.device_width > 0 else 0
        h = int(self.device_height) if self.device_height > 0 else 0
        fps = int(self.device_fps) if self.device_fps > 0 else 0
        caps = []
        # Jika kamu mau paksa MJPG:
        # - banyak webcam default-nya MJPG di resolusi tinggi
        caps.append("image/jpeg")
        if w > 0 and h > 0:
            caps.append(f"width={w}")
            caps.append(f"height={h}")
        if fps > 0:
            caps.append(f"framerate={fps}/1")
        caps_str = ",".join(caps)

        return (
            f"v4l2src device={dev} ! {caps_str} ! "
            f"jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    # ---------------- Capture open/close ----------------
    def _apply_device_settings(self, cap: cv2.VideoCapture) -> None:
        # buffer kecil untuk latency
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        fourcc = _fourcc_from_str(self.device_fourcc)
        if fourcc is not None:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass

        if self.device_width > 0:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.device_width))
            except Exception:
                pass

        if self.device_height > 0:
            try:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.device_height))
            except Exception:
                pass

        if self.device_fps > 0:
            try:
                cap.set(cv2.CAP_PROP_FPS, float(self.device_fps))
            except Exception:
                pass

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        # pipeline override
        if self.source == "pipeline" and self.pipeline:
            if self.backend == "gstreamer":
                cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
            else:
                cap = cv2.VideoCapture(self.pipeline)
            return cap if cap.isOpened() else None

        # device (USB webcam)
        if self.source == "device":
            dev = self.device_path if self.device_path else int(self.device_index)

            # kalau backend=gstreamer, boleh pakai v4l2 pipeline
            if self.backend == "gstreamer":
                dev_str = dev if isinstance(dev, str) else f"/dev/video{int(dev)}"
                pipe = self._gst_v4l2_device(dev_str)
                cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap
                # fallback ke V4L2 OpenCV
                # (biar tetap jalan walau gst kurang lengkap)
            # backend opencv (dipaksa CAP_V4L2)
            try:
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            except Exception:
                cap = cv2.VideoCapture(dev)

            if cap.isOpened():
                self._apply_device_settings(cap)
                return cap

            # fallback terakhir
            try:
                cap2 = cv2.VideoCapture(dev, cv2.CAP_ANY)
            except Exception:
                cap2 = cv2.VideoCapture(dev)

            if cap2.isOpened():
                self._apply_device_settings(cap2)
                return cap2

            return None

        # url / rtsp / http mjpeg
        u = self.url
        if _is_int(u):
            # kadang user tulis "0" walau source=url
            idx = int(u)
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            except Exception:
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self._apply_device_settings(cap)
                return cap

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

        if self.rotate_deg:
            frame = _rotate(frame, self.rotate_deg)

        if self.flip_h and self.flip_v:
            frame = cv2.flip(frame, -1)
        elif self.flip_h:
            frame = cv2.flip(frame, 1)
        elif self.flip_v:
            frame = cv2.flip(frame, 0)

        if self.resize_w > 0 and self.resize_h > 0:
            frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        if self.swap_rb:
            frame = frame[:, :, ::-1]

        if self.output_encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    # ---------------- Publish ----------------
    def _publish_frame(self, pkt: FramePacket) -> None:
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

    def _publish_tick(self) -> None:
        with self._latest_lock:
            pkt = self._latest

        if pkt is None:
            return

        # anti spam frame yang sama
        if pkt.t_mono <= self._last_sent_pkt_tmono:
            return

        self._publish_frame(pkt)
        self._last_sent_pkt_tmono = pkt.t_mono
        self._log_stats(pkt)

    def _log_stats(self, pkt: Optional[FramePacket]) -> None:
        if self.log_stats_sec <= 0.0:
            return
        now_mono = time.monotonic()
        if (now_mono - self._last_stats_log) < float(self.log_stats_sec):
            return

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

        self._t0 = now_mono
        self._cnt_cap = 0
        self._cnt_pub = 0
        self._last_stats_log = now_mono

    # ---------------- Reader loop ----------------
    def _reader_loop(self) -> None:
        last_open_warn_mono = time.monotonic()
        last_pub_mono = 0.0

        while not self._stop.is_set():
            try:
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
                                f"Gagal buka stream. retry {self.reconnect_sec:.1f}s | "
                                f"source={self.source} url={self.url} dev={self.device_path or self.device_index}"
                            )
                            last_open_warn_mono = now_mono
                        time.sleep(max(0.1, float(self.reconnect_sec)))
                        continue
                    else:
                        self.get_logger().info("Capture opened")
                        self._consecutive_read_fail = 0

                # optional grab_skip
                for _ in range(max(0, int(self.grab_skip))):
                    try:
                        cap.grab()
                    except Exception:
                        break

                ok, frame = cap.read()
                if (not ok) or frame is None:
                    self._consecutive_read_fail += 1
                    now_mono = time.monotonic()

                    # warning kalau read fail terus (ini yang bikin “topic ada tapi gak ada frame”)
                    if (now_mono - self._last_read_fail_warn) > 2.0:
                        self.get_logger().warn(
                            "cap.read() gagal (tidak ada frame). "
                            f"cnt={self._consecutive_read_fail} | "
                            f"dev={self.device_path or self.device_index} | "
                            "Coba: ganti device_index (0/1), atau set device_path=/dev/video0 atau /dev/video1, "
                            "atau paksa device_fourcc=YUYV/MJPG."
                        )
                        self._last_read_fail_warn = now_mono

                    # kalau sudah fail beberapa kali, reopen biar reset driver
                    if self._consecutive_read_fail >= 10:
                        self._need_reopen.set()
                        self._consecutive_read_fail = 0

                    time.sleep(max(0.05, float(self.reconnect_sec)))
                    continue

                self._consecutive_read_fail = 0
                self._cnt_cap += 1

                out = self._process_frame(frame)
                pkt = FramePacket(
                    frame=out,
                    stamp_ros=self.get_clock().now(),
                    t_mono=time.monotonic(),
                )

                with self._latest_lock:
                    self._latest = pkt

                if self.publish_in_reader:
                    max_fps = float(self.max_fps)
                    if max_fps > 0.0:
                        min_dt = 1.0 / max(0.01, max_fps)
                    else:
                        min_dt = 0.0

                    now_mono = time.monotonic()
                    if (min_dt <= 0.0) or ((now_mono - last_pub_mono) >= min_dt):
                        self._publish_frame(pkt)
                        last_pub_mono = now_mono
                        self._log_stats(pkt)

            except Exception as e:
                # jangan sampai thread mati silent
                self.get_logger().error(f"reader_loop exception: {e}")
                self._need_reopen.set()
                time.sleep(0.2)

        self._close_capture()

    # ---------------- Shutdown ----------------
    def destroy_node(self) -> bool:
        self._stop.set()
        try:
            if self._pub_timer is not None:
                self._pub_timer.cancel()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
