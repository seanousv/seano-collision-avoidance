#!/usr/bin/env python3
import argparse
import os
import time
from typing import Optional

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoFilePublisher(Node):
    def __init__(
        self,
        video_path: str,
        raw_topic: str,
        reliable_topic: str,
        fps: float,
        duration_s: float,
        loop: bool,
        frame_id: str,
    ) -> None:
        super().__init__("video_file_pub_node")

        self.video_path = video_path
        self.raw_topic = raw_topic
        self.reliable_topic = reliable_topic
        self.target_fps = fps
        self.duration_s = duration_s
        self.loop = loop
        self.frame_id = frame_id

        self.bridge = CvBridge()
        self.pub_raw = self.create_publisher(Image, self.raw_topic, 10)
        self.pub_reliable = self.create_publisher(Image, self.reliable_topic, 10)

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.loop_count = 0
        self.start_wall = time.time()
        self.last_log_wall = self.start_wall
        self.last_log_frame = 0

        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video tidak ditemukan: {self.video_path}")

        self._open_capture()

        period = 1.0 / max(self.target_fps, 0.1)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"video_file_pub_node start | video={self.video_path} "
            f"| raw_topic={self.raw_topic} "
            f"| reliable_topic={self.reliable_topic} "
            f"| fps={self.target_fps:.2f} "
            f"| duration_s={self.duration_s:.1f} "
            f"| loop={self.loop}"
        )

    def _open_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Gagal membuka video: {self.video_path}")

    def _rewind(self) -> bool:
        assert self.cap is not None
        ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ok:
            self._open_capture()

        ret, _ = self.cap.read()
        if not ret:
            return False

        # balik lagi ke frame pertama sesungguhnya
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.loop_count += 1
        self.get_logger().info(f"EOF tercapai, rewind video. loop_count={self.loop_count}")
        return True

    def _should_stop(self) -> bool:
        if self.duration_s <= 0:
            return False
        return (time.time() - self.start_wall) >= self.duration_s

    def _tick(self) -> None:
        if self._should_stop():
            elapsed = time.time() - self.start_wall
            fps_actual = self.frame_count / elapsed if elapsed > 0 else 0.0
            self.get_logger().info(
                f"Stop by duration | elapsed={elapsed:.2f}s "
                f"| frames={self.frame_count} "
                f"| fps_actual={fps_actual:.2f} "
                f"| loops={self.loop_count}"
            )
            self._shutdown()
            return

        assert self.cap is not None
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                ok = self._rewind()
                if not ok:
                    self.get_logger().error("Gagal rewind video setelah EOF.")
                    self._shutdown()
                    return
                ret, frame = self.cap.read()

            if not ret:
                self.get_logger().info("EOF video tercapai, node berhenti.")
                self._shutdown()
                return

        if frame is None:
            self.get_logger().warn("Frame kosong diterima, skip.")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        self.pub_raw.publish(msg)
        self.pub_reliable.publish(msg)
        self.frame_count += 1

        now = time.time()
        if now - self.last_log_wall >= 2.0:
            delta_t = now - self.last_log_wall
            delta_f = self.frame_count - self.last_log_frame
            fps_window = delta_f / delta_t if delta_t > 0 else 0.0
            self.get_logger().info(
                f"publishing | frames={self.frame_count} "
                f"| fps_window={fps_window:.2f} "
                f"| loops={self.loop_count}"
            )
            self.last_log_wall = now
            self.last_log_frame = self.frame_count

    def _shutdown(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.destroy_node()
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish video file to ROS 2 image topics.")
    parser.add_argument(
        "--video",
        required=True,
        help="Path video file, mis. /home/seano/.../videodemo.mp4",
    )
    parser.add_argument(
        "--raw-topic",
        default="/seano/camera/image_raw",
        help="Topic raw image",
    )
    parser.add_argument(
        "--reliable-topic",
        default="/seano/camera/image_raw_reliable",
        help="Topic reliable image",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="FPS publish target",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=45.0,
        help="Durasi publish per run dalam detik. <=0 artinya tanpa batas.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop video saat EOF tercapai.",
    )
    parser.add_argument(
        "--frame-id",
        default="camera_link",
        help="Frame ID ROS image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()

    node = VideoFilePublisher(
        video_path=args.video,
        raw_topic=args.raw_topic,
        reliable_topic=args.reliable_topic,
        fps=args.fps,
        duration_s=args.duration,
        loop=args.loop,
        frame_id=args.frame_id,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._shutdown()


if __name__ == "__main__":
    main()