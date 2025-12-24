#!/usr/bin/env python3
import time
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # Parameters
        self.declare_parameter("url", "http://127.0.0.1:8080/video")  # ganti sesuai IP Webcam
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("resize_width", 0)   # 0 = no resize
        self.declare_parameter("resize_height", 0)  # 0 = no resize

        self.url = self.get_parameter("url").get_parameter_value().string_value
        self.fps = float(self.get_parameter("fps").value)
        self.resize_w = int(self.get_parameter("resize_width").value)
        self.resize_h = int(self.get_parameter("resize_height").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)

        self.cap = None
        self.last_open_try = 0.0
        self.open_camera()

        period = 1.0 / max(self.fps, 1e-3)
        self.timer = self.create_timer(period, self.timer_cb)

        self.get_logger().info(f"camera_node started | url={self.url} | fps={self.fps}")

    def open_camera(self):
        now = time.time()
        if now - self.last_open_try < 2.0:
            return
        self.last_open_try = now

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.get_logger().info(f"Opening stream: {self.url}")
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            self.get_logger().warn("Failed to open stream. Will retry...")
            return

        self.cap = cap
        self.get_logger().info("Stream opened.")

    def timer_cb(self):
        if self.cap is None or not self.cap.isOpened():
            self.open_camera()
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warn("Frame read failed. Reopening...")
            self.open_camera()
            return

        if self.resize_w > 0 and self.resize_h > 0:
            frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
