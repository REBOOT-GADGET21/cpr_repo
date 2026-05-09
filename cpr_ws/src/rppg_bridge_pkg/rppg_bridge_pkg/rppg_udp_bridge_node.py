#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS2 bridge for rPPG values sent by realtime_rppg_realsense.py over UDP.

This node intentionally publishes only the values that the existing rPPG UI used:
BPM, warning text, face bbox, and waveform chunk. Camera frames stay on /ear/frame.
"""

import json
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


class RppgUdpBridgeNode(Node):
    def __init__(self):
        super().__init__("rppg_udp_bridge_node")

        self.declare_parameter("udp_host", "127.0.0.1")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("timer_period", 0.02)

        self.udp_host = self.get_parameter("udp_host").value
        self.udp_port = int(self.get_parameter("udp_port").value)
        timer_period = float(self.get_parameter("timer_period").value)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind((self.udp_host, self.udp_port))

        self.bpm_pub        = self.create_publisher(Float32, "/rppg/bpm", 10)
        self.warning_pub    = self.create_publisher(String, "/rppg/warning_text", 10)
        self.bbox_pub       = self.create_publisher(String, "/rppg/face_bbox", 10)
        self.wave_pub       = self.create_publisher(String, "/rppg/wave_chunk", 10)

        self.timer = self.create_timer(timer_period, self.poll_udp)
        self.get_logger().info(f"Listening for rPPG UDP on {self.udp_host}:{self.udp_port}")

    def poll_udp(self):
        while True:
            try:
                data, _addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError as exc:
                self.get_logger().warning(f"UDP receive failed: {exc}")
                return

            try:
                payload = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.get_logger().warning(f"Ignoring invalid rPPG UDP packet: {exc}")
                continue

            if payload.get("type") != "rppg_update":
                continue

            self.publish_payload(payload)

    def publish_payload(self, payload):
        bpm = payload.get("bpm")
        if bpm is not None:
            msg = Float32()
            msg.data = float(bpm)
            self.bpm_pub.publish(msg)

        warning_msg = String()
        warning_msg.data = str(payload.get("warning_text", ""))
        self.warning_pub.publish(warning_msg)

        bbox_msg = String()
        bbox_msg.data = self.format_number_list(payload.get("face_bbox", [0, 0, 0, 0]), as_int=True)
        self.bbox_pub.publish(bbox_msg)

        wave_msg = String()
        wave_msg.data = self.format_number_list(payload.get("wave_chunk", []), as_int=False)
        self.wave_pub.publish(wave_msg)

    @staticmethod
    def format_number_list(values, as_int=False):
        if not isinstance(values, (list, tuple)):
            return ""

        formatted = []
        for value in values:
            try:
                if as_int:
                    formatted.append(str(int(value)))
                else:
                    formatted.append(f"{float(value):.6g}")
            except (TypeError, ValueError):
                continue
        return ",".join(formatted)

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RppgUdpBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

