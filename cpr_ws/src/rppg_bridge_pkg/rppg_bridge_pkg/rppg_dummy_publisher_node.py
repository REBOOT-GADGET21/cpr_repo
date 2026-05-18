#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Publish plausible fixed rPPG values for the CPR UI.

This node is for UI tests when the real camera rPPG pipeline is not running.
It leaves EAR topics and /ear/frame untouched, and only publishes the rPPG
topics that cpr_ui_node.py already subscribes to.
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String


class RppgDummyPublisherNode(Node):
    def __init__(self):
        super().__init__("rppg_dummy_publisher_node")

        self.declare_parameter("bpm", 74.0)
        self.declare_parameter("sample_rate", 30.0)
        self.declare_parameter("chunk_size", 1)
        self.declare_parameter("amplitude", 1.0)
        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("graph_points_per_cycle", 100.0)

        self.bpm = float(self.get_parameter("bpm").value)
        self.sample_rate = float(self.get_parameter("sample_rate").value)
        self.chunk_size = int(self.get_parameter("chunk_size").value)
        self.amplitude = float(self.get_parameter("amplitude").value)
        timer_period = float(self.get_parameter("timer_period").value)
        self.graph_points_per_cycle = float(
            self.get_parameter("graph_points_per_cycle").value
        )

        self.seq = 0
        self.sample_index = 0

        self.bpm_pub = self.create_publisher(Float32, "/rppg/bpm", 10)
        self.quality_pub = self.create_publisher(Bool, "/rppg/quality_ok", 10)
        self.warning_pub = self.create_publisher(String, "/rppg/warning_text", 10)
        self.face_pub = self.create_publisher(Bool, "/rppg/face_detected", 10)
        self.fps_pub = self.create_publisher(Float32, "/rppg/fps", 10)
        self.sample_rate_pub = self.create_publisher(Float32, "/rppg/sample_rate", 10)
        self.seq_pub = self.create_publisher(Int32, "/rppg/seq", 10)
        self.bbox_pub = self.create_publisher(String, "/rppg/face_bbox", 10)
        self.wave_pub = self.create_publisher(String, "/rppg/wave_chunk", 10)

        self.timer = self.create_timer(timer_period, self.publish_dummy_values)
        self.get_logger().info(
            f"Publishing dummy rPPG: {self.bpm:.1f} bpm, "
            f"{self.sample_rate:.1f} Hz wave on /rppg/*"
        )

    def publish_dummy_values(self):
        self.seq += 1

        self.publish_float(self.bpm_pub, self.bpm)
        self.publish_bool(self.quality_pub, True)
        self.publish_string(self.warning_pub, "")
        self.publish_bool(self.face_pub, True)
        self.publish_float(self.fps_pub, self.sample_rate)
        self.publish_float(self.sample_rate_pub, self.sample_rate)
        self.publish_int(self.seq_pub, self.seq)
        self.publish_string(self.bbox_pub, "190,90,260,260")

        samples = [self.make_wave_sample() for _ in range(max(1, self.chunk_size))]
        self.publish_string(self.wave_pub, ",".join(f"{sample:.5f}" for sample in samples))

    def make_wave_sample(self):
        phase = 2.0 * math.pi * self.sample_index / max(1.0, self.graph_points_per_cycle)
        sample = math.sin(phase)
        self.sample_index += 1
        return self.amplitude * sample

    @staticmethod
    def publish_float(publisher, value):
        msg = Float32()
        msg.data = float(value)
        publisher.publish(msg)

    @staticmethod
    def publish_bool(publisher, value):
        msg = Bool()
        msg.data = bool(value)
        publisher.publish(msg)

    @staticmethod
    def publish_int(publisher, value):
        msg = Int32()
        msg.data = int(value)
        publisher.publish(msg)

    @staticmethod
    def publish_string(publisher, value):
        msg = String()
        msg.data = str(value)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RppgDummyPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
