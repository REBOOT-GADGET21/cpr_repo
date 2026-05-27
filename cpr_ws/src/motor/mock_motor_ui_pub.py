#!/usr/bin/env python3

import time
import random

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float32, Int32, String, Float32MultiArray


class MockMotorUIPublisher(Node):
    def __init__(self):
        super().__init__("mock_motor_ui_publisher")

        # ============================================================
        # UI 테스트용 설정값
        # ============================================================

        # BPM 범위: 106~108 bpm 사이에서만 변화
        self.min_bpm = 106.0
        self.max_bpm = 108.0

        # 초기 기준 BPM
        self.current_bpm = 107.0

        # 표시할 압박 깊이 상태값 [cm]
        self.target_depth_cm = 5.0

        # UI 갱신 주기 [s]
        self.publish_period = 0.05  # 20 Hz

        # 현재 BPM에 따른 1회 압박 주기
        self.current_cycle_period = 60.0 / self.current_bpm

        # ============================================================
        # 위치/전류 mock 값
        # ============================================================

        self.initial_position = -3584
        self.contact_position = 30000

        # 5 cm = 75000 unit
        self.travel_units = 75000
        self.end_position = self.contact_position + self.travel_units

        # ============================================================
        # 내부 상태값
        # ============================================================

        self.running = False
        self.start_time = None
        self.next_compression_time = None

        self.motor_state = "IDLE"

        self.compression_count = 0
        self.compression_bpm = 0.0
        self.compression_time_s = 0.0
        self.compression_depth_cm = 0.0

        self.current_position = self.initial_position
        self.current_a = 0.0

        self.last_count_for_log = -1

        # ============================================================
        # Subscriber
        # ============================================================

        self.create_subscription(
            Bool,
            "/motor_start",
            self.cb_motor_start,
            10
        )

        self.create_subscription(
            Bool,
            "/motor_stop",
            self.cb_motor_stop,
            10
        )

        # ============================================================
        # Publisher
        # 기존 MotorRosBridge pub 구조와 맞춤
        # ============================================================

        self.pub_abs_pos = self.create_publisher(
            Int32,
            "/motor_absolute_position",
            10
        )

        # /motor_compression_status
        # data[0] = compression_count
        # data[1] = compression_bpm
        # data[2] = compression_time_s
        # data[3] = compression_depth_cm
        self.pub_compression_status = self.create_publisher(
            Float32MultiArray,
            "/motor_compression_status",
            10
        )

        self.pub_current_a = self.create_publisher(
            Float32,
            "/motor_current_a",
            10
        )

        self.pub_state = self.create_publisher(
            String,
            "/motor_state",
            10
        )

        # ============================================================
        # 로드셀 mock publisher
        # ============================================================

        self.pub_loadcell_total = self.create_publisher(
            Float32,
            "/loadcell_total",
            10
        )

        self.pub_loadcell_stop_request = self.create_publisher(
            Bool,
            "/loadcell_stop_request",
            10
        )

        self.pub_loadcell_status_code = self.create_publisher(
            Int32,
            "/loadcell_status_code",
            10
        )

        self.pub_loadcell_warning = self.create_publisher(
            Bool,
            "/loadcell_warning",
            10
        )

        # ============================================================
        # Timer
        # ============================================================

        self.timer = self.create_timer(
            self.publish_period,
            self.timer_callback
        )

        self.get_logger().info("mock_motor_ui_publisher started")
        self.get_logger().info(
            '실행: ros2 topic pub /motor_start std_msgs/Bool "data: true" --once'
        )
        self.get_logger().info(
            '정지: ros2 topic pub /motor_stop std_msgs/Bool "data: true" --once'
        )

        self.publish_reset_values(state="IDLE")

    # ============================================================
    # Callback
    # ============================================================

    def cb_motor_start(self, msg):
        if not msg.data:
            return

        self.get_logger().info("[MOCK] /motor_start=True received")

        self.running = True
        self.start_time = time.time()

        self.motor_state = "Compressing"

        self.compression_count = 0
        self.compression_bpm = 0.0
        self.compression_time_s = 0.0

        # 시작 직후 깊이는 목표 압박 깊이 상태로 표시
        self.compression_depth_cm = self.target_depth_cm

        # 시작 시 첫 압박 BPM 설정: 106~108 사이
        self.current_bpm = random.uniform(self.min_bpm, self.max_bpm)
        self.current_cycle_period = 60.0 / self.current_bpm

        # 첫 압박 완료 예정 시각
        self.next_compression_time = self.start_time + self.current_cycle_period

        self.current_position = self.end_position
        self.current_a = 8.0

        self.last_count_for_log = -1

        self.publish_all()

    def cb_motor_stop(self, msg):
        if not msg.data:
            return

        self.get_logger().warn("[MOCK] /motor_stop=True received -> RESET")

        self.running = False
        self.start_time = None
        self.next_compression_time = None

        self.publish_reset_values(state="STOPPED")

    # ============================================================
    # Publish
    # ============================================================

    def publish_reset_values(self, state):
        self.motor_state = str(state)

        self.compression_count = 0
        self.compression_bpm = 0.0
        self.compression_time_s = 0.0
        self.compression_depth_cm = 0.0

        self.current_bpm = 107.0
        self.current_cycle_period = 60.0 / self.current_bpm

        self.current_position = self.initial_position
        self.current_a = 0.0

        self.publish_all()

    def publish_all(self):
        # 모터 상태
        self.pub_state.publish(
            String(data=str(self.motor_state))
        )

        # 모터 위치
        self.pub_abs_pos.publish(
            Int32(data=int(self.current_position))
        )

        # 모터 전류
        self.pub_current_a.publish(
            Float32(data=float(self.current_a))
        )

        # 압박 상태 통합 발행
        status_msg = Float32MultiArray()
        status_msg.data = [
            float(self.compression_count),      # data[0] 압박 횟수
            float(self.compression_bpm),        # data[1] BPM
            float(self.compression_time_s),     # data[2] 압박 시간
            float(self.compression_depth_cm),   # data[3] 압박 깊이 상태
        ]
        self.pub_compression_status.publish(status_msg)

        # 로드셀 mock 값
        if self.running:
            # 압박 중에는 약 450N 근처로 표시
            loadcell_total_n = 450.0 + random.uniform(-10.0, 10.0)
            loadcell_stop_request = False
            loadcell_status_code = 0
            loadcell_warning = False
        else:
            loadcell_total_n = 0.0
            loadcell_stop_request = False
            loadcell_status_code = 0
            loadcell_warning = False

        self.pub_loadcell_total.publish(
            Float32(data=float(loadcell_total_n))
        )

        self.pub_loadcell_stop_request.publish(
            Bool(data=loadcell_stop_request)
        )

        self.pub_loadcell_status_code.publish(
            Int32(data=int(loadcell_status_code))
        )

        self.pub_loadcell_warning.publish(
            Bool(data=loadcell_warning)
        )

    # ============================================================
    # Main timer
    # ============================================================

    def timer_callback(self):
        # 정지 상태에서도 현재 값 계속 발행
        if not self.running or self.start_time is None:
            self.publish_all()
            return

        now = time.time()

        # ========================================================
        # 압박 시간
        # ========================================================
        self.compression_time_s = now - self.start_time

        # ========================================================
        # 압박 횟수 / BPM
        #
        # 핵심:
        # - BPM을 106~108 사이에서 압박 1회마다 정함
        # - 그 BPM에 맞는 주기 60/BPM 뒤에 다음 count 증가
        # - 따라서 BPM 값과 count 증가 속도가 서로 일치함
        # ========================================================
        if self.next_compression_time is not None:
            while now >= self.next_compression_time:
                # 압박 1회 완료
                self.compression_count += 1

                # 이번 압박의 실제 BPM을 106~108 사이로 설정
                self.current_bpm = random.uniform(self.min_bpm, self.max_bpm)
                self.compression_bpm = round(self.current_bpm, 1)

                # 현재 BPM에 맞는 다음 압박 주기
                self.current_cycle_period = 60.0 / self.current_bpm

                # 다음 압박 완료 예정 시각
                self.next_compression_time += self.current_cycle_period

                # 압박 깊이 상태값 갱신
                # 0~5cm 파형이 아니라, 실제 도달 깊이처럼 4.90~5.00cm 근처 표시
                self.compression_depth_cm = round(
                    self.target_depth_cm - random.uniform(0.00, 0.10),
                    2
                )

        # count가 아직 0이면 BPM은 0으로 표시
        if self.compression_count == 0:
            self.compression_bpm = 0.0
            self.compression_depth_cm = self.target_depth_cm

        # ========================================================
        # 위치/전류 mock
        # 깊이 상태값에 맞춰 압박 위치 근처로 표시
        # ========================================================
        depth_ratio = self.compression_depth_cm / self.target_depth_cm

        self.current_position = self.contact_position + int(
            self.travel_units * depth_ratio
        )

        self.current_a = 8.0 + random.uniform(-0.2, 0.2)

        self.motor_state = "Compressing"

        self.publish_all()

        # 로그는 count가 바뀔 때만 출력
        if self.compression_count != self.last_count_for_log:
            self.last_count_for_log = self.compression_count

            self.get_logger().info(
                f"count={self.compression_count} | "
                f"bpm={self.compression_bpm:.1f} | "
                f"time={self.compression_time_s:.2f}s | "
                f"depth={self.compression_depth_cm:.2f}cm | "
                f"period={self.current_cycle_period:.3f}s"
            )


def main(args=None):
    rclpy.init(args=args)

    node = MockMotorUIPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()