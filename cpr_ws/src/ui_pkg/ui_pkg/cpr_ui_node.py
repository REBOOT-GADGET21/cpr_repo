import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import String, Float32, Int32, Bool, Float32MultiArray
from sensor_msgs.msg import Image

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QStackedWidget,
    QFrame,
    QTextEdit,
)
from PyQt5.QtGui import QFont, QImage, QPixmap

try:
    import pyqtgraph as pg
except ImportError:
    pg = None


# ==============================
# UI 표시 모드
# ==============================
MODE_EAR = "EAR"
MODE_RPPG = "RPPG"


class CPRUINode(Node):
    """ROS2 topic을 subscribe/publish하는 UI용 노드."""

    def __init__(self):
        super().__init__("cpr_ui_node")

        # -------------------------
        # UI -> Motor 명령 publish
        # -------------------------
        self.pub_motor_start = self.create_publisher(Bool, "/motor_start", 10)
        self.pub_motor_stop = self.create_publisher(Bool, "/motor_stop", 10)

        # -------------------------
        # Motor / Loadcell subscribe
        # -------------------------
        self.motor_abs_pos = 0
        self.motor_contact_pos = 0
        self.motor_target_pos = 0
        self.motor_count = 0
        self.motor_bpm = 0.0
        self.motor_time = 0.0
        self.motor_depth_cm = 0.0
        self.motor_current_a = 0.0
        self.motor_state = "IDLE"

        self.loadcell_total = 0.0
        self.loadcell_stop_request = False
        self.loadcell_status_code = 0
        self.loadcell_warning = False

        self.create_subscription(Int32, "/motor_absolute_position", self.cb_motor_abs_pos, 10)
        self.create_subscription(Int32, "/motor_contact_position", self.cb_motor_contact_pos, 10)
        self.create_subscription(Int32, "/motor_target_position", self.cb_motor_target_pos, 10)
        # 모터의 값을 묶어서 받는 토픽/ 압박 횟수, bpm, 시간, 깊이를 한번에 받음
        self.create_subscription(Float32MultiArray, "/motor_compression_status", self.cb_motor_compression_status, 10)
        self.create_subscription(Float32, "/motor_current_a", self.cb_motor_current_a, 10)
        self.create_subscription(String, "/motor_state", self.cb_motor_state, 10)

        self.create_subscription(Float32, "/loadcell_total", self.cb_loadcell_total, 10)
        self.create_subscription(Bool, "/loadcell_stop_request", self.cb_loadcell_stop_request, 10)
        self.create_subscription(Int32, "/loadcell_status_code", self.cb_loadcell_status_code, 10)
        self.create_subscription(Bool, "/loadcell_warning", self.cb_loadcell_warning, 10)

        # -------------------------
        # EAR subscribe
        # -------------------------
        self.eye_state = "N/A"
        self.ear = 0.0
        self.closed_duration = 0.0
        self.motion_score = 0.0
        self.final_response = "N/A"

        self.create_subscription(String, "Response/eye_state", self.cb_eye_state, 10)
        self.create_subscription(Float32, "Response/ear", self.cb_ear, 10)
        self.create_subscription(Float32, "Response/closed_duration", self.cb_closed_duration, 10)
        self.create_subscription(Float32, "Response/motion_score", self.cb_motion_score, 10)
        self.create_subscription(String, "Response/final_response", self.cb_final_response, 10)

        # -------------------------
        # rPPG subscribe
        # 없는 토픽은 임의 설계
        # -------------------------
        self.rppg_face_detected = False
        self.rppg_bpm = 0.0
        self.rppg_quality_ok = False
        self.rppg_warning_text = ""
        self.rppg_fps = 0.0
        self.rppg_sample_rate = 30.0
        self.rppg_seq = 0
        self.rppg_wave_samples = deque(maxlen=600)
        self.rppg_bbox = [0, 0, 0, 0]

        # 간단한 std_msgs 기반 설계
        self.create_subscription(Float32, "/rppg/bpm", self.cb_rppg_bpm, 10)
        self.create_subscription(Bool, "/rppg/quality_ok", self.cb_rppg_quality_ok, 10)
        self.create_subscription(String, "/rppg/warning_text", self.cb_rppg_warning_text, 10)
        self.create_subscription(Bool, "/rppg/face_detected", self.cb_rppg_face_detected, 10)
        self.create_subscription(Float32, "/rppg/fps", self.cb_rppg_fps, 10)
        self.create_subscription(Float32, "/rppg/sample_rate", self.cb_rppg_sample_rate, 10)
        self.create_subscription(Int32, "/rppg/seq", self.cb_rppg_seq, 10)

        # bbox는 x,y,w,h를 문자열 "x,y,w,h" 형태로 받는 단순 설계
        self.create_subscription(String, "/rppg/face_bbox", self.cb_rppg_bbox, 10)

        # wave chunk도 문자열 "0.1,0.2,0.3" 형태로 받는 단순 설계
        self.create_subscription(String, "/rppg/wave_chunk", self.cb_rppg_wave_chunk, 10)

        # UI화면에는 EAR 영상만 뜸
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.latest_frame_msg = None
        self.latest_ear_qimage = None
        self.create_subscription(Image, "/ear/frame", self.cb_ear_frame, image_qos)

    # ==============================
    # UI 명령 publish
    # ==============================
    def publish_motor_start(self):
        msg = Bool()
        msg.data = True
        self.pub_motor_start.publish(msg)
        self.get_logger().info("Published /motor_start True")

    def publish_motor_stop(self):
        msg = Bool()
        msg.data = True
        self.pub_motor_stop.publish(msg)
        self.get_logger().info("Published /motor_stop True")

    # ==============================
    # Motor callbacks
    # ==============================
    def cb_motor_abs_pos(self, msg):
        self.motor_abs_pos = msg.data

    def cb_motor_contact_pos(self, msg):
        self.motor_contact_pos = msg.data

    def cb_motor_target_pos(self, msg):
        self.motor_target_pos = msg.data

    def cb_motor_compression_status(self, msg):
        values = list(msg.data)
        if len(values) >= 1:
            self.motor_count = int(values[0])
        if len(values) >= 2:
            self.motor_bpm = float(values[1])
        if len(values) >= 3:
            self.motor_time = float(values[2])
        if len(values) >= 4:
            self.motor_depth_cm = float(values[3])

    def cb_motor_current_a(self, msg):
        self.motor_current_a = msg.data

    def cb_motor_state(self, msg):
        self.motor_state = msg.data

    # ==============================
    # Loadcell callbacks
    # ==============================
    def cb_loadcell_total(self, msg):
        self.loadcell_total = msg.data

    def cb_loadcell_stop_request(self, msg):
        self.loadcell_stop_request = msg.data

    def cb_loadcell_status_code(self, msg):
        self.loadcell_status_code = msg.data

    def cb_loadcell_warning(self, msg):
        self.loadcell_warning = msg.data

    # ==============================
    # EAR callbacks
    # ==============================
    def cb_eye_state(self, msg):
        self.eye_state = msg.data

    def cb_ear(self, msg):
        self.ear = msg.data

    def cb_closed_duration(self, msg):
        self.closed_duration = msg.data

    def cb_motion_score(self, msg):
        self.motion_score = msg.data

    def cb_final_response(self, msg):
        self.final_response = msg.data

    # ==============================
    # rPPG callbacks
    # ==============================
    def cb_rppg_bpm(self, msg):
        self.rppg_bpm = msg.data

    def cb_rppg_quality_ok(self, msg):
        self.rppg_quality_ok = msg.data

    def cb_rppg_warning_text(self, msg):
        self.rppg_warning_text = msg.data

    def cb_rppg_face_detected(self, msg):
        self.rppg_face_detected = msg.data

    def cb_rppg_fps(self, msg):
        self.rppg_fps = msg.data

    def cb_rppg_sample_rate(self, msg):
        self.rppg_sample_rate = msg.data

    def cb_rppg_seq(self, msg):
        self.rppg_seq = msg.data

    def cb_rppg_bbox(self, msg):
        try:
            values = [int(v.strip()) for v in msg.data.split(",")]
            if len(values) == 4:
                self.rppg_bbox = values
        except Exception:
            pass

    def cb_rppg_wave_chunk(self, msg):
        try:
            samples = [float(v.strip()) for v in msg.data.split(",") if v.strip()]
            for sample in samples:
                self.rppg_wave_samples.append(sample)
        except Exception:
            pass
    # Qimage로 변환하여 UI에 표시
    def cb_ear_frame(self, msg):
        self.latest_frame_msg = msg
        self.latest_ear_qimage = self.ros_image_to_qimage(msg)

    @staticmethod
    def ros_image_to_qimage(msg):
        if msg.encoding == "bgr8":
            image_format = getattr(QImage, "Format_BGR888", None)
            if image_format is not None:
                return QImage(bytes(msg.data), msg.width, msg.height, msg.step, image_format).copy()

            qimage = QImage(bytes(msg.data), msg.width, msg.height, msg.step, QImage.Format_RGB888)
            return qimage.rgbSwapped().copy()

        if msg.encoding == "rgb8":
            return QImage(bytes(msg.data), msg.width, msg.height, msg.step, QImage.Format_RGB888).copy()

        return None


class StartPage(QWidget):
    def __init__(self, on_start):
        super().__init__()
        self.on_start = on_start

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("CPR ROBOT")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 36, QFont.Bold))

        subtitle = QLabel("Automatic CPR Monitoring System")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 16))

        start_btn = QPushButton("START")
        start_btn.setFixedSize(240, 80)
        start_btn.setFont(QFont("Arial", 24, QFont.Bold))
        start_btn.clicked.connect(self.on_start)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(40)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        self.setLayout(layout)


class MonitorPage(QWidget):
    def __init__(self, ros_node: CPRUINode, on_stop):
        super().__init__()
        self.ros_node = ros_node
        self.on_stop = on_stop
        self.current_mode = MODE_EAR
        self.mode_start_time = time.time()

        self.setup_ui()

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)

    def setup_ui(self):
        root = QVBoxLayout()

        # -------------------------
        # 상단 CPR 고정 정보 영역
        # -------------------------
        self.status_frame = QFrame()
        self.status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QGridLayout()

        self.lbl_time = self.make_big_label("진행 시간: 00:00")
        self.lbl_depth = self.make_big_label("압박 깊이: -- cm")
        self.lbl_bpm = self.make_big_label("압박 속도: -- BPM")
        self.lbl_recoil = self.make_big_label("최대 이완: --")
        self.lbl_count = self.make_big_label("압박 횟수: 0")
        self.lbl_motor_state = self.make_big_label("모터 상태: IDLE")

        status_layout.addWidget(self.lbl_time, 0, 0)
        status_layout.addWidget(self.lbl_depth, 0, 1)
        status_layout.addWidget(self.lbl_bpm, 0, 2)
        status_layout.addWidget(self.lbl_recoil, 1, 0)
        status_layout.addWidget(self.lbl_count, 1, 1)
        status_layout.addWidget(self.lbl_motor_state, 1, 2)
        self.status_frame.setLayout(status_layout)

        # -------------------------
        # 중앙: 카메라 + 동적 패널
        # -------------------------
        center_layout = QHBoxLayout()

        self.camera_box = QLabel("Camera View\n/ear/frame 연결 전")
        self.camera_box.setAlignment(Qt.AlignCenter)
        self.camera_box.setMinimumSize(640, 420)
        self.camera_box.setStyleSheet("background-color: #111; color: white; border: 2px solid #444;")
        self.camera_box.setFont(QFont("Arial", 18, QFont.Bold))

        self.dynamic_frame = QFrame()
        self.dynamic_frame.setFrameShape(QFrame.StyledPanel)
        self.dynamic_layout = QVBoxLayout()

        self.lbl_mode = self.make_title_label("EAR MONITORING")
        self.lbl_line1 = self.make_big_label("")
        self.lbl_line2 = self.make_big_label("")
        self.lbl_line3 = self.make_big_label("")
        self.lbl_line4 = self.make_big_label("")
        self.lbl_line5 = self.make_big_label("")

        self.graph = None
        self.curve = None
        if pg is not None:
            self.graph = pg.PlotWidget()
            self.graph.setMinimumHeight(180)
            self.curve = self.graph.plot([])
        else:
            self.graph = QLabel("pyqtgraph 미설치: waveform 표시 비활성")
            self.graph.setAlignment(Qt.AlignCenter)

        self.dynamic_layout.addWidget(self.lbl_mode)
        self.dynamic_layout.addWidget(self.lbl_line1)
        self.dynamic_layout.addWidget(self.lbl_line2)
        self.dynamic_layout.addWidget(self.lbl_line3)
        self.dynamic_layout.addWidget(self.lbl_line4)
        self.dynamic_layout.addWidget(self.lbl_line5)
        self.dynamic_layout.addWidget(self.graph)
        self.set_panel_stretch(graph_visible=False)
        self.dynamic_frame.setLayout(self.dynamic_layout)

        center_layout.addWidget(self.camera_box, 3)
        center_layout.addWidget(self.dynamic_frame, 2)

        # -------------------------
        # 하단 로그 / 버튼
        # -------------------------
        bottom_layout = QHBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(100)
        self.log("UI started")

        stop_btn = QPushButton("STOP")
        stop_btn.setFixedSize(160, 70)
        stop_btn.setFont(QFont("Arial", 20, QFont.Bold))
        stop_btn.clicked.connect(self.on_stop)

        bottom_layout.addWidget(self.log_box)
        bottom_layout.addWidget(stop_btn)

        root.addWidget(self.status_frame)
        root.addLayout(center_layout)
        root.addLayout(bottom_layout)
        self.setLayout(root)

    def make_big_label(self, text):
        label = QLabel(text)
        label.setFont(QFont("Arial", 16, QFont.Bold))
        label.setMinimumHeight(40)
        return label

    def make_title_label(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 24, QFont.Bold))
        label.setMinimumHeight(60)
        return label

    def log(self, text):
        now = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{now}] {text}")

    def update_mode_by_cycle(self):
        """2분 EAR → 30초 rPPG → 반복."""
        elapsed = time.time() - self.mode_start_time

        if self.current_mode == MODE_EAR and elapsed >= 120.0:
            self.current_mode = MODE_RPPG
            self.mode_start_time = time.time()
            self.log("Switching to rPPG monitoring")

        elif self.current_mode == MODE_RPPG and elapsed >= 30.0:
            self.current_mode = MODE_EAR
            self.mode_start_time = time.time()
            self.log("Switching to EAR monitoring")

    def update_ui(self):
        self.update_mode_by_cycle()
        self.update_cpr_status()
        self.update_camera_view()

        if self.current_mode == MODE_EAR:
            self.update_ear_panel()
        else:
            self.update_rppg_panel()

    def update_cpr_status(self):
        n = self.ros_node
        minutes = int(n.motor_time // 60)
        seconds = int(n.motor_time % 60)

        self.lbl_time.setText(f"압박 시간: {minutes:02d}:{seconds:02d}")
        self.lbl_bpm.setText(f"BPM: {n.motor_bpm:05.1f}")
        self.lbl_count.setText(f"압박 횟수: {n.motor_count:02d}")
        self.lbl_motor_state.setText(f"모터 상태: {n.motor_state}")

        self.lbl_depth.setText(f"압박 깊이: {n.motor_depth_cm:.1f} cm")

        recoil_text = "WARNING" if n.loadcell_warning else "GOOD"
        self.lbl_recoil.setText(f"최대 이완/편심: {recoil_text}")

    def update_camera_view(self):
        n = self.ros_node
        if n.latest_ear_qimage is None:
            self.camera_box.setText("Camera View\n/ear/frame 연결 전")
            return

        pixmap = QPixmap.fromImage(n.latest_ear_qimage)
        self.camera_box.setPixmap(
            pixmap.scaled(
                self.camera_box.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def update_ear_panel(self):
        n = self.ros_node
        remain = max(0, 120 - int(time.time() - self.mode_start_time))
        self.set_compact_dynamic_text(False)

        self.lbl_mode.setText(f"EAR MONITORING  |  남은 시간 {remain}s")
        self.lbl_line1.setText(f"EAR 값: {n.ear:.3f}")
        self.lbl_line2.setText(f"눈 상태: {n.eye_state}")
        self.lbl_line3.setText(f"눈 감김 시간: {n.closed_duration:.2f} s")
        self.lbl_line4.setText(f"움직임 점수: {n.motion_score:.3f}")
        self.lbl_line5.setText(f"최종 반응: {n.final_response}")

        self.set_graph_visible(False)
        if self.curve is not None:
            self.curve.setData([])

    def update_rppg_panel(self):
        n = self.ros_node
        remain = max(0, 30 - int(time.time() - self.mode_start_time))
        quality = "GOOD" if n.rppg_quality_ok else "BAD / WAIT"
        self.set_compact_dynamic_text(True)

        self.lbl_mode.setText(f"rPPG MONITORING  |  남은 시간 {remain}s")
        self.lbl_line1.setText(f"BPM: {n.rppg_bpm:.1f}")
        self.lbl_line2.setText(f"신호 품질: {quality}")
        self.lbl_line3.setText(f"경고: {n.rppg_warning_text if n.rppg_warning_text else 'None'}")
        self.lbl_line4.setText(f"Face detected: {n.rppg_face_detected}")
        self.lbl_line5.setText(f"FPS: {n.rppg_fps:.1f} / Seq: {n.rppg_seq}")

        if self.curve is not None:
            self.set_graph_visible(True)
            y = list(n.rppg_wave_samples)
            self.curve.setData(y)
        else:
            self.set_graph_visible(True)

    def set_graph_visible(self, visible):
        self.graph.setVisible(visible)
        self.set_panel_stretch(graph_visible=visible)

    def set_panel_stretch(self, graph_visible):
        self.dynamic_layout.setStretch(0, 1)
        for i in range(1, 6):
            self.dynamic_layout.setStretch(i, 1)
        self.dynamic_layout.setStretch(6, 4 if graph_visible else 0)

    def set_compact_dynamic_text(self, compact):
        title_font = QFont("Arial", 20 if compact else 24, QFont.Bold)
        line_font = QFont("Arial", 13 if compact else 16, QFont.Bold)
        self.lbl_mode.setFont(title_font)
        for label in (self.lbl_line1, self.lbl_line2, self.lbl_line3, self.lbl_line4, self.lbl_line5):
            label.setFont(line_font)
            label.setMinimumHeight(28 if compact else 40)


class MainWindow(QMainWindow):
    def __init__(self, ros_node: CPRUINode):
        super().__init__()
        self.ros_node = ros_node

        self.setWindowTitle("CPR Robot Integrated UI")
        self.resize(1280, 800)

        self.stack = QStackedWidget()
        self.start_page = StartPage(self.start_cpr)
        self.monitor_page = MonitorPage(self.ros_node, self.stop_cpr)

        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.monitor_page)
        self.setCentralWidget(self.stack)

        self.ros_timer = QTimer()
        self.ros_timer.timeout.connect(self.spin_ros_once)
        self.ros_timer.start(10)

    def spin_ros_once(self):
        rclpy.spin_once(self.ros_node, timeout_sec=0.0)

    def start_cpr(self):
        self.ros_node.publish_motor_start()
        self.monitor_page.current_mode = MODE_EAR
        self.monitor_page.mode_start_time = time.time()
        self.monitor_page.log("CPR started by UI")
        self.stack.setCurrentIndex(1)

    def stop_cpr(self):
        self.ros_node.publish_motor_stop()
        self.monitor_page.log("CPR stopped by UI")
        self.stack.setCurrentIndex(0)

    def closeEvent(self, event):
        self.ros_node.destroy_node()
        rclpy.shutdown()
        event.accept()


def main(args=None):
    rclpy.init(args=args)
    ros_node = CPRUINode()

    app = QApplication(sys.argv)
    window = MainWindow(ros_node)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
