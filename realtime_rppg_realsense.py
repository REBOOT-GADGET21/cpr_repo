#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Real-time rPPG inference from Intel RealSense (or webcam fallback).
   <<<추론 - 체크포인트를 불러와서 실시간 실행하는 코드>>

Usage examples:
  python realtime_rppg_realsense.py --model tscan --checkpoint ./final_model_release/PURE_TSCAN.pth --use-realsense
  python realtime_rppg_realsense.py --model physnet --checkpoint ./final_model_release/PURE_PhysNet_DiffNormalized.pth --use-realsense
"""

import argparse
import json
import os
import sys
import time
from collections import deque
import socket
from pathlib import Path

import cv2
import numpy as np
import torch

# Make the repository root importable when running from src/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_util import make_model
from signal_processing import estimate_bpm_from_signal, infer_signal
from frame_source import FrameSource
from ui import draw_face_panel, make_panel_background, next_log_path, update_ekg_trace


# ******* Debug UI / UDP 기본 설정 *******
SHOW_DEBUG_WINDOW = True
DEFAULT_UDP_HOST = "127.0.0.1"
DEFAULT_UDP_PORT = 5005

# ******* 이전 Arduino 확인용 UDP 송신 코드 *******
# BPM 값이 잘 출력되는지 외부 센서와 비교하기 위해 사용했던 코드입니다.
# 새 구조에서는 rPPG 결과를 JSON UDP로 ROS2 bridge에 보내므로 사용하지 않습니다.
# old_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# OLD_UDP_IP = "127.0.0.1"
# OLD_UDP_PORT = 5000

# ******* CLI(Command Line Interface)) 설정 관리 위한 함수 ********
def parse_args():
    parser = argparse.ArgumentParser(description="Real-time rPPG with RealSense/Webcam using TSCAN or PhysNet.")
    parser.add_argument("--model", choices=["tscan", "physnet"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-realsense", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--input-size", type=int, default=72)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--infer-stride", type=int, default=15)
    parser.add_argument("--diff-flag", dest="diff_flag", action="store_true")
    parser.add_argument("--no-diff-flag", dest="diff_flag", action="store_false")
    parser.set_defaults(diff_flag=True)
    parser.add_argument("--tscan-frame-depth", type=int, default=10)
    parser.add_argument("--tscan-clip-len", type=int, default=180)      # TSCAN은 기본 프레임 180, default = 180
    parser.add_argument("--physnet-frame-num", type=int, default=128)   # PhysNet은 기본 프레임 128, default = 64
    parser.add_argument("--face-detect-interval", type=int, default=1)
    parser.add_argument("--send-udp", action="store_true", help="Send rPPG UI values to a UDP bridge.")
    parser.add_argument("--udp-host", type=str, default=DEFAULT_UDP_HOST)
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    parser.add_argument("--show-debug-window", dest="show_debug_window", action="store_true")
    parser.add_argument("--no-show-debug-window", dest="show_debug_window", action="store_false")
    parser.set_defaults(show_debug_window=SHOW_DEBUG_WINDOW)
    return parser.parse_args()


def bbox_xyxy_to_list(face_bbox):
    if face_bbox is None:
        return [0, 0, 0, 0]
    return [int(v) for v in face_bbox]


def make_rppg_payload(last_bpm, warning_text, face_bbox, wave_buffer, max_wave_samples=150):
    wave_values = list(wave_buffer)[-max_wave_samples:]
    return {
        "type": "rppg_update",
        "bpm": None if last_bpm is None else float(last_bpm),
        "warning_text": warning_text or "",
        "face_bbox": bbox_xyxy_to_list(face_bbox),
        "wave_chunk": [float(v) for v in wave_values],
    }


def send_rppg_udp(sock, payload, udp_host, udp_port):
    if sock is None:
        return
    try:
        packet = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        sock.sendto(packet, (udp_host, udp_port))
    except Exception as exc:
        print(f"[WARN] UDP send failed: {exc}")


# Simple face-box stabilizer for real-time inference.
# It keeps the detector in place, but smooths the raw box so the ROI does not jitter.
'''이전 bbox 대비 너무 많이 변경되면 그대로 유지할 수 있도록 함
중심점이 너무 많이 이동하면 유지'''
class FaceBoxStabilizer:
    def __init__(
        self,
        alpha=0.55,
        detect_interval=1,
        max_missed_frames=20,
        area_change_threshold=0.7,
        center_move_threshold=0.45,
        crop_shrink_ratio=0.05,
        min_box_size=32,
    ):
        self.alpha = alpha
        self.detect_interval = max(1, detect_interval)
        self.max_missed_frames = max_missed_frames
        self.area_change_threshold = area_change_threshold
        self.center_move_threshold = center_move_threshold
        self.crop_shrink_ratio = crop_shrink_ratio
        self.min_box_size = min_box_size
        self.prev_box = None  # Stored as (x, y, w, h)
        self.missed_frames = 0
        self.frame_idx = 0

    def _clip_box(self, box, frame_shape):
        h_img, w_img = frame_shape[:2]
        x, y, w, h = box
        x = max(0.0, min(x, w_img - 1.0))
        y = max(0.0, min(y, h_img - 1.0))
        w = max(1.0, min(w, w_img - x))
        h = max(1.0, min(h, h_img - y))
        return np.array([x, y, w, h], dtype=np.float32)

    def _pick_face(self, faces):
        # If we do not have a previous box, use the largest face.
        if self.prev_box is None:
            return max(faces, key=lambda b: b[2] * b[3])

        prev_x, prev_y, prev_w, prev_h = self.prev_box
        prev_cx = prev_x + prev_w / 2.0
        prev_cy = prev_y + prev_h / 2.0

        # If we already have a stable face, choose the face closest to the previous center.
        def face_score(face):
            x, y, w, h = face
            cx = x + w / 2.0
            cy = y + h / 2.0
            dist = (cx - prev_cx) ** 2 + (cy - prev_cy) ** 2
            return (dist, -(w * h))

        return min(faces, key=face_score)

    def _should_reject(self, curr_box):
        if self.prev_box is None:
            return False

        prev_x, prev_y, prev_w, prev_h = self.prev_box
        curr_x, curr_y, curr_w, curr_h = curr_box

        prev_area = max(prev_w * prev_h, 1.0)
        curr_area = max(curr_w * curr_h, 1.0)
        area_change = abs(curr_area - prev_area) / prev_area
        if area_change > self.area_change_threshold:
            return True

        prev_cx = prev_x + prev_w / 2.0
        prev_cy = prev_y + prev_h / 2.0
        curr_cx = curr_x + curr_w / 2.0
        curr_cy = curr_y + curr_h / 2.0
        center_shift = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5
        max_side = max(prev_w, prev_h, 1.0)
        if center_shift > self.center_move_threshold * max_side:
            return True

        return False

    def _shrink_box(self, box, frame_shape):
        # Shrink a little so the crop contains less background while keeping the face intact.
        x, y, w, h = box
        shrink_x = w * self.crop_shrink_ratio * 0.5
        shrink_y = h * self.crop_shrink_ratio * 0.5
        x += shrink_x
        y += shrink_y
        w -= 2.0 * shrink_x
        h -= 2.0 * shrink_y
        return self._clip_box(np.array([x, y, w, h], dtype=np.float32), frame_shape)

    def _box_to_xyxy(self, box, frame_shape):
        box = self._clip_box(box, frame_shape)
        x, y, w, h = box
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        h_img, w_img = frame_shape[:2]
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y2 = max(y1 + 1, min(y2, h_img))
        return (x1, y1, x2, y2)

    def _final_box_to_xyxy(self, box, frame_shape):
        return self._box_to_xyxy(self._shrink_box(box, frame_shape), frame_shape)

    def update(self, frame_bgr, detector):
        self.frame_idx += 1

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))

        if len(faces) == 0:
            self.missed_frames += 1
            if self.prev_box is not None and self.missed_frames <= self.max_missed_frames:
                return self._final_box_to_xyxy(self.prev_box, frame_bgr.shape)
            return None

        current_box = np.array(self._pick_face(faces), dtype=np.float32)
        current_box = self._clip_box(current_box, frame_bgr.shape)

        if self.prev_box is not None and self._should_reject(current_box):
            self.missed_frames += 1
            if self.missed_frames <= self.max_missed_frames:
                return self._final_box_to_xyxy(self.prev_box, frame_bgr.shape)
            return None

        if self.prev_box is None:
            smooth_box = current_box
        else:
            # EMA smoothing on x, y, w, h to reduce jitter.
            smooth_box = self.alpha * self.prev_box + (1.0 - self.alpha) * current_box

        smooth_box = self._clip_box(smooth_box, frame_bgr.shape)

        final_box = self._shrink_box(smooth_box, frame_bgr.shape)

        if final_box[2] < self.min_box_size or final_box[3] < self.min_box_size:
            self.missed_frames += 1
            if self.prev_box is not None and self.missed_frames <= self.max_missed_frames:
                return self._final_box_to_xyxy(self.prev_box, frame_bgr.shape)
            return None

        self.prev_box = smooth_box
        self.missed_frames = 0
        return self._box_to_xyxy(final_box, frame_bgr.shape)


#****************** 메인루프 BPM 추정과 관련된 함수 ******************
def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[WARN] CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    clip_len = args.tscan_clip_len if args.model == "tscan" else args.physnet_frame_num
    frame_buffer = deque(maxlen=clip_len)
    bpm_signal_buffer = deque(maxlen=max(int(args.window_seconds * args.fps), clip_len))
    ppg_wave_buffer = deque(maxlen=max(300, args.cam_width))  # 파형 출력용
    base_panel_w = min(max(50060, args.cam_width // 2 + 120), max(240, args.cam_width - 40))
    base_panel_h = min(max(260, args.cam_height // 2 + 80), max(180, args.cam_height - 40))
    PANEL_W = max(180, int(base_panel_w * 2 / 3))
    PANEL_H = max(140, int(base_panel_h * 2 / 3))
    ekg_background = make_panel_background(PANEL_W, PANEL_H)
    ekg_canvas = ekg_background.copy()
    ekg_state = {"pos": 0.0, "prev_point": None, "total_drawn": 0}
    face_stabilizer = FaceBoxStabilizer(detect_interval=args.face_detect_interval)

    cascade_path = "./dataset/haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")

    model = make_model(args, device)
    source = FrameSource(args.use_realsense, args.cam_width, args.cam_height, args.fps, args.camera_index)
    source.open()

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.send_udp else None
    udp_send_interval = max(args.infer_stride / float(args.fps if args.fps > 0 else 30), 0.1)
    last_no_face_udp_time = 0.0

    frame_idx = 0
    last_bpm = None
    kf_x = None     # 현재 추정 BPM
    kf_p = 1.0      # 추정 오차 공분산
    kf_q = 0.1      # (BPM이 얼마나 빨리 변할 수 있는가)크면 측정값 변화를 빠르게 받아들이므로 응답성이 좋아지고, 노이즈를 더 받음/ 작으면 더 부드럽지만 반응이 둔해짐
    kf_r = 30.0     # (추론 값을 얼마나 신뢰하는가)크면 측정값을 덜 믿고 이전 추정치를 유지하려고 하므로 안정적이지만 변화에 덜 민감함
    warning_text = ""
    initial_bpm_guess = 78.0
    t0 = time.time()

    log_dir = os.path.join(os.getcwd(), "log")
    log_path = next_log_path(log_dir)
    log_file = open(log_path, "w")
    log_file.write("time(s),bpm\n")
    print(f"Logging BPM to {log_path}")

    t0 = time.time()
    start_time = time.time()

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame_bgr = source.read()
            if not ok or frame_bgr is None:
                continue

            # Stabilize the face box before ROI crop and overlay drawing.
            last_bbox = face_stabilizer.update(frame_bgr, face_detector)
            face_detected = last_bbox is not None

            # ROI 처리는 매 프레임, 얼굴만 잘라서 72x72로 리사이즈해서 영상을 버퍼에 저장
            if face_detected:
                x1, y1, x2, y2 = last_bbox
                roi_bgr = frame_bgr[y1:y2, x1:x2]
                if roi_bgr.size > 0:
                    roi_bgr = cv2.resize(roi_bgr, (args.input_size, args.input_size), interpolation=cv2.INTER_AREA)
                    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                    frame_buffer.append(roi_rgb.astype(np.float32))
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if len(frame_buffer) == clip_len and (frame_idx % args.infer_stride == 0):
                clip = np.stack(frame_buffer, axis=0)
                pred_sig = infer_signal(model, args.model, clip, device, args)
                tail = min(args.infer_stride, len(pred_sig))
                for v in pred_sig[-tail:]:
                    bpm_signal_buffer.append(float(v))
                # 딥러닝이 바로 BPM내는 것이 아닌/ 딥러닝->rPPG 신호(시간파형)->신호처리로 BPM흐름
                bpm, filtered_sig, quality = estimate_bpm_from_signal(
                    list(bpm_signal_buffer),
                    args.fps,
                    diff_flag=args.diff_flag,
                    prev_bpm=last_bpm,
                    init_bpm_guess=initial_bpm_guess,
                )
                new_samples = []
                if filtered_sig is not None and tail > 0:
                    start = max(0, len(filtered_sig) - tail)
                    new_samples = [float(v) for v in filtered_sig[start:]]
                    for v in new_samples:
                        ppg_wave_buffer.append(v)
                    signal_std = np.std(filtered_sig[-tail:]) if tail > 0 else 0.0
                    if signal_std < 0.025:
                        quality = False
                    if new_samples:
                        arr = np.asarray(ppg_wave_buffer, dtype=np.float32)
                        mean = np.mean(arr)
                        max_dev = max(np.max(arr - mean), np.max(mean - arr))
                        scale = max(max_dev, 1e-3)
                        frame_rate = args.fps if args.fps > 0 else 30
                        bpm_value = last_bpm if (last_bpm is not None and last_bpm > 0) else initial_bpm_guess
                        samples_per_cycle = max(2, int(round(frame_rate * 60.0 / bpm_value)))
                        # 한 화면에 몇 주기가 보이게 할지 step을 조정 (현재는 5주기)
                        visible_samples = max(samples_per_cycle * 7, 1)
                        # BPM이 더 빨라지면 더 촘촘히, 느리면 더 넓게 보이도록
                        step = PANEL_W / visible_samples
                        clear_pixels = max(3, int(PANEL_W * 0.03))
                        stats = {
                            "mean": mean,
                            "scale": scale,
                            "vert_range": max(1, (PANEL_H // 2) - 12),
                            "y_mid": PANEL_H // 2,
                            "width": PANEL_W,
                            "height": PANEL_H,
                            "step": step,
                            "clear_pixels": clear_pixels,
                        }
                        for v in new_samples:
                            update_ekg_trace(ekg_canvas, ekg_background, v, ekg_state, stats)
                if bpm is not None and quality:
                    candidate_bpm = bpm
                    if kf_x is None:
                        kf_x = candidate_bpm
                    else:
                        kf_p = kf_p + kf_q
                        kf_k = kf_p / (kf_p + kf_r)
                        kf_x = kf_x + kf_k * (candidate_bpm - kf_x)
                        kf_p = (1 - kf_k) * kf_p
                    elapsed = time.time() - start_time
                    last_bpm = kf_x
                    # 이전 Arduino 확인용 UDP 송신 코드. 새 구조에서는 사용하지 않습니다.
                    # if last_bpm is not None:
                    #     payload = f"{last_bpm:.1f}"
                    #     old_sock.sendto(payload.encode(), (OLD_UDP_IP, OLD_UDP_PORT))
                    #     print("SEND:", last_bpm)

                    log_file.write(f"{elapsed:.2f},{last_bpm:.1f}\n")
                    log_file.flush()
                    warning_text = "CAUTION" if last_bpm < 50 or last_bpm > 130 else ""
                    payload = make_rppg_payload(last_bpm, warning_text, last_bbox, ppg_wave_buffer)
                    send_rppg_udp(udp_sock, payload, args.udp_host, args.udp_port)

            if not face_detected and args.send_udp:
                now = time.time()
                if now - last_no_face_udp_time >= udp_send_interval:
                    payload = make_rppg_payload(last_bpm, warning_text, None, ppg_wave_buffer)
                    send_rppg_udp(udp_sock, payload, args.udp_host, args.udp_port)
                    last_no_face_udp_time = now

            disp_fps = 1.0 / max(time.time() - t0, 1e-6)
            t0 = time.time()
            if args.show_debug_window:
                draw_face_panel(
                    frame_bgr,
                    ekg_canvas,
                    ekg_background,
                    last_bbox,
                    last_bpm,
                    disp_fps,
                    warning_text,
                )
                cv2.imshow("Real-time rPPG", frame_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            frame_idx += 1
    finally:
        source.close()
        if udp_sock is not None:
            udp_sock.close()
        log_file.close()
        if args.show_debug_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
