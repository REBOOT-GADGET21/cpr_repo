#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI 클래스 - Medical Monitor Style v3
레이아웃: [오른쪽 ECG 파형 패널] | [왼쪽 카메라 + 수치 패널]
파형: polyline 누적 방식으로 매끄럽게 렌더링
"""

import os
import glob
import time

import cv2
import numpy as np

__all__ = [
    "make_panel_background",
    "update_ekg_trace",
    "draw_face_panel",
    "next_log_path",
    "find_face_bbox",
]

# ══════════════════════════════════════════════════════════════════════════════
#  컬러 팔레트 (BGR)
# ══════════════════════════════════════════════════════════════════════════════
C_BG            = ( 10,  14,  22)   # 배경: 짙은 네이비
C_GRID_MAJOR    = ( 28,  42,  65)   # 굵은 격자선
C_GRID_MINOR    = ( 16,  24,  38)   # 얇은 격자선
C_BASELINE      = ( 45,  65, 100)   # 중앙 기준선
C_WAVE_CORE     = (  0,  40, 255)   # 파형 핵심색 (빨강)
C_WAVE_MID      = (  0,  20, 180)   # 파형 중간 glow
C_WAVE_OUTER    = (  0,  10,  90)   # 파형 바깥 glow
C_ACCENT        = (  0, 210, 180)   # 강조색 (청록)
C_ACCENT2       = (  0, 180, 255)   # 두 번째 강조 (하늘)
C_DIM           = ( 60,  80, 100)   # 흐린 텍스트
C_WHITE         = (220, 230, 240)   # 메인 텍스트
C_WARN          = ( 20, 160, 240)   # 경고색
C_PANEL_LINE    = ( 30,  50,  80)   # 패널 구분선
C_HR_NORMAL     = ( 20, 235,  90)   # 정상 심박 (그린)
C_HR_HIGH       = (  0, 120, 255)   # 빠른 심박
C_HR_LOW        = (  0, 210, 255)   # 느린 심박
C_BORDER        = (  0, 160, 120)   # 외곽 테두리
C_CAM_OVERLAY   = (  8,  12,  20)   # 카메라 패널 오버레이
C_FACE_BOX      = (  0, 220, 160)   # 얼굴 감지 박스 (청록)

# ══════════════════════════════════════════════════════════════════════════════
#  레이아웃 상수
# ══════════════════════════════════════════════════════════════════════════════
HEADER_H   = 40
BPM_H      = 100
FOOTER_H   = 28
MARGIN     = 10
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SM    = cv2.FONT_HERSHEY_SIMPLEX
FONT_PLAIN = cv2.FONT_HERSHEY_PLAIN


# ══════════════════════════════════════════════════════════════════════════════
#  헬퍼 함수
# ══════════════════════════════════════════════════════════════════════════════
def _draw_rounded_rect(img, pt1, pt2, color, thickness=1, r=6):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness, cv2.LINE_AA)


def _fill_rect_alpha(img, y1, y2, color_bgr, alpha=0.80, x1=0, x2=None):
    if x2 is None:
        x2 = img.shape[1]
    y1, y2 = max(0, y1), min(img.shape[0], y2)
    x1, x2 = max(0, x1), min(img.shape[1], x2)
    if y2 <= y1 or x2 <= x1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color_bgr, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _put_text_shadow(img, text, org, font, scale, color, thickness, shadow_offset=2):
    sx, sy = org[0] + shadow_offset, org[1] + shadow_offset
    shadow_color = tuple(max(0, int(c * 0.12)) for c in color)
    cv2.putText(img, text, (sx, sy), font, scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org,      font, scale, color,         thickness,     cv2.LINE_AA)


def _draw_polyline_glow(canvas, pts, color_core, color_mid, color_outer):
    """
    3-레이어 glow polyline.
    cv2.polylines 단일 호출 → 연결점 아티팩트 없음.
    """
    if len(pts) < 2:
        return
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [arr], False, color_outer, 2, cv2.LINE_AA)
    # cv2.polylines(canvas, [arr], False, color_mid,   2, cv2.LINE_AA)
    cv2.polylines(canvas, [arr], False, color_core,  1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  ECG 패널 배경
# ══════════════════════════════════════════════════════════════════════════════
def make_panel_background(width, height):
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = C_BG

    minor_step_x = max(1, width  // 40)
    minor_step_y = max(1, height // 20)
    for px in range(0, width,  minor_step_x):
        cv2.line(bg, (px, 0), (px, height - 1), C_GRID_MINOR, 1)
    for py in range(0, height, minor_step_y):
        cv2.line(bg, (0, py), (width - 1, py),  C_GRID_MINOR, 1)

    major_step_x = minor_step_x * 5
    major_step_y = minor_step_y * 5
    for px in range(0, width,  major_step_x):
        cv2.line(bg, (px, 0), (px, height - 1), C_GRID_MAJOR, 1)
    for py in range(0, height, major_step_y):
        cv2.line(bg, (0, py), (width - 1, py),  C_GRID_MAJOR, 1)

    y_mid = height // 2
    cv2.line(bg, (0, y_mid), (width - 1, y_mid), C_BASELINE, 1)
    return bg


# ══════════════════════════════════════════════════════════════════════════════
#  파형 업데이트 — polyline 누적 방식 (매끄러움)
# ══════════════════════════════════════════════════════════════════════════════
def update_ekg_trace(canvas, background, value, state, stats):
    """
    변경 사항:
      state["point_buf"] 에 (x, y) 점을 누적하고
      매 프레임마다 cv2.polylines 로 한 번에 렌더링.
      → cv2.line 연결점 아티팩트 완전 제거.
      wrap(주기 전환) 로직은 원본과 동일하게 유지.
    """
    width      = stats["width"]
    height     = stats["height"]
    x_float    = state["pos"]
    x_idx      = int(x_float) % width
    clear_len  = stats.get("clear_pixels", 5)

    # 지우기 (원본과 동일)
    for dx in range(1, clear_len + 1):
        canvas[:, (x_idx + dx) % width] = background[:, (x_idx + dx) % width]
    canvas[:, x_idx] = background[:, x_idx]

    # y 좌표 계산
    mean       = stats["mean"]
    scale      = stats["scale"]
    vert_range = stats["vert_range"]
    y_mid      = stats["y_mid"]
    normed     = np.clip((value - mean) / scale, -1.0, 1.0)
    y          = int(np.clip(y_mid - normed * vert_range, 0, height - 1))

    # 다음 위치 & wrap 판단 (원본과 동일)
    next_pos     = x_float + stats["step"]
    wrapped      = next_pos >= width
    next_pos_mod = next_pos % width

    # point_buf 초기화
    if "point_buf" not in state:
        state["point_buf"] = []

    if wrapped:
        state["point_buf"] = []
        state["prev_point"] = None
    else:
        state["point_buf"].append((x_idx, y))
        buf = state["point_buf"]

        if len(buf) >= 2:
            xs      = [p[0] for p in buf]
            x_start = xs[0]
            x_end   = xs[-1]
            if x_end >= x_start:
                canvas[:, x_start:x_end + 1] = background[:, x_start:x_end + 1]
            _draw_polyline_glow(canvas, buf, C_WAVE_CORE, C_WAVE_MID, C_WAVE_OUTER)
        elif len(buf) == 1:
            cv2.circle(canvas, buf[0], 1, C_WAVE_CORE, -1, cv2.LINE_AA)

    state["pos"]         = next_pos_mod
    state["total_drawn"] += 1


# ══════════════════════════════════════════════════════════════════════════════
#  얼굴 옆 파형 HUD 오버레이
# ══════════════════════════════════════════════════════════════════════════════
def draw_face_panel(frame_bgr, canvas, background, face_bbox, last_bpm, disp_fps, warning_text, gap=20):
    if face_bbox is None:
        return

    fh, fw = frame_bgr.shape[:2]
    pw, ph = canvas.shape[1], canvas.shape[0]
    x1f, y1f, x2f, y2f = face_bbox

    # 얼굴 박스 오른쪽 20px 우선 배치
    x1 = x2f + gap
    if x1 >= fw:
        return
    x1 = int(max(0, x1))

    y1 = int(np.clip(y1f - 10, 0, max(0, fh - ph)))
    x2 = min(fw, x1 + pw)
    y2 = min(fh, y1 + ph)
    vis_w = x2 - x1
    vis_h = y2 - y1
    if vis_w <= 0 or vis_h <= 0:
        return

    roi = frame_bgr[y1:y2, x1:x2]
    wave_mask = np.any(canvas[:vis_h, :vis_w] != background[:vis_h, :vis_w], axis=2)
    roi[wave_mask] = canvas[:vis_h, :vis_w][wave_mask]
    frame_bgr[y1:y2, x1:x2] = roi

    if last_bpm is not None:
        if last_bpm > 100 or last_bpm < 50:
            bpm_color, status_text = C_HR_HIGH, "CAUTION"
        else:
            bpm_color, status_text = C_HR_NORMAL, "STABLE"

        _put_text_shadow(frame_bgr, "HEART RATE", (x1, y1 + 8), FONT_SM, 0.34, C_DIM, 1)
        bpm_str = f"{int(last_bpm)}"
        bpm_dec = int(last_bpm * 10) % 10
        _put_text_shadow(frame_bgr, bpm_str, (x1, y1 + 56), FONT, 1.7, bpm_color, 2, shadow_offset=2)
        (big_w, _), _ = cv2.getTextSize(bpm_str, FONT, 2.0, 3)
        _put_text_shadow(frame_bgr, f".{bpm_dec}", (x1 + big_w + 2, y1 + 45), FONT_SM, 0.58, bpm_color, 1)
        _put_text_shadow(frame_bgr, "bpm", (x1 + big_w + 2, y1 + 66), FONT_SM, 0.36, C_DIM, 1)
        _put_text_shadow(frame_bgr, status_text, (x2 - 78, y1 + 10), FONT_SM, 0.40, bpm_color, 1)
    else:
        _put_text_shadow(frame_bgr, "HEART RATE", (x1, y1 + 8), FONT_SM, 0.34, C_DIM, 1)

    if warning_text:
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        center_x = x1 + max(0, (vis_w - text_size[0]) // 2)
        cv2.putText(frame_bgr, warning_text, (center_x, y1 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WARN, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  로그 경로
# ══════════════════════════════════════════════════════════════════════════════
def next_log_path(base_dir=".", base_name="bpm_log", ext="txt"):
    os.makedirs(base_dir, exist_ok=True)
    pattern = os.path.join(base_dir, f"{base_name}_*.{ext}")
    highest = 0
    for entry in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(entry))[0]
        if "_" not in name:
            continue
        suffix = name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    candidate = f"{base_name}_{highest + 1}.{ext}"
    return os.path.join(base_dir, candidate)


# ══════════════════════════════════════════════════════════════════════════════
#  얼굴 감지 ROI
# ══════════════════════════════════════════════════════════════════════════════
def find_face_bbox(frame_bgr, detector, prev_bbox=None, scale=1.5):
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        cx   = x + w / 2.0
        cy   = y + h / 2.0
        side = max(w, h) * scale
        x1   = int(max(0, cx - side / 2.0))
        y1   = int(max(0, cy - side / 2.0))
        x2   = int(min(frame_bgr.shape[1], cx + side / 2.0))
        y2   = int(min(frame_bgr.shape[0], cy + side / 2.0))
        return (x1, y1, x2, y2)
    return prev_bbox
