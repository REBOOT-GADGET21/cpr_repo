#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
FrameSource 클래스
RealSense 카메라 입력 
"""

import cv2
import numpy as np

__all__ = ["FrameSource"]

class FrameSource:
    def __init__(self, use_realsense, width, height, fps, camera_index):
        self.use_realsense = use_realsense
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.pipeline = None
        self.cap = None
        self.rs = None

    def open(self):
        if self.use_realsense:
            try:
                import pyrealsense2 as rs

                self.rs = rs
                self.pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self.pipeline.start(cfg)
                return
            except Exception as exc:
                print(f"[WARN] RealSense open failed ({exc}). Falling back to cv2 camera.")

        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera stream.")

    def read(self):
        if self.pipeline is not None:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            img = np.asanyarray(color_frame.get_data())
            return True, img
        ok, img = self.cap.read()
        return ok, img

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.cap is not None:
            self.cap.release()
