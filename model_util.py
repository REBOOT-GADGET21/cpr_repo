#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 모델 관련 유틸 함수 모음 """
import os
import sys
from pathlib import Path

import torch

# Make the repository root importable when this module is imported from src/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN

__all__ = ["load_checkpoint_flexible", "make_model"]

#*************************************************************************************************************  
# PyTorch 모델 가중치를 module. 유무와 상관없이 유연하게 로드
# 학습된 모델 가중치 로드해서 실시간 추론이 가능하게 함
def load_checkpoint_flexible(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    add_module = all(not k.startswith("module.") for k in state.keys())
    if add_module:
        state = {f"module.{k}": v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=True)
            return
        except RuntimeError:
            pass

    stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(stripped, strict=True)


#***************** 모델 생성 함수 *****************
def make_model(args, device):
    if args.model == "tscan":
        model = TSCAN(frame_depth=args.tscan_frame_depth, img_size=args.input_size)
        if device.type == "cuda":
            model = torch.nn.DataParallel(model, device_ids=[device.index if device.index is not None else 0])
        model = model.to(device).eval()
        load_checkpoint_flexible(model, args.checkpoint, device)
        return model

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=args.physnet_frame_num).to(device).eval()
    load_checkpoint_flexible(model, args.checkpoint, device)
    return model
