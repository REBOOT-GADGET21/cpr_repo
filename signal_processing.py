#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 함수 모음 """
import numpy as np
import scipy.signal
import torch

from scipy.signal import butter, find_peaks
from scipy.sparse import spdiags

__all__ = ["_detrend", "estimate_bpm_from_signal", "infer_signal"]

#*************************************************************************************************************
# 느린 추세를 제거 하는 함수 -> 상대적으로 고주파 성분을 잘 보이게 함
# numpy shape:      행렬의 차원과 각 차원의 크기를 나타내는 튜플
# numpy identity:   어떤 행렬을 곱해도 원래의 행렬이 나오는 행렬
# numpy ones:       모든 요소가 1인 배열
# lamda_value:      신호에서 제거할 느린 추세의 정도를 제어하는 매개변수/ 크면 더 강하게 제거하지만 신호 왜곡 가능성도 증가
def _detrend(input_signal, lambda_value=100):
    signal_length = input_signal.shape[0]
    h_mat = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    d_mat = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    return np.dot((h_mat - np.linalg.inv(h_mat + (lambda_value ** 2) * np.dot(d_mat.T, d_mat))), input_signal)


#*************************************************************************************************************
# x 이상인 가장 가까운 2의 거듭제곱을 반환/ FFT 계산에서 2의 거듭제곱 이면 더 빠름, 패딩하는 데 사용
def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


#*************************************************************************************************************
# 프레이간 변화량을 정규화된 차분으로 만드는 전처리 방식
# 절대 밝기보다 상대변화를 강조해서 조명/피부색 같은 상수 성분을 줄이고, 혈류로 인한 미세 변화를 강조하기 위함
# 입력: data -> np.ndarray, shape (n, h, w, c) (프레임 n장)
# 출력: np.ndarray, shape (n, h, w, c) (길이를 원래와 같게 맞추려고 padding 함)
def diff_normalize_data(data):
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
            data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7
        ) # 조명,피부색 절대값 같은 느린변화를 줄이고/ 혈류로 인한 미세 변화를 강조하기 위한 목적
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)     # 전체 std로 나누어 스케일 통일
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0) 
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


#*************************************************************************************************************
# 데이터를 표준화 (z-score)하는 함수
def standardized_data(data):
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


#*************************************************************************************************************
# 시간 신호(추정된 rPPG)를 주파수 분석해서 BPM을 추정하는 함수
# freq, pxx = periodogram(sig, fs)
# mask: 0.6 ~ 3.3 Hz
# bpm = peak_freq * 60
# 가장 강한 심박 후보 주파수를 bpm으로 변환 
def estimate_bpm_from_signal(ppg_signal, fs, diff_flag=True, prev_bpm=None, init_bpm_guess=78.0):
    # 최소 3초 이상 신호가 있어야 BPM 추정이 가능하도록 함 (짧은 신호는 잡음이 너무 많아서 정확한 추정이 어려움)
    if len(ppg_signal) < max(32, int(fs * 3)):
        return None, None, False
    sig = np.asarray(ppg_signal, dtype=np.float64)              # numpy 배열로 통일
    # 누적합(cumsum)으로 원래 파형 같은 형태로 복원하기 위헤/ _dtrenf로 저주파 드리프트 제거
    if diff_flag:
        sig = _detrend(np.cumsum(sig), 100)
    else:
        sig = _detrend(sig, 100)
    # 0.6Hz=36bpm, 3.3Hz=198bpm 범위의 밴드패스 필터링으로 심박이 나올 만한 대역의 신호만 남기기
    b, a = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype="bandpass")
    sig = scipy.signal.filtfilt(b, a, sig)                      # 위상 지연 없애기
    nfft = _next_power_of_2(sig.shape[0])
    freq, pxx = scipy.signal.periodogram(np.expand_dims(sig, 0), fs=fs, nfft=nfft, detrend=False)
    fmask = np.argwhere((freq >= 0.6) & (freq <= 3.3))
    if fmask.size == 0:
        return None, sig
    mask_freq = np.take(freq, fmask).reshape(-1)
    mask_pxx = np.take(pxx, fmask).reshape(-1)
    if mask_freq.size == 0:
        return None, sig, False

    mean_power = float(np.mean(mask_pxx))
    prominence = mean_power * 0.25
    height = mean_power * (1.0 if prev_bpm is None else 1.01)
    peaks, _ = find_peaks(mask_pxx, height=height, prominence=prominence)
    if peaks.size == 0:
        top_k = min(3, mask_freq.size)
        peak_candidates = np.argpartition(-mask_pxx, top_k - 1)[:top_k]
    else:
        peak_candidates = peaks[:3]
    candidates = []
    target_bpm = prev_bpm if prev_bpm is not None else init_bpm_guess
    for idx in peak_candidates:
        candidate_bpm = float(mask_freq[idx] * 60.0)
        power = float(mask_pxx[idx])
        score = power - abs(candidate_bpm - target_bpm) * 0.3
        candidates.append((score, candidate_bpm, power))
    best_score, bpm, peak_power = max(candidates, key=lambda x: x[0])
    return bpm, sig, True


#****************** 추론과 관련된 함수 ************************
def infer_signal(model, model_name, clip_rgb, device, args):
    clip_rgb = clip_rgb.astype(np.float32)
    # TSCAN은 motion branch(차분) + appearance branch(원본/정규화)같이 두 종류 특징을 같이 쓰는 방식
    if model_name == "tscan":
        diff = diff_normalize_data(clip_rgb)
        raw = standardized_data(clip_rgb)
        data = np.concatenate([diff, raw], axis=-1)
        data = np.transpose(data, (0, 3, 1, 2))
        if data.shape[0] % args.tscan_frame_depth != 0: # frame_depth 단위로 처리하기 위해 길이를 맞추는 방식
            valid = (data.shape[0] // args.tscan_frame_depth) * args.tscan_frame_depth
            data = data[:valid]
        x = torch.from_numpy(data).to(device, dtype=torch.float32)
        with torch.no_grad():
            pred = model(x).squeeze(-1).detach().cpu().numpy()
        return pred

    # physNet은 차분 정규화된 데이터만 쓰는 방식/ 배치,채널,높이,너비 형태의 3D conv스타일 입력을 받는 경우가 많아서 이렇게 진행
    data = diff_normalize_data(clip_rgb)
    data = np.transpose(data, (3, 0, 1, 2))
    x = torch.from_numpy(data).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        pred, _, _, _ = model(x)
    return pred.squeeze(0).detach().cpu().numpy()
