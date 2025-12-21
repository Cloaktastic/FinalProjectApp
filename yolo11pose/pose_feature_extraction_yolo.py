# -*- coding: utf-8 -*-
"""
Module Person 1 - Pose & Feature Extraction (YOLOv11-Pose)
Trích xuất đặc trưng từ video event (buffer 5–10 giây) bằng YOLOv11-Pose

Mục tiêu:
- Thay OpenPose bằng YOLOv11-Pose để tăng tốc và dễ tích hợp với pipeline Ultralytics
- Vẫn giữ cơ chế giống code OpenPose cũ:
  + Tính vận tốc rơi (v_peak) dựa trên chuyển động hông theo trục y
  + Tính thay đổi góc thân (delta_angle) dựa trên vector "thân người"
  + Tính thời gian nằm im (immobile_duration) sau thời điểm rơi mạnh nhất

Ghi chú quan trọng:
- YOLOv11-Pose trả về keypoints theo chuẩn COCO 17 điểm, KHÔNG có "MID_HIP" và "NECK" như BODY_25.
- Vì vậy, ta xấp xỉ:
  + HIP_CENTER = trung điểm (left_hip, right_hip)
  + NECK = trung điểm (left_shoulder, right_shoulder)
- Với camera thông thường, vector (NECK - HIP_CENTER) đủ ổn cho góc thân.

Tối ưu hiệu năng:
- Chỉ chạy pose trên clip event ngắn (5–10 giây), không chạy liên tục realtime
- frame_step để skip frame (giảm số frame phải infer)
- batch_size để infer theo lô (giảm overhead gọi model nhiều lần)
- giảm imgsz (ví dụ 320/416) để nhanh hơn (đổi lại keypoints có thể kém ổn định)
- half=True nếu có GPU (FP16) để tăng tốc
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import cv2
from scipy.signal import butter, filtfilt

# Ultralytics
from ultralytics import YOLO


# =========================
# Mapping keypoints COCO-17
# =========================
# Thứ tự COCO 17 keypoints (phổ biến):
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder,
# 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist,
# 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee,
# 15 left_ankle, 16 right_ankle

L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12


# =========================
# Bộ lọc làm mượt
# =========================
def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Bộ lọc thông thấp Butterworth để làm mượt dữ liệu theo thời gian.

    Args:
        data: Dữ liệu đầu vào (1D)
        cutoff: Tần số cắt (Hz)
        fs: Tần số lấy mẫu (Hz)
        order: Bậc filter

    Returns:
        Dữ liệu đã lọc (1D)
    """
    # Nếu dữ liệu quá ngắn thì khỏi lọc (filtfilt cần tối thiểu vài điểm)
    if data is None or len(data) < max(10, order * 3):
        return data

    nyq = 0.5 * fs
    normal_cutoff = cutoff / max(nyq, 1e-6)
    normal_cutoff = min(max(normal_cutoff, 1e-6), 0.99)

    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


# =========================
# Đọc video + lấy FPS
# =========================
def get_video_fps(video_path: str, default_fps: int = 30) -> int:
    """
    Đọc FPS từ video. Nếu đọc lỗi thì fallback về default_fps.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-3:
        return int(default_fps)
    return int(round(fps))


def iter_video_frames(
    video_path: str,
    frame_step: int = 2,
    max_frames: Optional[int] = None,
    resize_width: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Đọc frame từ video theo bước nhảy frame_step (skip bớt frame để tăng tốc).
    Có thể resize về resize_width để giảm compute.

    Returns:
        frames: danh sách frame BGR (numpy)
        frame_ids: danh sách chỉ số frame tương ứng trong video gốc
    """
    assert frame_step >= 1, "frame_step phải >= 1"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    frames: List[np.ndarray] = []
    frame_ids: List[int] = []
    fid = 0
    kept = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Chỉ lấy 1 frame mỗi frame_step
        if fid % frame_step == 0:
            if resize_width is not None and resize_width > 0:
                h, w = frame.shape[:2]
                if w != resize_width:
                    new_h = int(h * (resize_width / w))
                    frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

            frames.append(frame)
            frame_ids.append(fid)
            kept += 1

            if max_frames is not None and kept >= max_frames:
                break

        fid += 1

    cap.release()
    return frames, frame_ids


# =========================
# Chạy YOLOv11-Pose
# =========================
def run_yolo_pose_on_frames(
    model: YOLO,
    frames: List[np.ndarray],
    imgsz: int = 416,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Optional[str] = None,
    half: bool = True,
    batch_size: int = 16,
) -> List:
    """
    Chạy YOLO Pose trên danh sách frames theo batch để giảm overhead.

    Returns:
        results_list: list Results (mỗi phần tử tương ứng 1 frame)
    """
    results_all = []
    n = len(frames)
    if n == 0:
        return results_all

    # Chạy theo lô
    for s in range(0, n, batch_size):
        batch = frames[s : s + batch_size]
        # Ultralytics cho phép truyền list numpy images trực tiếp
        batch_results = model.predict(
            source=batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            verbose=False,
        )
        results_all.extend(batch_results)

    return results_all


def select_main_person_keypoints(result, min_kpt_score: float = 0.2) -> Optional[np.ndarray]:
    """
    Chọn bộ keypoints của 1 người chính trong frame.

    Chiến lược chọn:
    - Nếu có nhiều người: chọn người có bbox area lớn nhất (thường là người gần camera nhất)
    - Nếu không có người hoặc keypoints rỗng: return None

    Returns:
        kpts: ndarray shape (17, 3) với (x, y, score), hoặc None
    """
    # result.keypoints.xy: (num_people, 17, 2)
    # result.keypoints.conf: (num_people, 17) hoặc None tùy version
    if result is None or result.keypoints is None:
        return None

    kpt_xy = getattr(result.keypoints, "xy", None)
    kpt_conf = getattr(result.keypoints, "conf", None)
    boxes = getattr(result, "boxes", None)

    if kpt_xy is None or len(kpt_xy) == 0:
        return None

    num_people = kpt_xy.shape[0]

    # Nếu không có bbox thì chọn người 0
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        idx = 0
    else:
        # Chọn bbox area lớn nhất
        xyxy = boxes.xyxy.cpu().numpy()  # (num_people, 4)
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(np.argmax(areas))

    xy = kpt_xy[idx].cpu().numpy()  # (17, 2)

    # Lấy score cho từng keypoint (nếu có)
    if kpt_conf is not None:
        sc = kpt_conf[idx].cpu().numpy()  # (17,)
    else:
        # Nếu model/version không cung cấp conf keypoint, set tạm 1.0
        sc = np.ones((xy.shape[0],), dtype=np.float32)

    kpts = np.concatenate([xy, sc[:, None]], axis=1)  # (17, 3)

    # Nếu đa số keypoints quá thấp thì coi như không hợp lệ
    valid = np.sum(kpts[:, 2] >= min_kpt_score)
    if valid < 4:
        return None

    return kpts


def keypoints_to_trunk_points(
    kpts17: np.ndarray,
    min_score: float = 0.2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Từ keypoints COCO-17, suy ra 2 điểm đại diện cho thân người:
    - hip_center: trung điểm 2 hông
    - neck_proxy: trung điểm 2 vai (xấp xỉ cổ)

    Returns:
        hip_center: (2,) hoặc None
        neck_proxy: (2,) hoặc None
    """
    if kpts17 is None or kpts17.shape != (17, 3):
        return None, None

    # Lấy 2 vai
    ls = kpts17[L_SHOULDER]
    rs = kpts17[R_SHOULDER]
    # Lấy 2 hông
    lh = kpts17[L_HIP]
    rh = kpts17[R_HIP]

    # Kiểm tra score tối thiểu
    if ls[2] < min_score or rs[2] < min_score:
        neck_proxy = None
    else:
        neck_proxy = (ls[:2] + rs[:2]) / 2.0

    if lh[2] < min_score or rh[2] < min_score:
        hip_center = None
    else:
        hip_center = (lh[:2] + rh[:2]) / 2.0

    return hip_center, neck_proxy


# =========================
# Trích feature giống bản OpenPose
# =========================
def extract_features_from_trunk_series(
    hip_series: np.ndarray,    # (T, 2) chứa (x, y) có thể NaN
    neck_series: np.ndarray,   # (T, 2) chứa (x, y) có thể NaN
    fps: int,
    use_height_normalization: bool = True,
) -> Dict[str, float]:
    """
    Tính 3 feature: v_peak, delta_angle, immobile_duration
    giống logic code OpenPose cũ nhưng thay input là chuỗi hip/neck.

    Args:
        hip_series: (T, 2)
        neck_series: (T, 2)
        fps: FPS hiệu dụng sau khi đã skip frame (đã xử lý ở hàm gọi)
        use_height_normalization: có chuẩn hóa theo "chiều cao gần đúng" không

    Returns:
        dict features
    """
    T = hip_series.shape[0]
    dt = 1.0 / max(fps, 1)

    # Lấy y của hông
    y_hip = hip_series[:, 1].astype(np.float32)  # (T,)

    # Tính trunk length (khoảng cách neck - hip)
    trunk_len = np.linalg.norm(neck_series - hip_series, axis=1).astype(np.float32)  # (T,)
    approx_height = trunk_len * 2.0

    # Chuẩn hóa y (giảm phụ thuộc khoảng cách tới camera)
    if use_height_normalization:
        y_norm = y_hip / (approx_height + 1e-6)
    else:
        y_norm = y_hip

    # Thay NaN bằng nội suy đơn giản để filter/gradient không vỡ
    def nan_interpolate(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        n = len(x)
        mask = np.isfinite(x)
        if np.sum(mask) < 3:
            return np.full_like(x, np.nan)
        idx = np.arange(n)
        x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
        return x

    y_work = nan_interpolate(y_norm)
    if not np.isfinite(y_work).any():
        return {"v_peak": float("nan"), "delta_angle": float("nan"), "immobile_duration": float("nan")}

    # Làm mượt y
    y_filt = butter_lowpass_filter(y_work, cutoff=5.0, fs=fps, order=4)

    # Tính vận tốc theo y
    v = np.gradient(y_filt, dt)
    v_filt = butter_lowpass_filter(v, cutoff=10.0, fs=fps, order=4)
    v_peak = float(np.nanmax(np.abs(v_filt)))  # lấy độ lớn, tránh dấu

    # Tính góc thân: vector neck - hip
    vec = (neck_series - hip_series).astype(np.float32)  # (T,2)
    # Tính góc so với trục thẳng đứng (y tăng xuống dưới)
    # atan2(x, -y) giống bản cũ
    angles = np.degrees(np.arctan2(vec[:, 0], -vec[:, 1])).astype(np.float32)

    angles_work = nan_interpolate(angles)
    angles_filt = butter_lowpass_filter(angles_work, cutoff=5.0, fs=fps, order=4)

    # idx_peak theo v_filt
    idx_peak = int(np.nanargmax(np.abs(v_filt)))

    # Cửa sổ 0.5 giây trước/sau
    win = int(0.5 * fps)
    before = angles_filt[max(0, idx_peak - win) : idx_peak]
    after = angles_filt[idx_peak : min(T, idx_peak + win)]

    angle_before = float(np.nanmean(before)) if len(before) > 0 else float("nan")
    angle_after = float(np.nanmean(after)) if len(after) > 0 else float("nan")
    delta_angle = float(abs(angle_after - angle_before))

    # Thời gian bất động sau đỉnh
    thr_pos = 0.01  # ngưỡng thay đổi vị trí nhỏ (trên y đã normalize)
    thr_ang = 3.0   # ngưỡng thay đổi góc nhỏ (độ)

    immobile_frames = 0
    for t in range(idx_peak, T - 1):
        dy = abs(y_filt[t + 1] - y_filt[t])
        da = abs(angles_filt[t + 1] - angles_filt[t])
        if dy < thr_pos and da < thr_ang:
            immobile_frames += 1

    immobile_duration = float(immobile_frames * dt)

    return {
        "v_peak": v_peak,
        "delta_angle": delta_angle,
        "immobile_duration": immobile_duration,
    }


# =========================
# Hàm cấp cao: video -> features
# =========================
def extract_features_yolov11_pose(
    video_path: str,
    pose_model_path: str = "yolo11n-pose.pt",
    device: Optional[str] = None,
    imgsz: int = 416,
    conf: float = 0.25,
    iou: float = 0.7,
    half: bool = True,
    frame_step: int = 2,
    batch_size: int = 16,
    resize_width: Optional[int] = None,
    min_kpt_score: float = 0.2,
) -> Dict[str, float]:
    """
    Pipeline tổng:
    1) Đọc frame từ video theo frame_step (skip để nhanh)
    2) Chạy YOLOv11-Pose theo batch
    3) Lấy hip_center và neck_proxy theo từng frame
    4) Tính features (v_peak, delta_angle, immobile_duration)

    Args:
        video_path: đường dẫn video event (5–10s)
        pose_model_path: path model YOLOv11 pose (vd: yolo11n-pose.pt)
        device: "cuda:0" hoặc "cpu" hoặc None (ultralytics tự chọn)
        imgsz: kích thước input cho YOLO (320/416/640). Nhỏ hơn => nhanh hơn
        conf, iou: ngưỡng detect
        half: FP16 nếu có GPU (không có GPU thì nên đặt False)
        frame_step: lấy 1 frame mỗi frame_step frame (2 => ~15fps nếu video 30fps)
        batch_size: infer theo lô
        resize_width: resize frame trước khi infer để giảm compute (tùy chọn)
        min_kpt_score: ngưỡng score tối thiểu của keypoint để tin dùng

    Returns:
        dict features
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    # 1) FPS gốc và FPS hiệu dụng sau khi skip frame
    fps_src = get_video_fps(video_path, default_fps=30)
    fps_eff = max(1, int(round(fps_src / max(frame_step, 1))))

    # 2) Đọc frames (skip + optional resize)
    frames, frame_ids = iter_video_frames(
        video_path=video_path,
        frame_step=frame_step,
        max_frames=None,
        resize_width=resize_width,
    )

    if len(frames) == 0:
        return {"v_peak": float("nan"), "delta_angle": float("nan"), "immobile_duration": float("nan")}

    # 3) Load model pose
    model = YOLO(pose_model_path)

    # Nếu chạy CPU thì half nên tắt để tránh lỗi/không hiệu quả
    if device is not None and "cpu" in device.lower():
        half = False

    # 4) Infer pose theo batch
    results_list = run_yolo_pose_on_frames(
        model=model,
        frames=frames,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        half=half,
        batch_size=batch_size,
    )

    # 5) Tạo chuỗi hip/neck (T,2), missing -> NaN
    T = len(results_list)
    hip_series = np.full((T, 2), np.nan, dtype=np.float32)
    neck_series = np.full((T, 2), np.nan, dtype=np.float32)

    for t, res in enumerate(results_list):
        kpts = select_main_person_keypoints(res, min_kpt_score=min_kpt_score)
        if kpts is None:
            continue

        hip, neck = keypoints_to_trunk_points(kpts, min_score=min_kpt_score)
        if hip is not None:
            hip_series[t] = hip
        if neck is not None:
            neck_series[t] = neck

    # 6) Nếu neck/hip bị thiếu nhiều, features sẽ kém tin cậy
    #    Ở đây ta vẫn cố tính, các bước nội suy sẽ giúp ổn hơn
    feats = extract_features_from_trunk_series(
        hip_series=hip_series,
        neck_series=neck_series,
        fps=fps_eff,
        use_height_normalization=True,
    )

    return feats


# =========================
# Demo chạy thử
# =========================
if __name__ == "__main__":
    # Video event (buffer té ngã 5–10s)
    video_path = "yolo11pose/tmp_event_cut.mp4"

    # Model pose (bạn tải/đặt cùng thư mục hoặc đường dẫn đầy đủ)
    # Ví dụ: "yolo11n-pose.pt" (nhẹ nhất) hoặc "yolo11s-pose.pt"
    pose_model_path = "yolo11n-pose.pt"

    # Gợi ý cấu hình:
    # - Nếu có GPU: device="cuda:0", half=True
    # - Nếu CPU: device="cpu", half=False, tăng frame_step và giảm imgsz
    device = "cpu"

    features = extract_features_yolov11_pose(
        video_path=video_path,
        pose_model_path=pose_model_path,
        device=device,
        imgsz=416,           # giảm xuống 320 nếu muốn nhanh hơn nữa
        conf=0.25,
        iou=0.7,
        half=False,          # CPU -> False
        frame_step=2,        # 30fps -> xử lý ~15fps
        batch_size=8,        # CPU batch nhỏ thôi
        resize_width=640,    # giảm resolution trước khi infer (tùy máy)
        min_kpt_score=0.2,
    )

    print("Đặc trưng trích xuất được:")
    print(features)
    # Kết quả:
    # {
    #   "v_peak": ...,
    #   "delta_angle": ...,
    #   "immobile_duration": ...
    # }
