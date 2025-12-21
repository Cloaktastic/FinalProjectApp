# -*- coding: utf-8 -*-
"""
visualize_event_pose_yolo.py

Module visualize pose cho video event (buffer 5–10s) bằng YOLOv11-Pose.
Mục tiêu:
- Xuất video output có overlay keypoints + skeleton
- Vẽ vector thân (hip_center -> neck_proxy)
- Hiển thị góc thân và vận tốc (theo trục y của hông) theo thời gian
- (Tuỳ chọn) hiển thị trạng thái "bất động" dựa trên ngưỡng dy/da giống module features

Lưu ý:
- YOLOv11-Pose dùng COCO-17 keypoints, không có MID_HIP/NECK như OpenPose BODY_25
- Ta xấp xỉ:
  + HIP_CENTER = trung điểm (left_hip, right_hip)
  + NECK_PROXY = trung điểm (left_shoulder, right_shoulder)

Yêu cầu:
pip install ultralytics opencv-python numpy scipy
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import os
import math

import numpy as np
import cv2
from scipy.signal import butter, filtfilt
from ultralytics import YOLO


# =========================
# Mapping keypoints COCO-17
# =========================
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12

# Các cặp khớp để vẽ skeleton (COCO 17)
COCO_SKELETON_EDGES = [
    (5, 6),    # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (11, 12),          # hips
    (5, 11), (6, 12),  # torso sides
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16) # right leg
]


# =========================
# Helper: đọc FPS
# =========================
def get_video_fps(video_path: str, default_fps: int = 30) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-3:
        return int(default_fps)
    return int(round(fps))


# =========================
# Helper: filter làm mượt
# =========================
def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    if data is None or len(data) < max(10, order * 3):
        return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / max(nyq, 1e-6)
    normal_cutoff = min(max(normal_cutoff, 1e-6), 0.99)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def nan_interpolate(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    n = len(x)
    mask = np.isfinite(x)
    if np.sum(mask) < 3:
        return np.full_like(x, np.nan)
    idx = np.arange(n)
    x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return x


# =========================
# Lấy kpts người chính
# =========================
def select_main_person_kpts(result, min_kpt_score: float = 0.2) -> Optional[np.ndarray]:
    """
    Chọn 1 người chính trong frame dựa trên bbox area lớn nhất.
    Return: (17,3) [x,y,score] hoặc None
    """
    if result is None or result.keypoints is None:
        return None

    kpt_xy = getattr(result.keypoints, "xy", None)
    kpt_conf = getattr(result.keypoints, "conf", None)
    boxes = getattr(result, "boxes", None)

    if kpt_xy is None or len(kpt_xy) == 0:
        return None

    num_people = kpt_xy.shape[0]

    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        idx = 0
    else:
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(np.argmax(areas))

    xy = kpt_xy[idx].cpu().numpy()  # (17,2)
    if kpt_conf is not None:
        sc = kpt_conf[idx].cpu().numpy()  # (17,)
    else:
        sc = np.ones((xy.shape[0],), dtype=np.float32)

    kpts = np.concatenate([xy, sc[:, None]], axis=1)

    # Nếu quá ít điểm có score ổn => bỏ
    if np.sum(kpts[:, 2] >= min_kpt_score) < 4:
        return None

    return kpts


def trunk_points_from_kpts(kpts17: np.ndarray, min_score: float = 0.2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Suy ra hip_center và neck_proxy từ COCO-17.
    Return: (hip_center(2,), neck_proxy(2,)) hoặc (None, None) nếu thiếu.
    """
    if kpts17 is None or kpts17.shape != (17, 3):
        return None, None

    ls = kpts17[L_SHOULDER]
    rs = kpts17[R_SHOULDER]
    lh = kpts17[L_HIP]
    rh = kpts17[R_HIP]

    neck = None
    hip = None

    if ls[2] >= min_score and rs[2] >= min_score:
        neck = (ls[:2] + rs[:2]) / 2.0

    if lh[2] >= min_score and rh[2] >= min_score:
        hip = (lh[:2] + rh[:2]) / 2.0

    return hip, neck


# =========================
# Vẽ overlay
# =========================
def draw_keypoints_and_skeleton(
    frame: np.ndarray,
    kpts17: np.ndarray,
    min_kpt_score: float = 0.2
) -> np.ndarray:
    """
    Vẽ keypoints và skeleton lên frame.
    """
    out = frame

    # Vẽ skeleton
    for a, b in COCO_SKELETON_EDGES:
        if kpts17[a, 2] >= min_kpt_score and kpts17[b, 2] >= min_kpt_score:
            ax, ay = int(kpts17[a, 0]), int(kpts17[a, 1])
            bx, by = int(kpts17[b, 0]), int(kpts17[b, 1])
            cv2.line(out, (ax, ay), (bx, by), (0, 255, 0), 2)

    # Vẽ keypoints
    for i in range(17):
        if kpts17[i, 2] >= min_kpt_score:
            x, y = int(kpts17[i, 0]), int(kpts17[i, 1])
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)

    return out


def compute_angle_deg(hip: np.ndarray, neck: np.ndarray) -> float:
    """
    Tính góc thân so với trục thẳng đứng.
    Công thức giống module features: atan2(dx, -dy), y tăng xuống dưới.
    """
    vec = neck - hip  # (dx, dy)
    ang = math.degrees(math.atan2(vec[0], -vec[1]))
    return float(ang)


def put_text_block(frame: np.ndarray, lines: List[str], x: int = 10, y: int = 30) -> None:
    """
    Ghi nhiều dòng text lên frame.
    """
    dy = 22
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, s, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


# =========================
# Main: tạo video visualize
# =========================
def visualize_event_pose(
    video_path: str,
    output_path: str,
    pose_model_path: str = "yolo11n-pose.pt",
    device: Optional[str] = None,
    imgsz: int = 416,
    conf: float = 0.25,
    iou: float = 0.7,
    half: bool = True,
    frame_step: int = 1,
    min_kpt_score: float = 0.2,
    # Ngưỡng "bất động" giống module features (tính trên tín hiệu đã normalize + filter)
    thr_pos: float = 0.01,
    thr_ang: float = 3.0,
) -> None:
    """
    Tạo video output overlay pose.

    frame_step:
    - Nếu muốn output video giữ nguyên FPS và không bị giật: để 1
    - Nếu muốn nhanh hơn và chấp nhận output giật hơn: để 2 hoặc 3

    Lưu ý:
    - Nếu frame_step > 1, video output sẽ có ít frame hơn => thời gian chạy nhanh hơn
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    fps_src = get_video_fps(video_path, default_fps=30)
    fps_eff = max(1, int(round(fps_src / max(frame_step, 1))))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter: dùng mp4v để dễ chạy trên Windows
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_eff, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được video output: {output_path}")

    # Load model
    model = YOLO(pose_model_path)
    if device is not None and "cpu" in device.lower():
        half = False

    # Ta sẽ lưu hip/necks theo thời gian để tính vận tốc/góc (và trạng thái bất động)
    hip_list: List[Optional[np.ndarray]] = []
    neck_list: List[Optional[np.ndarray]] = []
    angle_list: List[float] = []
    y_norm_list: List[float] = []

    # Đọc toàn bộ frame (theo step), chạy predict từng frame (event ngắn nên OK)
    # Nếu muốn nhanh hơn nữa, có thể gom batch, nhưng code visualize thường ưu tiên đơn giản.
    fid = 0
    kept_idx = 0  # chỉ số frame đã giữ (sau step)
    frames_cache: List[np.ndarray] = []  # lưu frame để overlay lại khi đã có v/góc đã lọc

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if fid % frame_step != 0:
            fid += 1
            continue

        # Predict pose cho frame hiện tại
        res_list = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            half=half,
            verbose=False,
        )
        res = res_list[0] if isinstance(res_list, list) and len(res_list) > 0 else None
        kpts = select_main_person_kpts(res, min_kpt_score=min_kpt_score)

        hip = None
        neck = None
        angle = float("nan")
        y_norm = float("nan")

        if kpts is not None:
            # Vẽ skeleton + keypoints
            frame = draw_keypoints_and_skeleton(frame, kpts, min_kpt_score=min_kpt_score)

            # Lấy hip/neck proxy
            hip, neck = trunk_points_from_kpts(kpts, min_score=min_kpt_score)

            if hip is not None and neck is not None:
                # Vẽ vector thân
                hx, hy = int(hip[0]), int(hip[1])
                nx, ny = int(neck[0]), int(neck[1])
                cv2.line(frame, (hx, hy), (nx, ny), (255, 0, 0), 3)

                angle = compute_angle_deg(hip, neck)

                # Chuẩn hóa y theo "chiều cao gần đúng" để ổn định hơn
                trunk_len = float(np.linalg.norm(neck - hip))
                approx_height = trunk_len * 2.0
                if approx_height > 1e-6:
                    y_norm = float(hip[1] / approx_height)

        hip_list.append(hip)
        neck_list.append(neck)
        angle_list.append(angle)
        y_norm_list.append(y_norm)

        frames_cache.append(frame.copy())

        kept_idx += 1
        fid += 1

    cap.release()

    # Nếu không có frame nào
    T = len(frames_cache)
    if T == 0:
        writer.release()
        raise RuntimeError("Video rỗng hoặc không đọc được frame.")

    # Chuyển sang numpy để tính vận tốc/góc lọc mượt
    angles = np.array(angle_list, dtype=np.float32)
    y_norm = np.array(y_norm_list, dtype=np.float32)

    # Nội suy NaN để lọc/gradient không hỏng
    angles_i = nan_interpolate(angles)
    y_i = nan_interpolate(y_norm)

    # Lọc mượt
    angles_f = butter_lowpass_filter(angles_i, cutoff=5.0, fs=fps_eff, order=4)
    y_f = butter_lowpass_filter(y_i, cutoff=5.0, fs=fps_eff, order=4)

    dt = 1.0 / max(fps_eff, 1)
    v = np.gradient(y_f, dt)
    v_f = butter_lowpass_filter(v, cutoff=10.0, fs=fps_eff, order=4)

    # Tìm idx_peak theo |v|
    idx_peak = int(np.nanargmax(np.abs(v_f)))

    # Duyệt tính trạng thái bất động theo dy/da sau idx_peak
    immobile_flags = [False] * T
    for t in range(0, T - 1):
        if t < idx_peak:
            continue
        dy = float(abs(y_f[t + 1] - y_f[t]))
        da = float(abs(angles_f[t + 1] - angles_f[t]))
        if dy < thr_pos and da < thr_ang:
            immobile_flags[t] = True

    # Xuất video: overlay text theo thời gian
    for t in range(T):
        frame = frames_cache[t]

        # Tính delta_angle theo cửa sổ 0.5s trước/sau idx_peak (để hiển thị tham khảo)
        win = int(0.5 * fps_eff)
        before = angles_f[max(0, idx_peak - win): idx_peak]
        after = angles_f[idx_peak: min(T, idx_peak + win)]
        angle_before = float(np.nanmean(before)) if len(before) > 0 else float("nan")
        angle_after = float(np.nanmean(after)) if len(after) > 0 else float("nan")
        delta_angle = abs(angle_after - angle_before)

        lines = [
            f"frame: {t}/{T-1} (src_fps={fps_src}, eff_fps={fps_eff}, step={frame_step})",
            f"angle(deg): {angles_f[t]:.1f} | delta_angle~: {delta_angle:.1f}",
            f"v_y_norm: {v_f[t]:.3f} | v_peak: {float(np.nanmax(np.abs(v_f))):.3f}",
            f"peak_frame: {idx_peak} | after_peak_immobile: {immobile_flags[t]}",
        ]

        # Đánh dấu peak frame trên hình cho dễ nhìn
        if t == idx_peak:
            cv2.putText(frame, "PEAK", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)

        put_text_block(frame, lines, x=10, y=30)

        writer.write(frame)

    writer.release()


# =========================
# Demo chạy thử
# =========================
if __name__ == "__main__":
    # Video event (buffer 5–10 giây)
    video_path = "yolo11pose/tmp_event_cut.mp4"

    # File output visualize
    output_path = "yolo11pose/tmp_event_cut_pose_vis.mp4"

    # Model pose (nhẹ nhất: yolo11n-pose.pt)
    pose_model_path = "yolo11n-pose.pt"

    # Cấu hình gợi ý:
    # - CPU: device="cpu", half=False, imgsz=320/416, frame_step=1..2
    # - GPU: device="cuda:0", half=True, imgsz=416/640, frame_step=1
    device = "cpu"

    visualize_event_pose(
        video_path=video_path,
        output_path=output_path,
        pose_model_path=pose_model_path,
        device=device,
        imgsz=416,
        conf=0.25,
        iou=0.7,
        half=False,          # CPU -> False
        frame_step=1,        # 1: video mượt, 2: nhanh hơn nhưng ít frame hơn
        min_kpt_score=0.2,
        thr_pos=0.01,
        thr_ang=3.0,
    )

    print(f"Đã xuất video visualize: {output_path}")
