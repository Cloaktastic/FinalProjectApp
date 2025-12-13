"""
Module Person 1 - Pose & Feature Extraction
Trích xuất đặc trưng từ video sự kiện sử dụng OpenPose và YOLOv8
Tất cả các comment trong code đều bằng tiếng Việt
"""

import json
import os
import subprocess
import glob
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2
from scipy.signal import butter, filtfilt

# Hằng số cho các chỉ số khớp trong BODY_25
MID_HIP = 8  # Chỉ số hông giữa trong mô hình BODY_25
NECK = 1     # Chỉ số cổ trong mô hình BODY_25

# FRAME_STEP = 2 => xử lý 1 frame, bỏ 1 frame (nếu video gốc ~30 FPS thì hiệu dụng ~15 FPS)
FRAME_STEP = 2
# Giảm độ phân giải mạng để tăng tốc (height = 256, width tự suy ra theo tỷ lệ khung hình)
NET_RES = "-1x256"


def run_openpose_on_video(video_path: str, output_dir: str, openpose_bin: str) -> List[str]:
    """
    Chạy OpenPose BODY_25 trên video đầu vào.
    Lưu trữ các điểm khớp dưới dạng JSON vào output_dir (một file JSON cho mỗi frame).
    Trả về danh sách các đường dẫn file JSON, được sắp xếp theo thứ tự frame.
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Tên video (không có đuôi) để lọc file JSON liên quan
    video_name = Path(video_path).stem

    # ===== (1) Xoá JSON cũ của video này để tránh xài lại kết quả lỗi =====
    old_jsons = glob.glob(os.path.join(output_dir, f"{video_name}_*.json"))
    for p in old_jsons:
        try:
            os.remove(p)
        except OSError:
            pass

    # Lấy đường dẫn tuyệt đối tới binary và root của OpenPose
    openpose_bin_path = Path(openpose_bin).resolve()
    openpose_root = openpose_bin_path.parent.parent   # ...\bin\..  => C:\openpose
    model_folder = openpose_root / "models"

    # Lệnh gọi OpenPose với các tham số cần thiết
    cmd = [
        str(openpose_bin_path),
        "--video", str(Path(video_path).resolve()),
        "--write_json", str(Path(output_dir).resolve()),
        "--display", "0",              # không hiện GUI
        "--model_pose", "BODY_25",
        "--number_people_max", "1",
        "--render_pose", "0",          # KHÔNG render để tránh ghi video
        "--model_folder", str(model_folder),
        "--frame_step", str(FRAME_STEP),
        "--net_resolution", NET_RES,
        "--scale_number", "1",
        "--scale_gap", "0.25",
        # "--num_gpu", "0",  # nếu đang CPU build thì có thể bật, GPU build thì bỏ
    ]

    # Chạy lệnh OpenPose
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(openpose_root),   # chạy từ C:\openpose (nơi có folder models)
        )
        print(f"OpenPose đã chạy thành công trên {video_path}")
        # Nếu cần debug sâu:
        # print("[DEBUG OpenPose stdout]\n", result.stdout)
        # print("[DEBUG OpenPose stderr]\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Lỗi khi chạy OpenPose:")
        print(e.stderr)
        raise

    # Thu thập tất cả các file JSON cho video này và sắp xếp theo thứ tự frame
    json_files = glob.glob(os.path.join(output_dir, f"{video_name}_*_keypoints.json"))

    # Debug: in ra số lượng file JSON và 5 file đầu tiên
    print(f"[DEBUG] Tổng số file JSON trong {output_dir}: {len(json_files)}")
    for p in sorted(json_files)[:5]:
        print("[DEBUG] JSON mẫu:", os.path.basename(p))

    def frame_index(path: str) -> int:
        # Ví dụ: tmp_event_000000000123_keypoints.json
        stem_parts = Path(path).stem.split('_')
        for part in reversed(stem_parts):
            if part.isdigit():
                return int(part)
        return 0

    json_files.sort(key=frame_index)
    return json_files


def load_keypoints_from_openpose(json_paths: List[str]) -> np.ndarray:
    """
    Tải các điểm khớp BODY_25 từ danh sách file JSON OpenPose.
    Trả về một mảng NumPy với hình dạng (T, 25, 3), trong đó:
        - T là số lượng frame
        - 25 là số lượng khớp trong BODY_25
        - 3 tương ứng với (x, y, score)
    """
    all_keypoints = []

    for idx, json_path in enumerate(json_paths):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            people = data.get('people', [])

            # Debug vài frame đầu
            if idx < 5:
                print(f"[DEBUG] {os.path.basename(json_path)} - len(people) = {len(people)}")

            if people and len(people) > 0:
                keypoints_2d = np.array(people[0]['pose_keypoints_2d'])
                keypoints = keypoints_2d.reshape(-1, 3)  # (25, 3)
            else:
                keypoints = np.full((25, 3), np.nan)

            all_keypoints.append(keypoints)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Lỗi khi đọc file {json_path}: {e}")
            all_keypoints.append(np.full((25, 3), np.nan))

    all_keypoints_arr = np.array(all_keypoints)
    hip_all = all_keypoints_arr[:, MID_HIP, :2]
    neck_all = all_keypoints_arr[:, NECK, :2]
    valid_mask = ~np.isnan(hip_all).any(axis=1) & ~np.isnan(neck_all).any(axis=1)

    print(f"[DEBUG] Tổng số frame JSON: {all_keypoints_arr.shape[0]}")
    print(f"[DEBUG] Số frame có skeleton hợp lệ (hip & neck không NaN): {valid_mask.sum()}")

    return all_keypoints_arr


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Bộ lọc thông thấp Butterworth để làm mượt dữ liệu.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def interpolate_1d_nan(arr: np.ndarray) -> np.ndarray:
    """
    Nội suy tuyến tính cho mảng 1D có NaN.
    Trả về mảng mới đã được lấp NaN (nếu có >= 2 điểm hợp lệ).
    Nếu toàn NaN hoặc chỉ có 1 điểm hợp lệ thì giữ nguyên.
    """
    arr = arr.astype(float).copy()
    n = arr.shape[0]

    # mặt nạ phần tử không phải NaN
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        # Không đủ điểm để nội suy
        return arr

    x = np.arange(n)
    arr[~mask] = np.interp(x[~mask], x[mask], arr[mask])
    return arr


def extract_features_from_keypoints(
    keypoints: np.ndarray,
    fps: int = 30,
    use_height_normalization: bool = True,
) -> Dict[str, float]:
    """
    Trích xuất các đặc trưng từ mảng điểm khớp.

    Args:
        keypoints: Mảng NumPy với hình dạng (T, 25, 3) [x, y, score]
        fps: Số frame mỗi giây (đÃ hiệu chỉnh theo FRAME_STEP)
        use_height_normalization: Có sử dụng chuẩn hóa chiều cao hay không
    """
    T = keypoints.shape[0]
    if T < 5:
        # Nếu số frame quá ít thì không đủ dữ liệu để tính toán
        hip_all = keypoints[:, MID_HIP, :2]
        neck_all = keypoints[:, NECK, :2]
        valid_mask = ~np.isnan(hip_all).any(axis=1) & ~np.isnan(neck_all).any(axis=1)
        print(f"[DEBUG] T = {T}, valid_mask.sum() = {valid_mask.sum()}")

        return {
            "v_peak": float("nan"),
            "delta_angle": float("nan"),
            "immobile_duration": 0.0,
        }

    # Trích xuất vị trí hông và cổ cho tất cả frame
    hip_all = keypoints[:, MID_HIP, :2]   # (T, 2)
    neck_all = keypoints[:, NECK, :2]     # (T, 2)

    # Tạo mask các frame có đủ dữ liệu (không NaN ở hip và neck)
    valid_mask = ~np.isnan(hip_all).any(axis=1) & ~np.isnan(neck_all).any(axis=1)

    if valid_mask.sum() < 5:
        # Nếu số frame có người xuất hiện quá ít thì bỏ qua
        return {
            "v_peak": float("nan"),
            "delta_angle": float("nan"),
            "immobile_duration": 0.0,
        }

    # Chỉ lấy các frame có người để phân tích
    hip = hip_all[valid_mask]    # (T_valid, 2)
    neck = neck_all[valid_mask]  # (T_valid, 2)
    T_valid = hip.shape[0]

    # Chuẩn hóa tọa độ y của hông
    y_hip = hip[:, 1]  # (T_valid,)

    # Nội suy NaN trong y_hip nếu có
    y_hip = interpolate_1d_nan(y_hip)

    # Tính chiều dài thân và chiều cao gần đúng
    trunk_vec = neck - hip                           # (T_valid, 2)
    trunk_len = np.linalg.norm(trunk_vec, axis=1)    # (T_valid,)
    trunk_len = interpolate_1d_nan(trunk_len)
    approx_height = trunk_len * 2.0

    # Tránh chia cho 0 bằng cách nội suy rồi thêm epsilon
    approx_height[approx_height == 0] = np.nan
    approx_height = interpolate_1d_nan(approx_height)

    if use_height_normalization:
        y_norm = y_hip / (approx_height + 1e-6)
    else:
        y_norm = y_hip.copy()

    # Nếu toàn bộ vẫn NaN thì không tính được gì
    if np.all(np.isnan(y_norm)):
        return {
            "v_peak": float("nan"),
            "delta_angle": float("nan"),
            "immobile_duration": 0.0,
        }

    # Làm mượt dữ liệu với bộ lọc thông thấp
    dt = 1.0 / fps
    try:
        y_filt = butter_lowpass_filter(y_norm, cutoff=5.0, fs=fps, order=4)
    except Exception as e:
        print("Lỗi khi lọc y_filt:", e)
        return {
            "v_peak": float("nan"),
            "delta_angle": float("nan"),
            "immobile_duration": 0.0,
        }

    # Tính tốc độ thẳng đứng bằng cách lấy đạo hàm số
    v = np.gradient(y_filt, dt)

    # Làm mượt tốc độ
    try:
        v_filt = butter_lowpass_filter(v, cutoff=10.0, fs=fps, order=4)
    except Exception as e:
        print("Lỗi khi lọc v_filt:", e)
        return {
            "v_peak": float("nan"),
            "delta_angle": float("nan"),
            "immobile_duration": 0.0,
        }

    if np.all(np.isnan(v_filt)):
        # Nếu sau lọc mà vẫn toàn NaN thì không có thông tin tốc độ
        return {
            "v_peak": float("nan"),
            "delta_angle": float("nan"),
            "immobile_duration": 0.0,
        }

    # Tính đỉnh tốc độ
    v_peak = float(np.nanmax(v_filt))

    # Tính góc thân từ vector hip -> neck
    vec = neck - hip  # (T_valid, 2)
    angles = np.degrees(np.arctan2(vec[:, 0], -vec[:, 1]))  # (T_valid,)
    angles = interpolate_1d_nan(angles)

    try:
        angles_filt = butter_lowpass_filter(angles, cutoff=5.0, fs=fps, order=4)
    except Exception as e:
        print("Lỗi khi lọc angles_filt:", e)
        return {
            "v_peak": v_peak,
            "delta_angle": float("nan"),
            "immobile_duration": 0.0,
        }

    if np.all(np.isnan(angles_filt)):
        return {
            "v_peak": v_peak,
            "delta_angle": float("nan"),
            "immobile_duration": 0.0,
        }

    # Tìm chỉ số của đỉnh tốc độ trong dãy đã rút gọn
    idx_peak = int(np.nanargmax(v_filt))

    # Xác định cửa sổ thời gian trước và sau đỉnh
    win = int(0.5 * fps)
    before = angles_filt[max(0, idx_peak - win): idx_peak]
    after = angles_filt[idx_peak: min(T_valid, idx_peak + win)]

    # Tính góc trung bình trước và sau
    angle_before = float(np.nanmean(before)) if len(before) > 0 else float("nan")
    angle_after = float(np.nanmean(after)) if len(after) > 0 else float("nan")
    delta_angle = float(abs(angle_after - angle_before))

    # Tính thời gian bất động sau impact
    thr_pos = 0.01   # Ngưỡng thay đổi vị trí nhỏ (đơn vị chuẩn hoá)
    thr_ang = 3.0    # Ngưỡng thay đổi góc nhỏ (độ)

    immobile_frames = 0
    for t in range(idx_peak, T_valid - 1):
        dy = abs(y_filt[t + 1] - y_filt[t])
        da = abs(angles_filt[t + 1] - angles_filt[t])
        if dy < thr_pos and da < thr_ang:
            immobile_frames += 1

    immobile_duration = immobile_frames * dt  # giây

    return {
        "v_peak": v_peak,
        "delta_angle": delta_angle,
        "immobile_duration": float(immobile_duration),
    }


def extract_features(
    video_path: str,
    openpose_bin: str,
    output_dir: str,
    fps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Hàm cấp cao để trích xuất đặc trưng từ video sự kiện.
    """
    # Bước 1: Chạy OpenPose và lấy đường dẫn file JSON
    json_paths = run_openpose_on_video(video_path, output_dir, openpose_bin)

    # Bước 2: Tải điểm khớp từ file JSON
    keypoints = load_keypoints_from_openpose(json_paths)

    # Bước 3: Đọc FPS từ video nếu chưa được cung cấp
    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

    effective_fps = max(1, fps / FRAME_STEP)
    print(f"[DEBUG] fps gốc = {fps}, FRAME_STEP = {FRAME_STEP}, fps_effective = {effective_fps}")

    # Bước 4: Trích xuất đặc trưng từ điểm khớp
    features = extract_features_from_keypoints(keypoints, fps=int(effective_fps))

    return features


if __name__ == "__main__":
    # Ví dụ demo: chạy thử trên một video event
    video_path = "tmp_event_cut.mp4"  # hoặc tmp_event.mp4 tuỳ bạn
    openpose_bin = r"C:\openpose\bin\OpenPoseDemo.exe"
    output_dir = "openpose_output"

    # Tạo thư mục đầu ra nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Trích xuất đặc trưng
    features = extract_features(video_path, openpose_bin, output_dir)
    print("Đặc trưng trích xuất được:")
    print(features)
