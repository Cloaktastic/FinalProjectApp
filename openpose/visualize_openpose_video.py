"""
Script visualize OpenPose BODY_25
Đọc video gốc + JSON keypoints từ OpenPose, vẽ khung xương và xuất video debug.
Tất cả comment đều bằng tiếng Việt.
"""

import json
import glob
from pathlib import Path

import cv2
import numpy as np

# Định nghĩa các cặp khớp để nối lại thành skeleton BODY_25
# Tham khảo từ cấu trúc BODY_25 của OpenPose
BODY_25_PAIRS = [
    (0, 1),  # mũi - cổ
    (1, 2), (2, 3), (3, 4),        # tay phải
    (1, 5), (5, 6), (6, 7),        # tay trái
    (1, 8),                        # cổ - hông giữa
    (8, 9), (9, 10), (10, 11),     # chân phải
    (8, 12), (12, 13), (13, 14),   # chân trái
    (0, 15), (15, 17),             # mắt phải / tai phải
    (0, 16), (16, 18),             # mắt trái / tai trái
    (11, 24), (14, 21),            # gót chân phải / trái
    (19, 20), (21, 19),            # ngón chân trái
    (22, 23), (24, 22),            # ngón chân phải
]

def parse_frame_index(json_path: str) -> int:
    """
    Lấy index frame từ tên file JSON.
    Ví dụ: tmp_event_000000000123_keypoints.json -> 123
    """
    stem = Path(json_path).stem  # tmp_event_000000000123_keypoints
    parts = stem.split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return -1


def load_keypoints_dict(json_folder: str, video_stem: str):
    """
    Đọc toàn bộ JSON trong json_folder cho video có tên video_stem (vd: tmp_event_cut)
    Trả về dict: frame_index (int) -> mảng (25, 3)
    """
    pattern = str(Path(json_folder) / f"{video_stem}_*_keypoints.json")
    json_files = glob.glob(pattern)
    json_files.sort(key=parse_frame_index)

    kp_dict = {}

    print(f"[DEBUG] Tìm thấy {len(json_files)} file JSON cho video {video_stem}")

    for p in json_files:
        idx = parse_frame_index(p)
        try:
            with open(p, 'r') as f:
                data = json.load(f)

            people = data.get("people", [])
            if not people:
                continue

            keypoints_2d = np.array(people[0]["pose_keypoints_2d"], dtype=float).reshape(-1, 3)
            kp_dict[idx] = keypoints_2d
        except Exception as e:
            print(f"[WARN] Lỗi khi đọc {p}: {e}")

    print(f"[DEBUG] Số frame có skeleton trong dict: {len(kp_dict)}")
    return kp_dict


def draw_skeleton_body25(
    frame,
    keypoints: np.ndarray,
    threshold: float = 0.05,
):
    """
    Vẽ skeleton BODY_25 lên frame sử dụng keypoints (25, 3) [x, y, score]
    threshold: ngưỡng score tối thiểu để vẽ.
    """
    for i, (p1, p2) in enumerate(BODY_25_PAIRS):
        x1, y1, c1 = keypoints[p1]
        x2, y2, c2 = keypoints[p2]

        if c1 < threshold or c2 < threshold:
            continue

        p1_int = (int(x1), int(y1))
        p2_int = (int(x2), int(y2))

        # Vẽ line
        cv2.line(frame, p1_int, p2_int, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # Vẽ điểm khớp
    for j in range(keypoints.shape[0]):
        x, y, c = keypoints[j]
        if c < threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return frame


def visualize_openpose(
    video_path: str,
    json_folder: str,
    output_path: str,
):
    """
    Đọc video + JSON từ OpenPose, vẽ skeleton lên frame tương ứng và ghi video mới.
    """
    video_path = str(Path(video_path).resolve())
    json_folder = str(Path(json_folder).resolve())
    output_path = str(Path(output_path).resolve())

    video_stem = Path(video_path).stem

    # Đọc dict frame_index -> keypoints
    kp_dict = load_keypoints_dict(json_folder, video_stem)

    # Mở video gốc
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video: {video_path}")
        return

    # Lấy thông tin video để tạo VideoWriter
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    print(f"[INFO] Video input: {video_path}")
    print(f"[INFO] Kích thước: {width}x{height}, fps = {fps}")

    # Chọn codec - dùng mp4v hoặc XVID tuỳ bạn
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # hoặc "XVID"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[ERROR] Không tạo được video output: {output_path}")
        cap.release()
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Nếu frame này có keypoints từ OpenPose thì vẽ
        if frame_idx in kp_dict:
            frame = draw_skeleton_body25(frame, kp_dict[frame_idx], threshold=0.05)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] Đã ghi video skeleton ra: {output_path}")
    print(f"[INFO] Tổng số frame xử lý: {frame_idx}")


if __name__ == "__main__":
    # Ví dụ: dùng chung folder với person1_features.py
    video_path = "tmp_event_cut.mp4"               # video gốc
    json_folder = "openpose_output"               # cùng thư mục JSON mà OpenPose đã xuất
    output_path = "openpose_output/tmp_event_cut_pose.mp4"

    visualize_openpose(video_path, json_folder, output_path)
