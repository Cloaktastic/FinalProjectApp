import os
import cv2
import av
import time
import math
import queue
import threading
import subprocess
import requests
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO
from collections import deque
from typing import Optional, List, Tuple, Dict
from streamlit_webrtc import webrtc_streamer

# ============================================================
# CONFIG
# ============================================================
VIDEO = "Video"
CAMERA = "Camera"
SOURCES_LIST = [VIDEO, CAMERA]

VIDEO_DIR = "videos"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

MODEL_PATH = "models/100_epochs.pt"        # fall + person detector
POSE_MODEL_PATH = "models/yolo11n-pose.pt" # pose model

TOKEN_FILE = "token.txt"
OUT_DIR = "videos"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Elderly Fall Detection", page_icon="🚨", layout="wide")
st.title("🚨 Elderly Fall Detection")
st.caption("YOLO-based Fall & Person Detection (Video / Camera)")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Input")
source_radio = st.sidebar.radio("Select source", SOURCES_LIST, index=0)

st.sidebar.header("Detection Settings (Camera)")
confidence_value = st.sidebar.slider("Detection confidence", 0.05, 0.95, 0.25, 0.01)
confirm_seconds = st.sidebar.slider("Hold FALL bbox for (sec)", 1.0, 10.0, 3.0, 0.5)
cooldown_sec = st.sidebar.slider("Cooldown after confirm (sec)", 0.0, 20.0, 6.0, 0.5)
detection_interval = st.sidebar.slider("Detection interval (sec)", 0.0, 1.0, 0.15, 0.01)

st.sidebar.header("Event Clip Settings (Camera)")
pre_sec = st.sidebar.slider("Clip PRE seconds", 1, 20, 10, 1)
post_sec = st.sidebar.slider("Clip POST seconds", 1, 30, 20, 1)

st.sidebar.header("Models")
use_gpu = st.sidebar.selectbox("Device", ["Auto", "cpu", "cuda"], index=0)

# ----------------------------
# Camera performance
# ----------------------------
st.sidebar.header("Camera Performance")
camera_frame_step = st.sidebar.slider("Camera frame skip (process every N frames)", 1, 6, 2, 1)
# Buffer still stores all frames (for smooth clip). If you want lower memory, you can set >1.
buffer_store_step = st.sidebar.slider("Camera buffer store step (store every N frames)", 1, 3, 1, 1)

# ----------------------------
# Pose settings (Video + optional Camera compute)
# ----------------------------
st.sidebar.header("Pose (Video / optional Camera compute)")
pose_enabled = st.sidebar.toggle("Enable pose estimation", value=True)
pose_conf = st.sidebar.slider("Pose confidence", 0.05, 0.90, 0.30, 0.01)
pose_sample_fps = st.sidebar.slider("Pose sample FPS", 1, 15, 6, 1)

st.sidebar.subheader("Smoothing")
ema_alpha = st.sidebar.slider("EMA alpha (hip smoothing)", 0.05, 0.90, 0.35, 0.05)
pose_roll_window = st.sidebar.slider("Pose rolling window (sec)", 1.0, 10.0, 4.0, 0.5)

st.sidebar.subheader("Pose thresholds (tune)")
v_fall_confirm_min = st.sidebar.slider("v_peak FALL confirm (px/s)", 40, 600, 160, 10)
angle_change_confirm_min = st.sidebar.slider("Angle change confirm (deg)", 5, 90, 18, 1)
immobile_confirm_min = st.sidebar.slider("Immobile time confirm (sec)", 0.0, 6.0, 0.8, 0.1)
immobile_speed_th = st.sidebar.slider("Immobile speed th (px/s)", 5, 200, 45, 5)

st.sidebar.subheader("Immobile hysteresis")
immobile_enter_margin = st.sidebar.slider("Enter margin (px/s)", 0, 50, 10, 1)
immobile_exit_margin = st.sidebar.slider("Exit margin (px/s)", 0, 80, 15, 1)
immobile_enter_th = max(0.0, float(immobile_speed_th - immobile_enter_margin))
immobile_exit_th = float(immobile_speed_th + immobile_exit_margin)

st.sidebar.subheader("Verdict debounce")
debounce_window_sec = st.sidebar.slider("Debounce window (sec)", 0.2, 3.0, 1.2, 0.1)
debounce_min_votes = st.sidebar.slider("Min samples for vote", 1, 30, 6, 1)

st.sidebar.subheader("Angle hysteresis")
lying_enter_deg = st.sidebar.slider("Lying enter angle (deg)", 5, 40, 25, 1)
lying_exit_deg = st.sidebar.slider("Lying exit angle (deg)", 10, 70, 32, 1)
angle_ema_alpha = st.sidebar.slider("Angle EMA alpha", 0.05, 0.90, 0.35, 0.05)

st.sidebar.subheader("Pose latch (no flicker)")
likely_hold_sec = st.sidebar.slider("Hold LIKELY for (sec)", 0.0, 10.0, 4.0, 0.5)
confirm_hold_sec = st.sidebar.slider("Hold CONFIRMED for (sec)", 0.0, 30.0, 12.0, 1.0)
recover_stand_deg = st.sidebar.slider("Recover stand angle (deg)", 35, 90, 60, 1)
recover_hold_sec = st.sidebar.slider("Recover hold (sec)", 0.0, 10.0, 2.0, 0.5)

st.sidebar.subheader("Anti-sleep false positive (ARMING GATE)")
require_upright_before_fall = st.sidebar.toggle("Require upright before fall", value=True)
upright_angle_th = st.sidebar.slider("Upright angle threshold (deg)", 30, 90, 55, 1)
min_upright_time_sec = st.sidebar.slider("Min upright duration to ARM (sec)", 0.0, 3.0, 0.6, 0.1)
upright_memory_sec = st.sidebar.slider("Upright memory window (sec)", 0.5, 10.0, 3.5, 0.5)

st.sidebar.subheader("One-shot gates (impact moment only)")
require_angle_drop = st.sidebar.toggle("Require directional angle drop", value=True)
angle_drop_deg = st.sidebar.slider("Angle drop min (deg)", 5, 80, 25, 1)
angle_drop_window_sec = st.sidebar.slider("Angle drop window (sec)", 0.3, 3.0, 1.2, 0.1)

require_vertical_down = st.sidebar.toggle("Require vertical-down dominance", value=False)
vy_peak_min = st.sidebar.slider("vy_peak min (px/s)", 0, 600, 120, 10)
vy_ratio_min = st.sidebar.slider("vy_ratio min (0-1)", 0.0, 1.0, 0.55, 0.05)

st.sidebar.header("Camera Pose Compute")
compute_pose_on_camera = st.sidebar.toggle("Compute pose on camera (for OUTSIDE panel only)", value=False)

# ============================================================
# TELEGRAM
# ============================================================
def read_telegram_credentials():
    if not os.path.exists(TOKEN_FILE):
        return None, None
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    if len(lines) < 2:
        return None, None
    return lines[0], lines[1]

def tg_send_message(token, chat_id, text):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=15
        )
        return r.status_code == 200
    except Exception:
        return False

def tg_send_video(token, chat_id, path, caption=""):
    try:
        with open(path, "rb") as f:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendVideo",
                data={"chat_id": chat_id, "caption": caption},
                files={"video": (os.path.basename(path), f, "video/mp4")},
                timeout=180
            )
        return r.status_code == 200
    except Exception:
        return False

# ============================================================
# MODEL LOADING (CPU fallback)
# ============================================================
def resolve_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"

MODEL_LOCK = threading.Lock()
POSE_LOCK = threading.Lock()

@st.cache_resource
def load_model(path: str, device: str):
    try:
        m = YOLO(path)
        _ = m.to(device)
        return m
    except Exception as e:
        if device != "cpu":
            try:
                m = YOLO(path)
                _ = m.to("cpu")
                st.warning(f"Model {path} loaded on CPU (fallback). Reason: {e}")
                return m
            except Exception as e2:
                st.error(f"Failed to load model {path} on CPU fallback: {e2}")
                return None
        st.error(f"Failed to load model {path}: {e}")
        return None

device = resolve_device(use_gpu)
model = load_model(MODEL_PATH, device)

pose_model = None
if pose_enabled:
    pose_model = load_model(POSE_MODEL_PATH, device)

if model is None:
    st.stop()

# ============================================================
# VIDEO / CLIP WRITING UTILS
# ============================================================
def ffmpeg_fix_h264_yuv420p(in_path: str, out_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-movflags", "+faststart",
        "-an",
        out_path
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (r.returncode == 0) and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False

def _estimate_fps(frames_with_t: List[Tuple[float, np.ndarray]], default_fps: float = 30.0) -> float:
    if not frames_with_t or len(frames_with_t) < 2:
        return float(default_fps)
    frames_with_t = sorted(frames_with_t, key=lambda x: x[0])
    dur = float(frames_with_t[-1][0] - frames_with_t[0][0])
    if dur <= 1e-6:
        return float(default_fps)
    fps = (len(frames_with_t) - 1) / dur
    return float(min(60.0, max(5.0, fps)))

def _write_mp4_raw(frames_with_t: List[Tuple[float, np.ndarray]], out_path: str, fps: float) -> Optional[str]:
    if not frames_with_t:
        return None
    frames_with_t = sorted(frames_with_t, key=lambda x: x[0])
    h, w = frames_with_t[0][1].shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not vw.isOpened():
        return None
    for _, fr in frames_with_t:
        if fr is None:
            continue
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h))
        vw.write(fr)
    vw.release()
    return out_path

def _extract_window(frames_with_t: deque, t_ref: float, pre: float, post: float) -> List[Tuple[float, np.ndarray]]:
    if not frames_with_t:
        return []
    arr = list(frames_with_t)
    arr.sort(key=lambda x: x[0])
    t_min = arr[0][0]
    t_max = arr[-1][0]
    a = max(t_min, t_ref - pre)
    b = min(t_max, t_ref + post)
    return [(t, fr) for (t, fr) in arr if a <= t <= b]

def _try_fix_for_telegram(in_path: str) -> str:
    out_path = in_path.replace("_raw.mp4", ".mp4")
    ok = ffmpeg_fix_h264_yuv420p(in_path, out_path)
    return out_path if ok else in_path

# ============================================================
# DETECTION HELPERS (Camera boxes)
# ============================================================
Det = Tuple[int, int, int, int, int, float]  # x1,y1,x2,y2,cls,conf

def _get_name(cls_id: int) -> str:
    try:
        return model.names[int(cls_id)]
    except Exception:
        return str(cls_id)

def is_fall(name: str) -> bool:
    return str(name).lower() == "fall"

def is_person(name: str) -> bool:
    return str(name).lower() in ["person", "people", "human"]

def pack_detections(boxes) -> List[Det]:
    dets: List[Det] = []
    if boxes is None:
        return dets
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
    for (b, c, k) in zip(xyxy, conf, cls):
        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
        dets.append((x1, y1, x2, y2, int(k), float(c)))
    return dets

def draw_boxes_camera(
    img_bgr: np.ndarray,
    dets: List[Det],
    detected_confirmed: bool,
    fall_hold_start_t: Optional[float],
    t_now: float,
    confirm_seconds_: float
) -> np.ndarray:
    if dets is None:
        dets = []
    h, w = img_bgr.shape[:2]

    fall_count, person_count, other_count = 0, 0, 0
    for (x1, y1, x2, y2, cls_id, conf) in dets:
        class_name = _get_name(cls_id)
        if is_fall(class_name):
            fall_count += 1
            box_color = (0, 0, 255)
            label_text = f"FALL {'CONFIRMED' if detected_confirmed else 'DETECTED'}: {conf:.2f}"
        elif is_person(class_name):
            person_count += 1
            box_color = (0, 255, 0)
            label_text = f"PERSON: {conf:.2f}"
        else:
            other_count += 1
            box_color = (0, 255, 255)
            label_text = f"{str(class_name).upper()}: {conf:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img_bgr, label_text, (x1, max(12, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        if is_fall(class_name) and (not detected_confirmed) and (fall_hold_start_t is not None):
            held = max(0.0, t_now - fall_hold_start_t)
            remain = max(0.0, confirm_seconds_ - held)
            timer = f"Hold: {held:.1f}s | Remain: {remain:.1f}s"
            cv2.putText(img_bgr, timer, (x1, min(h - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    summary = f"{person_count} person | {fall_count} fall"
    if other_count:
        summary += f" | {other_count} other"
    (tw, th), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img_bgr, (w - tw - 20, 10), (w - 10, 10 + th + 14), (0, 0, 0), -1)
    cv2.putText(img_bgr, summary, (w - tw - 12, 10 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return img_bgr

# ============================================================
# POSE REALTIME (Video overlay + optional camera compute)
# ============================================================
LS, RS, LH, RH = 5, 6, 11, 12

def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0

def _angle_deg_from_horizontal(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    v = shoulder_mid - hip_mid
    dx, dy = float(v[0]), float(v[1])
    ang = abs(math.degrees(math.atan2(dy, dx)))
    if ang > 90:
        ang = 180 - ang
    return ang

def prune_series(dq: deque, t_now: float, window_sec: float):
    while dq and (t_now - dq[0][0]) > window_sec:
        dq.popleft()

def init_pose_rt(window_sec: float) -> Dict:
    return {
        "window_sec": float(window_sec),
        "last_pose_t": -1e18,

        "hip_ema": None,
        "prev_hip_ema": None,
        "prev_t": None,

        "v_curr": None,
        "vx_curr": None,
        "vy_down_curr": None,

        "angle_raw": None,
        "angle_ema": None,
        "angle_curr": None,

        "speed_series": deque(),
        "angle_series": deque(),
        "vy_series": deque(),
        "vy_ratio_series": deque(),

        "immobile_active": False,
        "immobile_start": None,
        "immobile_run": 0.0,

        "lying_active": False,

        "v_peak_recent": 0.0,
        "angle_change_recent": 0.0,
        "angle_drop_recent": 0.0,
        "vy_peak_recent": 0.0,
        "vy_ratio_peak_recent": 0.0,

        "drop_gate_ok": False,
        "vdown_gate_ok": False,

        "verdict_raw": None,
        "verdict_vote": None,
        "verdict_hist": deque(),

        "state": "NO_FALL",
        "state_until": None,
        "recover_start": None,

        "last_pose_ok": False,
        "pose_fail_count": 0,
        "pose_ok_count": 0,

        "upright_run": 0.0,
        "last_upright_t": None,
        "armed": False,
    }

def _compute_bbox_area_from_kps(kps_xy: np.ndarray, kps_conf: np.ndarray, conf_th: float = 0.15) -> float:
    valid = kps_conf >= conf_th
    if valid.sum() < 4:
        return 0.0
    pts = kps_xy[valid]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return float(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))

def _compute_tracking_score(prev_hip: Optional[np.ndarray], hip_mid: np.ndarray, area: float, w_area: float = 0.002) -> float:
    if prev_hip is None:
        dist = 0.0
    else:
        dist = float(np.linalg.norm(hip_mid - prev_hip))
    return float(-dist + w_area * area)

def select_stable_person(r, prev_hip_ema: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
    if r.keypoints is None or len(r.keypoints) == 0:
        return None, None, None, None

    kps_all_xy = r.keypoints.xy
    n = int(kps_all_xy.shape[0])

    if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
        kps_all_conf = r.keypoints.conf
    else:
        kps_all_conf = torch.ones((n, int(kps_all_xy.shape[1])), dtype=torch.float32)

    best = None
    best_score = None

    for i in range(n):
        kps_xy = kps_all_xy[i].cpu().numpy()
        kps_conf = kps_all_conf[i].cpu().numpy()

        if max(LS, RS, LH, RH) >= kps_xy.shape[0]:
            continue
        if (kps_conf[LS] < 0.15) or (kps_conf[RS] < 0.15) or (kps_conf[LH] < 0.15) or (kps_conf[RH] < 0.15):
            continue

        shoulder_mid = _mid(kps_xy[LS], kps_xy[RS]).astype(np.float32)
        hip_mid = _mid(kps_xy[LH], kps_xy[RH]).astype(np.float32)

        area = _compute_bbox_area_from_kps(kps_xy, kps_conf, conf_th=0.15)
        score = _compute_tracking_score(prev_hip_ema, hip_mid, area, w_area=0.002)

        if (best is None) or (score > best_score):
            best = (hip_mid, shoulder_mid, area, score)
            best_score = score

    if best is None:
        return None, None, None, None
    return best

def update_verdict_vote(pose_rt: Dict, t_now: float):
    prune_series(pose_rt["verdict_hist"], t_now, float(debounce_window_sec))
    hist = list(pose_rt["verdict_hist"])
    if len(hist) < int(debounce_min_votes):
        return

    counts: Dict[str, int] = {}
    for _, v in hist:
        counts[v] = counts.get(v, 0) + 1

    def prio(v: str) -> int:
        if v == "FALL_CONFIRMED":
            return 3
        if v == "FALL_LIKELY":
            return 2
        if v == "NO_FALL":
            return 1
        return 0

    items = list(counts.items())
    items.sort(key=lambda x: (x[1], prio(x[0])), reverse=True)
    pose_rt["verdict_vote"] = items[0][0]

def _compute_angle_drop(pose_rt: Dict, t_now: float) -> float:
    if not pose_rt["angle_series"]:
        return 0.0
    cur = pose_rt.get("angle_curr")
    if cur is None:
        return 0.0
    win = float(angle_drop_window_sec)
    vals = [a for (t, a) in pose_rt["angle_series"] if (t_now - t) <= win]
    if not vals:
        return 0.0
    return float(max(vals) - float(cur))

def _allowed_by_upright_gate(pose_rt: Dict, t_now: float) -> bool:
    if not require_upright_before_fall:
        return True
    last_upright = pose_rt.get("last_upright_t", None)
    if (not pose_rt.get("armed", False)) or (last_upright is None):
        return False
    return (t_now - float(last_upright)) <= float(upright_memory_sec)

def _update_one_shot_gates(pose_rt: Dict):
    impact_like = (
        float(pose_rt.get("v_peak_recent", 0.0)) >= 0.8 * float(v_fall_confirm_min) and
        float(pose_rt.get("angle_change_recent", 0.0)) >= 0.8 * float(angle_change_confirm_min)
    )
    if not impact_like:
        return

    if require_angle_drop:
        if float(pose_rt.get("angle_drop_recent", 0.0)) >= float(angle_drop_deg):
            pose_rt["drop_gate_ok"] = True
    else:
        pose_rt["drop_gate_ok"] = True

    if require_vertical_down:
        vy_peak = float(pose_rt.get("vy_peak_recent", 0.0))
        vyr_peak = float(pose_rt.get("vy_ratio_peak_recent", 0.0))
        if (vy_peak >= float(vy_peak_min)) and (vyr_peak >= float(vy_ratio_min)):
            pose_rt["vdown_gate_ok"] = True
    else:
        pose_rt["vdown_gate_ok"] = True

def compute_raw_verdict(pose_rt: Dict, t_now: float) -> str:
    angle_curr = pose_rt.get("angle_curr")
    if angle_curr is None:
        return "UNKNOWN"

    if not _allowed_by_upright_gate(pose_rt, t_now):
        return "NO_FALL"

    lying = bool(pose_rt.get("lying_active", False))
    v_peak_recent = float(pose_rt.get("v_peak_recent", 0.0))
    angle_change_recent = float(pose_rt.get("angle_change_recent", 0.0))
    immobile_run = float(pose_rt.get("immobile_run", 0.0))

    state_now = pose_rt.get("state", "NO_FALL")
    in_event = state_now in ("FALL_LIKELY", "FALL_CONFIRMED")

    if not in_event:
        if require_angle_drop and (not pose_rt.get("drop_gate_ok", False)):
            return "NO_FALL"
        if require_vertical_down and (not pose_rt.get("vdown_gate_ok", False)):
            return "NO_FALL"

    fall_confirm = (v_peak_recent >= v_fall_confirm_min) and (angle_change_recent >= angle_change_confirm_min) and (immobile_run >= immobile_confirm_min)
    fall_likely = (v_peak_recent >= 0.8 * v_fall_confirm_min) and (angle_change_recent >= 0.8 * angle_change_confirm_min)

    if fall_confirm and lying:
        return "FALL_CONFIRMED"
    if fall_confirm:
        return "FALL_LIKELY"
    if fall_likely and lying:
        return "FALL_LIKELY"
    return "NO_FALL"

def update_pose_state_machine(pose_rt: Dict, t_now: float):
    vote = pose_rt.get("verdict_vote") or pose_rt.get("verdict_raw") or "UNKNOWN"
    state = pose_rt.get("state", "NO_FALL")

    def set_state(new_state: str, hold_sec: float):
        pose_rt["state"] = new_state
        pose_rt["state_until"] = float(t_now + max(0.0, hold_sec))
        if new_state == "NO_FALL":
            pose_rt["drop_gate_ok"] = False
            pose_rt["vdown_gate_ok"] = False

    angle_curr = pose_rt.get("angle_curr")
    if angle_curr is not None and float(angle_curr) >= float(recover_stand_deg):
        if pose_rt.get("recover_start") is None:
            pose_rt["recover_start"] = float(t_now)
    else:
        pose_rt["recover_start"] = None

    recovered_ok = False
    if pose_rt.get("recover_start") is not None:
        recovered_ok = (t_now - float(pose_rt["recover_start"])) >= float(recover_hold_sec)

    if vote == "FALL_CONFIRMED":
        set_state("FALL_CONFIRMED", confirm_hold_sec)
        return

    if state == "FALL_CONFIRMED":
        until = pose_rt.get("state_until")
        if (until is not None) and (t_now < float(until)):
            return
        if not recovered_ok:
            set_state("FALL_CONFIRMED", 1.0)
            return
        set_state("NO_FALL", 0.0)
        return

    if vote == "FALL_LIKELY":
        if state != "FALL_LIKELY":
            set_state("FALL_LIKELY", likely_hold_sec)
        else:
            set_state("FALL_LIKELY", max(likely_hold_sec, 1.0))
        return

    if state == "FALL_LIKELY":
        until = pose_rt.get("state_until")
        if (until is not None) and (t_now < float(until)):
            return
        set_state("NO_FALL", 0.0)
        return

    set_state("NO_FALL", 0.0)

def update_pose_rt(pose_rt: Dict, pose_model, frame_bgr: np.ndarray, t_now: float):
    if pose_model is None:
        pose_rt["verdict_raw"] = "UNKNOWN"
        pose_rt["verdict_vote"] = "UNKNOWN"
        pose_rt["state"] = "NO_FALL"
        return

    min_step = 1.0 / max(1, int(pose_sample_fps))
    if (t_now - pose_rt["last_pose_t"]) < min_step:
        return
    pose_rt["last_pose_t"] = t_now

    try:
        with POSE_LOCK:
            r = pose_model(frame_bgr, verbose=False, conf=pose_conf)[0]

        hip_mid, shoulder_mid, area, score = select_stable_person(r, pose_rt["hip_ema"])
        if hip_mid is None or shoulder_mid is None:
            pose_rt["last_pose_ok"] = False
            pose_rt["pose_fail_count"] += 1
            return

        if pose_rt["hip_ema"] is None:
            pose_rt["hip_ema"] = hip_mid.copy()
        else:
            pose_rt["hip_ema"] = ema_alpha * hip_mid + (1.0 - ema_alpha) * pose_rt["hip_ema"]

        v_curr = None
        vx_abs = None
        vy_down = None
        if pose_rt["prev_hip_ema"] is not None and pose_rt["prev_t"] is not None:
            dt = max(1e-6, float(t_now - pose_rt["prev_t"]))
            d = pose_rt["hip_ema"] - pose_rt["prev_hip_ema"]
            dx, dy = float(d[0]), float(d[1])
            vx_abs = abs(dx) / dt
            vy_down = max(0.0, dy / dt)
            v_curr = float(np.linalg.norm(d) / dt)

        pose_rt["prev_hip_ema"] = pose_rt["hip_ema"].copy()
        pose_rt["prev_t"] = float(t_now)

        angle_raw = float(_angle_deg_from_horizontal(shoulder_mid, pose_rt["hip_ema"]))
        pose_rt["angle_raw"] = angle_raw
        if pose_rt["angle_ema"] is None:
            pose_rt["angle_ema"] = angle_raw
        else:
            pose_rt["angle_ema"] = float(angle_ema_alpha * angle_raw + (1.0 - angle_ema_alpha) * pose_rt["angle_ema"])
        pose_rt["angle_curr"] = float(pose_rt["angle_ema"])

        pose_rt["v_curr"] = v_curr
        pose_rt["vx_curr"] = vx_abs
        pose_rt["vy_down_curr"] = vy_down

        pose_rt["last_pose_ok"] = True
        pose_rt["pose_ok_count"] += 1

        if v_curr is not None:
            pose_rt["speed_series"].append((float(t_now), float(v_curr)))
        pose_rt["angle_series"].append((float(t_now), float(pose_rt["angle_curr"])))

        if vy_down is not None and vx_abs is not None:
            pose_rt["vy_series"].append((float(t_now), float(vy_down)))
            ratio = float(vy_down / (vx_abs + vy_down + 1e-6))
            pose_rt["vy_ratio_series"].append((float(t_now), ratio))

        prune_series(pose_rt["speed_series"], t_now, pose_rt["window_sec"])
        prune_series(pose_rt["angle_series"], t_now, pose_rt["window_sec"])
        prune_series(pose_rt["vy_series"], t_now, pose_rt["window_sec"])
        prune_series(pose_rt["vy_ratio_series"], t_now, pose_rt["window_sec"])

        pose_rt["v_peak_recent"] = float(max((v for _, v in pose_rt["speed_series"]), default=0.0))
        angs = [a for (_, a) in pose_rt["angle_series"]]
        pose_rt["angle_change_recent"] = float(max(angs) - min(angs)) if angs else 0.0
        pose_rt["vy_peak_recent"] = float(max((v for _, v in pose_rt["vy_series"]), default=0.0))
        pose_rt["vy_ratio_peak_recent"] = float(max((v for _, v in pose_rt["vy_ratio_series"]), default=0.0))
        pose_rt["angle_drop_recent"] = _compute_angle_drop(pose_rt, t_now)

        # immobile hysteresis
        if v_curr is not None:
            if not pose_rt["immobile_active"]:
                if v_curr <= immobile_enter_th:
                    pose_rt["immobile_active"] = True
                    pose_rt["immobile_start"] = float(t_now)
                    pose_rt["immobile_run"] = 0.0
                else:
                    pose_rt["immobile_start"] = None
                    pose_rt["immobile_run"] = 0.0
            else:
                if v_curr >= immobile_exit_th:
                    pose_rt["immobile_active"] = False
                    pose_rt["immobile_start"] = None
                    pose_rt["immobile_run"] = 0.0
                else:
                    if pose_rt["immobile_start"] is None:
                        pose_rt["immobile_start"] = float(t_now)
                    pose_rt["immobile_run"] = float(t_now - pose_rt["immobile_start"])

        # lying hysteresis
        a = pose_rt["angle_curr"]
        if not pose_rt["lying_active"]:
            if a <= float(lying_enter_deg):
                pose_rt["lying_active"] = True
        else:
            if a >= float(lying_exit_deg):
                pose_rt["lying_active"] = False

        # arming gate
        if a is not None and float(a) >= float(upright_angle_th):
            pose_rt["upright_run"] = float(pose_rt["upright_run"] + min_step)
            pose_rt["last_upright_t"] = float(t_now)
            if pose_rt["upright_run"] >= float(min_upright_time_sec):
                pose_rt["armed"] = True
        else:
            pose_rt["upright_run"] = 0.0

        _update_one_shot_gates(pose_rt)

        pose_rt["verdict_raw"] = compute_raw_verdict(pose_rt, t_now)
        pose_rt["verdict_hist"].append((float(t_now), str(pose_rt["verdict_raw"])))
        update_verdict_vote(pose_rt, t_now)
        update_pose_state_machine(pose_rt, t_now)

    except Exception:
        pose_rt["last_pose_ok"] = False
        pose_rt["pose_fail_count"] += 1
        return

def pose_panel_text(pose_rt: Optional[Dict], t_now: float) -> str:
    if pose_rt is None:
        return "POSE: (disabled)"
    ok = int(bool(pose_rt.get("last_pose_ok", False)))
    state = pose_rt.get("state", "NO_FALL")
    vote = pose_rt.get("verdict_vote", "WAITING")
    raw = pose_rt.get("verdict_raw", "WAITING")
    v_peak = float(pose_rt.get("v_peak_recent", 0.0))
    ang_ch = float(pose_rt.get("angle_change_recent", 0.0))
    imm = float(pose_rt.get("immobile_run", 0.0))
    lying = int(bool(pose_rt.get("lying_active", False)))
    drop_ok = int(bool(pose_rt.get("drop_gate_ok", False)))
    armed = int(bool(pose_rt.get("armed", False)))
    return (
        f"t={t_now:5.2f}s | POSE_OK={ok} | STATE={state}\n"
        f"VOTE={vote} | RAW={raw}\n"
        f"v_peak={v_peak:.0f}px/s | angleΔ={ang_ch:.0f}° | imm={imm:.1f}s | lying={lying}\n"
        f"drop_ok={drop_ok} | armed={armed}"
    )

# ============================================================
# CAMERA STATE + EVENT PIPELINE (non-blocking)
# ============================================================
def init_camera_state(buf_seconds: int = 40):
    return {
        "t0_stream": None,
        "frame_idx": 0,
        "buf": deque(maxlen=int(buf_seconds * 35)),  # store frames for clip
        "last_det_t": -1e9,
        "last_dets": [],
        "fall_hold_start": None,
        "detected": False,
        "t0_first": None,
        "t0_confirm": None,
        "cooldown_until": -1e9,

        "pending_export": False,
        "export_t_ref": None,
        "export_ready_at": None,
        "event_wall_time": None,

        "last_event_path": None,
        "last_event_caption": None,
        "last_event_send_ok": None,

        "job_q": queue.Queue(maxsize=3),
        "worker_started": False,

        # outside panel
        "ui_status": "Ready",
        "ui_pose": "POSE: (disabled)",

        # pose compute on camera (optional)
        "pose_rt": init_pose_rt(window_sec=pose_roll_window) if (pose_enabled and compute_pose_on_camera) else None,
    }

def update_confirm_logic(state, t_now: float, dets: List[Det], confirm_sec: float, cooldown: float) -> bool:
    if t_now < state["cooldown_until"]:
        return False

    just_confirmed = False
    fall_present = any(is_fall(_get_name(cls)) for (_, _, _, _, cls, _) in dets)

    if fall_present and state["t0_first"] is None:
        state["t0_first"] = t_now

    if not state["detected"]:
        if fall_present:
            if state["fall_hold_start"] is None:
                state["fall_hold_start"] = t_now
            else:
                if (t_now - state["fall_hold_start"]) >= confirm_sec:
                    state["detected"] = True
                    state["t0_confirm"] = t_now
                    state["cooldown_until"] = t_now + cooldown
                    state["fall_hold_start"] = None
                    just_confirmed = True
        else:
            state["fall_hold_start"] = None

    return just_confirmed

def build_caption(event_wall_ts: float) -> str:
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event_wall_ts))
    return f"🚨 FALL CONFIRMED\nTime: {time_str}"

def camera_worker_loop(job_q: queue.Queue):
    while True:
        job = job_q.get()
        if job is None:
            break
        try:
            clip_frames = job["clip_frames"]
            caption = job["caption"]
            tg_token = job["tg_token"]
            tg_chat = job["tg_chat"]
            out_dir = job["out_dir"]
            state_ref = job["state_ref"]

            out_path_fixed = None
            if clip_frames and len(clip_frames) >= 5:
                out_raw = os.path.join(out_dir, f"fall_event_{int(job['event_wall_ts'])}_raw.mp4")
                fps_est = _estimate_fps(clip_frames, default_fps=30.0)
                written = _write_mp4_raw(clip_frames, out_raw, fps_est)
                if written:
                    out_path_fixed = _try_fix_for_telegram(written)

            send_ok = None
            if tg_token and tg_chat:
                ok_msg = tg_send_message(tg_token, tg_chat, caption)
                ok_vid = True
                if out_path_fixed and os.path.exists(out_path_fixed):
                    ok_vid = tg_send_video(tg_token, tg_chat, out_path_fixed, caption="")
                send_ok = bool(ok_msg and ok_vid)

            state_ref["last_event_path"] = out_path_fixed
            state_ref["last_event_caption"] = caption
            state_ref["last_event_send_ok"] = send_ok

        except Exception:
            state_ref["last_event_send_ok"] = False
        finally:
            job_q.task_done()

def ensure_camera_worker(state):
    if state.get("worker_started", False):
        return
    t = threading.Thread(target=camera_worker_loop, args=(state["job_q"],), daemon=True)
    t.start()
    state["worker_started"] = True

def schedule_export_if_ready(state, t_now: float, tg_token: Optional[str], tg_chat: Optional[str]):
    if not state.get("pending_export", False):
        return

    ready_at = state.get("export_ready_at")
    if ready_at is None:
        return
    if t_now < float(ready_at):
        remain = max(0.0, float(ready_at) - t_now)
        state["ui_status"] = f"CONFIRMED ✅ | Collecting post frames... remaining={remain:.1f}s"
        return

    t_ref = float(state["export_t_ref"])
    clip_frames = _extract_window(state["buf"], t_ref, float(pre_sec), float(post_sec))
    caption = build_caption(state["event_wall_time"] or time.time())

    ensure_camera_worker(state)
    job = {
        "clip_frames": clip_frames,
        "caption": caption,
        "tg_token": tg_token,
        "tg_chat": tg_chat,
        "out_dir": OUT_DIR,
        "event_wall_ts": state["event_wall_time"] or time.time(),
        "state_ref": state,
    }
    try:
        state["job_q"].put_nowait(job)
        state["ui_status"] = "CONFIRMED ✅ | Export job queued (Telegram sending in background)"
    except queue.Full:
        state["last_event_send_ok"] = False
        state["last_event_caption"] = caption
        state["ui_status"] = "CONFIRMED ✅ | Export queue FULL (skip sending)"

    state["pending_export"] = False
    state["export_t_ref"] = None
    state["export_ready_at"] = None
    state["detected"] = False
    state["t0_first"] = None
    state["t0_confirm"] = None

# ============================================================
# VIDEO MODE (pose overlay debug)
# ============================================================
def scan_videos_recursive(root_dir: str, exts: set) -> List[str]:
    res = []
    if not os.path.exists(root_dir):
        return res
    for dp, _, fn in os.walk(root_dir):
        for f in fn:
            p = os.path.join(dp, f)
            if os.path.splitext(p)[1].lower() in exts:
                res.append(os.path.abspath(p))
    res.sort()
    return res

def draw_pose_overlay_on_frame(img_bgr: np.ndarray, pose_rt: Dict, t_now: float) -> np.ndarray:
    # Video mode: keep overlay on video (debug)
    h, w = img_bgr.shape[:2]
    lines = pose_panel_text(pose_rt, t_now).split("\n")
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.66
    thickness = 2
    line_h = 24
    panel_h = pad * 2 + line_h * len(lines)
    panel_w = min(w - 20, 1500)
    x0, y0 = 10, h - panel_h - 10
    cv2.rectangle(img_bgr, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    y = y0 + pad + 18
    for ln in lines:
        cv2.putText(img_bgr, ln, (x0 + pad, y), font, scale, (255, 255, 255), thickness)
        y += line_h
    return img_bgr

if source_radio == VIDEO:
    st.subheader("Video (debug/test)")

    videos_abs = scan_videos_recursive(VIDEO_DIR, VIDEO_EXTS)
    if not videos_abs:
        st.warning(f"No videos found in '{VIDEO_DIR}' (including subfolders).")
        st.stop()

    mapping: Dict[str, str] = {}
    labels: List[str] = []
    for p in videos_abs:
        rel = os.path.relpath(p, VIDEO_DIR).replace("\\", "/")
        labels.append(rel)
        mapping[rel] = p

    query = st.sidebar.text_input("Search video", value="")
    filtered = [x for x in labels if query.lower() in x.lower()] if query else labels
    if not filtered:
        st.warning("No videos match the search.")
        st.stop()

    sel = st.selectbox("Select video", filtered, index=0)
    video_path = mapping[sel]

    if st.button("Run on video"):
        if not pose_enabled or (pose_model is None):
            st.error("Pose model is not loaded or pose is disabled.")
            st.stop()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Cannot open video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1e-3 else 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pose_rt = init_pose_rt(window_sec=pose_roll_window)

        out_raw = os.path.join(OUT_DIR, f"video_pose_final_{int(time.time())}_raw.mp4")
        vw = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            st.error("Cannot create output video writer.")
            cap.release()
            st.stop()

        prog = st.progress(0.0)
        status = st.empty()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_now = frame_idx / fps
            frame_idx += 1

            update_pose_rt(pose_rt, pose_model, frame, t_now)
            annotated = draw_pose_overlay_on_frame(frame.copy(), pose_rt, t_now)
            vw.write(annotated)

            if total_frames > 0:
                prog.progress(min(1.0, frame_idx / total_frames))

            status.write(
                f"{frame_idx}/{total_frames} | STATE={pose_rt.get('state')} VOTE={pose_rt.get('verdict_vote')} "
                f"| v_peak={pose_rt.get('v_peak_recent',0):.0f} angleΔ={pose_rt.get('angle_change_recent',0):.0f} "
                f"| imm={pose_rt.get('immobile_run',0):.1f}s"
            )

        cap.release()
        vw.release()

        out_fixed = out_raw.replace("_raw.mp4", ".mp4")
        ok_fix = ffmpeg_fix_h264_yuv420p(out_raw, out_fixed)
        if not ok_fix:
            out_fixed = out_raw

        st.success("Done.")
        st.markdown("**Output video (pose overlay realtime)**")
        st.video(out_fixed)

# ============================================================
# CAMERA MODE (final demo + frameskip + outside info)
# ============================================================
elif source_radio == CAMERA:
    st.subheader("Camera (final demo)")

    tg_token, tg_chat = read_telegram_credentials()
    if (tg_token is None) or (tg_chat is None):
        st.warning("Telegram credentials not found in token.txt (2 lines: token, chat_id). Alerts will be skipped.")

    # Live info panel (outside video)
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### Live Status (outside video)")
        status_box = st.empty()
    with colB:
        st.markdown("### Pose Info (outside video)")
        pose_box = st.empty()

    if "cam_state" not in st.session_state:
        buf_seconds = int(pre_sec + post_sec + 10)
        st.session_state.cam_state = init_camera_state(buf_seconds=buf_seconds)

    state = st.session_state.cam_state

    # resize buffer if settings changed
    desired_buf_seconds = int(pre_sec + post_sec + 10)
    desired_maxlen = int(desired_buf_seconds * 35)
    if state["buf"].maxlen != desired_maxlen:
        state["buf"] = deque(state["buf"], maxlen=desired_maxlen)

    if state.get("t0_stream") is None:
        state["t0_stream"] = time.time()

    # pose compute on camera is optional
    if pose_enabled and compute_pose_on_camera and (pose_model is not None):
        if state.get("pose_rt") is None:
            state["pose_rt"] = init_pose_rt(window_sec=pose_roll_window)
    else:
        state["pose_rt"] = None

    ensure_camera_worker(state)

    def camera_callback(frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        t_wall = time.time()
        t_now = t_wall - float(state["t0_stream"])

        # update frame index
        state["frame_idx"] += 1
        idx = int(state["frame_idx"])

        # buffer store (for clip). Default store every frame.
        if (idx % int(buffer_store_step)) == 0:
            state["buf"].append((t_now, img.copy()))

        # frameskip: only run heavy compute every N frames
        do_process = (idx % int(camera_frame_step)) == 0

        # optional pose compute (for outside panel only)
        if do_process and state.get("pose_rt") is not None:
            update_pose_rt(state["pose_rt"], pose_model, img, t_now)
            state["ui_pose"] = pose_panel_text(state["pose_rt"], t_now)
        elif state.get("pose_rt") is None:
            state["ui_pose"] = "POSE: (disabled)"
        # if skipped, keep last ui_pose

        # detection (frameskip + detection_interval)
        if do_process and ((t_now - state["last_det_t"]) >= detection_interval):
            with MODEL_LOCK:
                res = model(img, verbose=False, conf=confidence_value)[0]
            dets = pack_detections(res.boxes)
            state["last_dets"] = dets
            state["last_det_t"] = t_now

            just_confirmed = update_confirm_logic(state, t_now, dets, confirm_seconds, cooldown_sec)
            if just_confirmed:
                state["event_wall_time"] = t_wall
                state["pending_export"] = True
                state["export_t_ref"] = float(state["t0_confirm"])
                state["export_ready_at"] = float(state["t0_confirm"] + float(post_sec))

        # schedule export when post frames collected
        schedule_export_if_ready(state, t_now, tg_token, tg_chat)

        # update status text (outside)
        fall_present = any(is_fall(_get_name(cls)) for (_, _, _, _, cls, _) in state.get("last_dets", []))
        hold_txt = ""
        if state.get("fall_hold_start") is not None and (not state.get("detected", False)) and fall_present:
            held = max(0.0, t_now - float(state["fall_hold_start"]))
            remain = max(0.0, float(confirm_seconds) - held)
            hold_txt = f" | hold={held:.1f}s remain={remain:.1f}s"

        cd = max(0.0, float(state.get("cooldown_until", -1e9)) - t_now)
        cd_txt = f" | cooldown={cd:.1f}s" if cd > 0 else ""

        event_txt = "CONFIRMED ✅" if state.get("detected", False) else ("DETECTING..." if fall_present else "NO_FALL")
        state["ui_status"] = (
            f"t={t_now:5.2f}s | frame={idx} | proc_every={camera_frame_step} | det_interval={detection_interval:.2f}s\n"
            f"EVENT={event_txt}{hold_txt}{cd_txt}\n"
            f"pending_export={int(bool(state.get('pending_export')))}"
        )

        # draw ONLY detection boxes on camera video (no long pose overlay on video)
        annotated = draw_boxes_camera(
            img.copy(),
            state.get("last_dets", []),
            bool(state.get("detected", False)),
            state.get("fall_hold_start"),
            t_now,
            confirm_seconds
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # basic metrics row
    cols = st.columns(5)
    with cols[0]:
        st.metric("Hold", f"{confirm_seconds:.1f}s")
    with cols[1]:
        st.metric("Pre/Post", f"{int(pre_sec)} / {int(post_sec)} s")
    with cols[2]:
        st.metric("FrameSkip", f"every {int(camera_frame_step)}")
    with cols[3]:
        st.metric("Det interval", f"{detection_interval:.2f}s")
    with cols[4]:
        st.metric("Cooldown", f"{cooldown_sec:.1f}s")

    webrtc_ctx = webrtc_streamer(
        key="camera_fall_detector_final_frameskip",
        video_frame_callback=camera_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Live outside panels (updates while playing)
    # (This loop runs in main thread; it refreshes the boxes periodically.)
    if webrtc_ctx.state.playing:
        for _ in range(2000000):
            if not webrtc_ctx.state.playing:
                break
            status_box.code(state.get("ui_status", ""))
            pose_box.code(state.get("ui_pose", "POSE: (disabled)"))
            time.sleep(0.2)
    else:
        status_box.code(state.get("ui_status", "Ready"))
        pose_box.code(state.get("ui_pose", "POSE: (disabled)"))

    with st.expander("Last event (debug)", expanded=False):
        if state.get("last_event_caption"):
            st.write(state.get("last_event_caption"))
        if state.get("last_event_path") and os.path.exists(state.get("last_event_path")):
            st.video(state.get("last_event_path"))
        if state.get("last_event_send_ok") is True:
            st.success("Telegram: sent successfully ✅")
        elif state.get("last_event_send_ok") is False:
            st.error("Telegram: failed ❌ (check token.txt / network)")
        else:
            st.info("Telegram: not sent yet (no event / missing credentials).")
