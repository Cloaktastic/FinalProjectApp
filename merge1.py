import os
import cv2
import av
import time
import math
import numpy as np
import streamlit as st
import subprocess
import requests
from PIL import Image
from ultralytics import YOLO
from collections import deque
from typing import Optional, List, Tuple, Dict
from streamlit_webrtc import webrtc_streamer
import matplotlib.pyplot as plt

# ----------------------------
# Sources
# ----------------------------
IMAGE = "Image"
VIDEO = "Video"
CAMERA = "Camera"
SOURCES_LIST = [IMAGE, VIDEO, CAMERA]

VIDEO_DIR = "videos"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}

MODEL_PATH = "models/100_epochs.pt"        # fall + person detector
POSE_MODEL_PATH = "models/yolo11n-pose.pt" # pose model (your path)

TOKEN_FILE = "token.txt"
OUT_DIR = "videos"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Page Layout
# ----------------------------
st.set_page_config(
    page_title="Elderly Fall Detection",
    page_icon="🚨",
    layout="wide"
)
st.title("🚨 Elderly Fall Detection")
st.caption("YOLO-based Fall & Person Detection (Image / Video / Camera)")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Model")
confidence_value = st.sidebar.slider("Confidence", 0.25, 1.00, 0.40, 0.01)

st.sidebar.header("Compute")
device_option = st.sidebar.selectbox("Run on", ["Auto", "CPU", "GPU (CUDA)"], index=0)

st.sidebar.header("Behavior")
confirm_seconds = st.sidebar.slider("Confirm fall after (sec)", 1.0, 10.0, 3.0, 0.5)
pre_sec = float(st.sidebar.slider("Clip pre (sec)", 1, 30, 8, 1))
post_sec = float(st.sidebar.slider("Clip post (sec)", 1, 30, 8, 1))
detection_interval = float(st.sidebar.slider("Detection interval (sec)", 0.02, 0.30, 0.05, 0.01))
cooldown_sec = float(st.sidebar.slider("Cooldown after confirm (sec)", 2, 30, 10, 1))

st.sidebar.header("Pose verification (YOLO11 Pose)")
pose_enabled = st.sidebar.toggle("Enable pose verification", value=True)
pose_conf = st.sidebar.slider("Pose confidence", 0.05, 0.90, 0.30, 0.01)

# Pose window references FIRST
pose_pre = st.sidebar.slider("Pose window pre (sec)", 0.5, 10.0, 3.0, 0.5)
pose_post = st.sidebar.slider("Pose window post (sec)", 0.5, 10.0, 4.0, 0.5)
pose_sample_fps = st.sidebar.slider("Pose sample FPS", 1, 15, 6, 1)

st.sidebar.subheader("Pose smoothing")
ema_alpha = st.sidebar.slider("EMA alpha (hip smoothing)", 0.05, 0.90, 0.35, 0.05)

st.sidebar.subheader("Pose decision thresholds (tune)")
v_fall_confirm_min = st.sidebar.slider("v_peak FALL confirm (px/s)", 40, 600, 160, 10)
angle_change_confirm_min = st.sidebar.slider("Angle change confirm (deg)", 5, 90, 18, 1)
immobile_confirm_min = st.sidebar.slider("Immobile time confirm (sec)", 0.0, 6.0, 0.8, 0.1)
immobile_speed_th = st.sidebar.slider("Immobile speed th (px/s)", 5, 200, 45, 5)

v_fall_likely_min = st.sidebar.slider("v_peak FALL likely (px/s)", 40, 600, 90, 10)
angle_change_likely_min = st.sidebar.slider("Angle change likely (deg)", 3, 60, 7, 1)

lying_angle_max = st.sidebar.slider("Lying angle max (deg from horizontal)", 0, 45, 18, 1)
lying_ratio_min = st.sidebar.slider("Lying ratio min (w/h)", 1.0, 3.5, 1.4, 0.05)

st.sidebar.header("Input")
source_radio = st.sidebar.radio("Select source", SOURCES_LIST)

# ----------------------------
# Device
# ----------------------------
def resolve_device(opt: str) -> str:
    if opt == "CPU":
        return "cpu"
    if opt == "GPU (CUDA)":
        return "cuda"
    return "cuda"

# ----------------------------
# Load models (cached)
# ----------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
    device = resolve_device(device_option)
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)
        st.sidebar.warning("CUDA not available. Fallback to CPU.")
    st.sidebar.caption(f"Device: **{device}**")
except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {MODEL_PATH}")
    st.exception(e)
    st.stop()

pose_model = None
if pose_enabled and os.path.exists(POSE_MODEL_PATH):
    try:
        pose_model = load_model(POSE_MODEL_PATH)
        try:
            pose_model.to(device)
        except Exception:
            pose_model.to("cpu")
    except Exception:
        pose_model = None

# ----------------------------
# Telegram (camera only)
# ----------------------------
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

# ----------------------------
# Utils
# ----------------------------
def is_fall(name: str) -> bool:
    return str(name).lower() == "fall"

def is_person(name: str) -> bool:
    return str(name).lower() in ["person", "people", "human"]

def ffmpeg_fix_h264_yuv420p(in_path: str, out_path: str):
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
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def scan_videos_recursive(root_dir: str, exts: set) -> List[str]:
    results: List[str] = []
    if not os.path.isdir(root_dir):
        return results
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in exts:
                results.append(os.path.join(dirpath, fn))
    results.sort(key=lambda p: p.lower())
    return results

# ----------------------------
# Packed detections (avoid torch objects in state)
# ----------------------------
Det = Tuple[int, int, int, int, int, float]  # x1,y1,x2,y2,cls,conf

def pack_detections(boxes) -> List[Det]:
    dets: List[Det] = []
    if boxes is None or len(boxes) == 0:
        return dets
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy()
        cls_id = int(b.cls[0].detach().cpu().numpy())
        conf = float(b.conf[0].detach().cpu().numpy())
        dets.append((int(x1), int(y1), int(x2), int(y2), cls_id, conf))
    return dets

def _get_name(cls_id: int) -> str:
    if isinstance(model.names, (list, tuple)):
        return model.names[cls_id] if 0 <= cls_id < len(model.names) else str(cls_id)
    return model.names.get(cls_id, str(cls_id))

def draw_boxes(
    img_bgr: np.ndarray,
    dets: List[Det],
    detected_confirmed: bool,
    motionless_start_t: Optional[float],
    t_now: float,
    confirm_seconds_: float,
    overlay_text: Optional[str] = None
) -> np.ndarray:
    if dets is None or len(dets) == 0:
        return img_bgr

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

        if is_fall(class_name) and (not detected_confirmed) and (motionless_start_t is not None):
            held = max(0.0, t_now - motionless_start_t)
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

    if overlay_text:
        cv2.rectangle(img_bgr, (10, h - 60), (min(w - 10, 10 + 1050), h - 10), (0, 0, 0), -1)
        cv2.putText(img_bgr, overlay_text, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img_bgr

# ----------------------------
# State logic (FIRST ref)
# ----------------------------
def init_state(buf_maxlen: int):
    return {
        "buf": deque(maxlen=buf_maxlen),
        "last_det_t": -1e9,
        "last_dets": [],
        "motionless_start": None,
        "detected": False,
        "t0_first": None,
        "t0_confirm": None,
        "cooldown_until": -1e9,
        "pose_done": False,
        "pose_verdict": None,
        "pose_result": None,
    }

def update_confirm_logic(state, t_now: float, dets: List[Det], confirm_sec: float, cooldown: float):
    if t_now < state["cooldown_until"]:
        return

    fall_present = any(is_fall(_get_name(cls)) for (_, _, _, _, cls, _) in dets)

    if fall_present and state["t0_first"] is None:
        state["t0_first"] = t_now

    if not state["detected"]:
        if fall_present:
            if state["motionless_start"] is None:
                state["motionless_start"] = t_now
            else:
                if (t_now - state["motionless_start"]) >= confirm_sec:
                    state["detected"] = True
                    state["t0_confirm"] = t_now
                    state["cooldown_until"] = t_now + cooldown
                    state["motionless_start"] = None
        else:
            state["motionless_start"] = None

# ============================================================
# POSE ANALYSIS (EMA smoothing)
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

def _bbox_ratio_from_pose(kps_xy: np.ndarray, kps_conf: np.ndarray, conf_th: float = 0.15) -> Optional[float]:
    valid = kps_conf >= conf_th
    if valid.sum() < 4:
        return None
    pts = kps_xy[valid]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return w / h

def clamp_pose_window(frames_with_t: List[Tuple[float, np.ndarray]], t_ref: float, pre: float, post: float) -> List[Tuple[float, np.ndarray]]:
    if not frames_with_t:
        return []
    frames_with_t = sorted(frames_with_t, key=lambda x: x[0])
    t_min = frames_with_t[0][0]
    t_max = frames_with_t[-1][0]
    a = max(t_min, t_ref - pre)
    b = min(t_max, t_ref + post)
    return [(t, fr) for (t, fr) in frames_with_t if a <= t <= b]

def analyze_pose_window(frames_with_t: List[Tuple[float, np.ndarray]]) -> Tuple[str, Dict]:
    if pose_model is None:
        return "UNKNOWN", {"reason": "pose_model_not_loaded"}
    if not frames_with_t:
        return "UNKNOWN", {"reason": "no_frames"}

    sampled: List[Tuple[float, np.ndarray]] = []
    last_keep = -1e18
    step = 1.0 / max(1, int(pose_sample_fps))
    for (t, fr) in frames_with_t:
        if (t - last_keep) >= step:
            sampled.append((t, fr))
            last_keep = t

    frames_sampled = len(sampled)
    frames_pose_ok = 0
    frames_pose_empty = 0

    hip_raw: List[Tuple[float, np.ndarray]] = []
    angle_series: List[Tuple[float, float]] = []
    ratio_series: List[Optional[float]] = []

    for (t, fr) in sampled:
        h, w = fr.shape[:2]
        scale = 960.0 / max(1, w)
        fr_in = cv2.resize(fr, (int(w * scale), int(h * scale))) if scale < 1.0 else fr

        r = pose_model(fr_in, verbose=False, conf=pose_conf)[0]

        if r.keypoints is None or r.keypoints.xy is None:
            frames_pose_empty += 1
            continue

        kxy = r.keypoints.xy
        n_people = int(kxy.shape[0]) if hasattr(kxy, "shape") else 0
        if n_people <= 0:
            frames_pose_empty += 1
            continue

        idx = 0
        if r.boxes is not None and len(r.boxes) > 0:
            areas = []
            for b in r.boxes:
                xyxy = b.xyxy[0].detach().cpu().numpy()
                areas.append(float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])))
            idx = int(np.argmax(areas))
            if idx >= n_people:
                idx = 0

        kps_xy = kxy[idx].detach().cpu().numpy()
        if r.keypoints.conf is not None and r.keypoints.conf.shape[0] == n_people:
            kps_conf = r.keypoints.conf[idx].detach().cpu().numpy()
        else:
            kps_conf = np.ones((kps_xy.shape[0],), dtype=np.float32)

        if fr_in is not fr:
            kps_xy = kps_xy / scale

        if max(LS, RS, LH, RH) >= kps_xy.shape[0]:
            frames_pose_empty += 1
            continue
        if (kps_conf[LS] < 0.15) or (kps_conf[RS] < 0.15) or (kps_conf[LH] < 0.15) or (kps_conf[RH] < 0.15):
            frames_pose_empty += 1
            continue

        shoulder_mid = _mid(kps_xy[LS], kps_xy[RS])
        hip_mid = _mid(kps_xy[LH], kps_xy[RH])
        ang = _angle_deg_from_horizontal(shoulder_mid, hip_mid)

        hip_raw.append((t, hip_mid))
        angle_series.append((t, ang))
        ratio_series.append(_bbox_ratio_from_pose(kps_xy, kps_conf, conf_th=0.15))
        frames_pose_ok += 1

    if len(hip_raw) < 3:
        return "UNKNOWN", {
            "reason": "insufficient_pose_points",
            "n_pose": len(hip_raw),
            "frames_sampled": frames_sampled,
            "frames_pose_ok": frames_pose_ok,
            "frames_pose_empty": frames_pose_empty,
            "pose_conf_used": pose_conf
        }

    hip_smooth: List[Tuple[float, np.ndarray]] = []
    ema = hip_raw[0][1].astype(np.float32).copy()
    hip_smooth.append((hip_raw[0][0], ema.copy()))
    for i in range(1, len(hip_raw)):
        t_i, p_i = hip_raw[i]
        ema = (1.0 - ema_alpha) * ema + ema_alpha * p_i.astype(np.float32)
        hip_smooth.append((t_i, ema.copy()))

    speeds: List[Tuple[float, float]] = []
    for i in range(1, len(hip_smooth)):
        t0_, p0 = hip_smooth[i - 1]
        t1_, p1 = hip_smooth[i]
        dt = max(1e-6, float(t1_ - t0_))
        v = float(np.linalg.norm(p1 - p0)) / dt
        speeds.append((t1_, v))

    v_peak = max(v for _, v in speeds) if speeds else 0.0

    immobile_time = 0.0
    run_start = None
    last_t = None
    for (t, v) in speeds:
        if v < immobile_speed_th:
            if run_start is None:
                run_start = t
            last_t = t
        else:
            if run_start is not None and last_t is not None:
                immobile_time = max(immobile_time, float(last_t - run_start))
            run_start = None
            last_t = None
    if run_start is not None and last_t is not None:
        immobile_time = max(immobile_time, float(last_t - run_start))

    angles = [a for _, a in angle_series]
    angle_min = float(np.min(angles))
    angle_max = float(np.max(angles))
    angle_change = float(angle_max - angle_min)

    ratios = [r for r in ratio_series if r is not None]
    ratio_med = float(np.median(ratios)) if ratios else None

    fall_confirm = (v_peak >= v_fall_confirm_min) and (angle_change >= angle_change_confirm_min) and (immobile_time >= immobile_confirm_min)
    fall_likely = (v_peak >= v_fall_likely_min) and (angle_change >= angle_change_likely_min)

    angle_near_horizontal = (angle_min <= lying_angle_max)
    ratio_lying = (ratio_med is not None and ratio_med >= lying_ratio_min)
    lying_strong = angle_near_horizontal and ratio_lying and (v_peak < v_fall_likely_min) and (angle_change < angle_change_likely_min)

    if fall_confirm:
        verdict = "FALL"
    elif fall_likely and (not lying_strong):
        verdict = "FALL"
    elif lying_strong:
        verdict = "LYING/SLEEP"
    else:
        verdict = "UNKNOWN"

    angle_ts = [{"t": float(t), "angle_deg": float(a)} for (t, a) in angle_series]
    speed_ts = [{"t": float(t), "speed_px_s": float(v)} for (t, v) in speeds]

    metrics = {
        "n_pose": len(hip_raw),
        "v_peak_px_s": float(v_peak),
        "immobile_time_s": float(immobile_time),
        "angle_min_deg_from_horizontal": float(angle_min),
        "angle_max_deg_from_horizontal": float(angle_max),
        "angle_change_deg": float(angle_change),
        "ratio_median_w_h": ratio_med,
        "window_duration_s": float(hip_raw[-1][0] - hip_raw[0][0]),
        "frames_sampled": frames_sampled,
        "frames_pose_ok": frames_pose_ok,
        "frames_pose_empty": frames_pose_empty,
        "pose_conf_used": pose_conf,
        "ema_alpha": float(ema_alpha),
        "fall_confirm_rule": bool(fall_confirm),
        "fall_likely_rule": bool(fall_likely),
        "lying_strong_rule": bool(lying_strong),
        "angle_series": angle_ts,
        "speed_series": speed_ts,
    }
    return verdict, metrics

def pose_overlay_text(verdict: str, metrics: Optional[Dict]) -> str:
    if not metrics:
        return f"POSE: {verdict}"
    if "reason" in metrics:
        return f"POSE: {verdict} | reason={metrics.get('reason')} | ok={metrics.get('frames_pose_ok',0)}/{metrics.get('frames_sampled',0)}"
    return (f"POSE: {verdict} | v_peak={metrics.get('v_peak_px_s', 0):.0f}px/s | "
            f"immobile={metrics.get('immobile_time_s', 0):.1f}s | "
            f"angleΔ={metrics.get('angle_change_deg', 0):.0f}°")

# ----------------------------
# Plot helper: reference FIRST, still draw CONFIRMED + v_peak line
# ----------------------------
def _draw_vline_with_label(x: float, label: str):
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    plt.axvline(x, linestyle="--")
    y = ymax - 0.05 * (ymax - ymin)
    plt.text(x, y, f" {label}", rotation=90, va="top", ha="left")

def _draw_hline_vpeak(vpeak: float):
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    plt.axhline(vpeak, linewidth=1.0, linestyle=":")
    y = vpeak
    x = xmin + 0.02 * (xmax - xmin)
    plt.text(x, y, f" v_peak={vpeak:.0f}px/s", va="bottom", ha="left")

def _explain_lines():
    st.caption("Angle reference: 0° = lying (horizontal), 90° = standing (vertical).")
    st.caption("FIRST = first fall bbox, CONFIRMED = held for configured seconds (motionless rule).")

def plot_pose_graphs_ref_first(metrics: Dict, t0_first: float, t0_confirm: Optional[float]):
    ang = metrics.get("angle_series", [])
    spd = metrics.get("speed_series", [])
    if (not isinstance(ang, list)) or (not isinstance(spd, list)) or (len(ang) < 2) or (len(spd) < 2):
        return

    t_a = np.array([d["t"] - t0_first for d in ang], dtype=float)
    y_a = np.array([d["angle_deg"] for d in ang], dtype=float)

    t_v = np.array([d["t"] - t0_first for d in spd], dtype=float)
    y_v = np.array([d["speed_px_s"] for d in spd], dtype=float)

    ia = np.argsort(t_a); t_a, y_a = t_a[ia], y_a[ia]
    iv = np.argsort(t_v); t_v, y_v = t_v[iv], y_v[iv]

    x_first = 0.0
    x_confirm = None
    if t0_confirm is not None:
        x_confirm = float(t0_confirm - t0_first)

    fig1 = plt.figure()
    plt.plot(t_a, y_a)
    plt.xlabel("Time (s) relative to FIRST (t=0)")
    plt.ylabel("Body angle (deg from horizontal)")
    plt.title("Body angle vs time")
    _draw_vline_with_label(x_first, "FIRST")
    if x_confirm is not None:
        _draw_vline_with_label(x_confirm, "CONFIRMED")
    st.pyplot(fig1, clear_figure=True)
    _explain_lines()

    vpeak = float(metrics.get("v_peak_px_s", 0.0))

    fig2 = plt.figure()
    plt.plot(t_v, y_v)
    plt.xlabel("Time (s) relative to FIRST (t=0)")
    plt.ylabel("Hip velocity (px/s)")
    plt.title("Velocity vs time")
    _draw_vline_with_label(x_first, "FIRST")
    if x_confirm is not None:
        _draw_vline_with_label(x_confirm, "CONFIRMED")
    if vpeak > 0:
        _draw_hline_vpeak(vpeak)
    st.pyplot(fig2, clear_figure=True)
    _explain_lines()

# ============================================================
# IMAGE MODE
# ============================================================
if source_radio == IMAGE:
    st.subheader("Image")
    c1, c2 = st.columns([1, 1])
    source_image = st.sidebar.file_uploader("Upload image", type=("jpg", "png", "jpeg", "bmp", "webp"))

    with c1:
        if source_image:
            uploaded_image = Image.open(source_image)
            st.image(uploaded_image, use_container_width=True)
        else:
            st.info("Upload an image to run detection.")

    with c2:
        if source_image and st.button("Run"):
            uploaded_image = Image.open(source_image)
            img_bgr = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
            res = model(img_bgr, verbose=False, conf=confidence_value)[0]
            dets = pack_detections(res.boxes)
            annotated = draw_boxes(img_bgr.copy(), dets, False, None, 0.0, confirm_seconds, None)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

# ============================================================
# VIDEO MODE (NO TELEGRAM)
# ============================================================
elif source_radio == VIDEO:
    st.subheader("Video")

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
    filtered = [lb for lb in labels if query.strip().lower() in lb.lower()] if query.strip() else labels
    selected_label = st.sidebar.selectbox("Choose video", filtered)
    video_path = mapping[selected_label]

    with open(video_path, "rb") as f:
        st.video(f.read())

    if st.button("Process video"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Cannot open video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        buf_maxlen = int((max(pre_sec + post_sec, pose_pre + pose_post) + 4) * fps)
        state = init_state(buf_maxlen)

        out_raw = os.path.join(OUT_DIR, f"annot_video_{int(time.time())}_raw.mp4")
        vw = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        prog = st.progress(0.0)
        status = st.empty()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_now = frame_idx / fps
            frame_idx += 1

            state["buf"].append((t_now, frame.copy()))

            if (t_now - state["last_det_t"]) >= detection_interval:
                res = model(frame, verbose=False, conf=confidence_value)[0]
                dets = pack_detections(res.boxes)
                state["last_dets"] = dets
                state["last_det_t"] = t_now
                update_confirm_logic(state, t_now, dets, confirm_seconds, cooldown_sec)

            overlay = None
            if state["pose_done"]:
                overlay = pose_overlay_text(state["pose_verdict"], state["pose_result"])

            annotated = draw_boxes(frame.copy(), state["last_dets"], state["detected"], state["motionless_start"], t_now, confirm_seconds, overlay)
            vw.write(annotated)

            if total_frames > 0:
                prog.progress(min(1.0, frame_idx / total_frames))
            status.write(f"{frame_idx}/{total_frames} | FIRST={state['t0_first']} | CONF={state['t0_confirm']} | PoseDone={state['pose_done']}")

        cap.release()
        vw.release()

        out_fixed = out_raw.replace("_raw.mp4", ".mp4")
        ffmpeg_fix_h264_yuv420p(out_raw, out_fixed)

        # Pose: run around FIRST even if CONFIRM not reached
        if pose_enabled and (pose_model is not None) and (state["t0_first"] is not None):
            frames_all = list(state["buf"])
            frames_all.sort(key=lambda x: x[0])
            window = clamp_pose_window(frames_all, state["t0_first"], pose_pre, pose_post)
            vrd, met = analyze_pose_window(window)
            state["pose_done"] = True
            state["pose_verdict"] = vrd
            state["pose_result"] = met

        st.success("Done.")
        st.markdown("**Annotated video**")
        st.video(out_fixed)

        if state["pose_done"] and state["pose_result"] and (state["t0_first"] is not None):
            st.markdown("### Pose graphs")
            plot_pose_graphs_ref_first(
                metrics=state["pose_result"],
                t0_first=state["t0_first"],
                t0_confirm=state["t0_confirm"]
            )

        st.markdown("## Pose conclusion")
        if state["t0_first"] is None:
            st.warning("No FALL bbox detected → pose does not run.")
        else:
            if not pose_enabled:
                st.info("Pose is disabled.")
            elif pose_model is None:
                st.error(f"Pose enabled but pose model not loaded: {POSE_MODEL_PATH}")
            else:
                if state["t0_confirm"] is None:
                    st.info("CONFIRMED not reached (video too short / hold not satisfied). Pose still ran around FIRST.")
                st.write("**Verdict:**", state["pose_verdict"])
                st.write("**Metrics:**")
                st.json(state["pose_result"])
                st.info(pose_overlay_text(state["pose_verdict"], state["pose_result"]))

# ============================================================
# CAMERA MODE (Telegram only here) - unchanged
# ============================================================
elif source_radio == CAMERA:
    st.subheader("Camera")

    tg_token, tg_chat = read_telegram_credentials()
    if (tg_token is None) or (tg_chat is None):
        st.warning("Telegram credentials not found in token.txt (2 lines: token, chat_id). Alerts will be skipped.")

    if "fall_state" not in st.session_state:
        approx_fps = 30
        buf_maxlen = int((pre_sec + post_sec + 4) * approx_fps)
        st.session_state.fall_state = init_state(buf_maxlen)
    state = st.session_state.fall_state

    def camera_callback(frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        t_wall = time.time()

        state["buf"].append((t_wall, img.copy()))

        if (t_wall - state["last_det_t"]) >= detection_interval:
            res = model(img, verbose=False, conf=confidence_value)[0]
            dets = pack_detections(res.boxes)
            state["last_dets"] = dets
            state["last_det_t"] = t_wall
            update_confirm_logic(state, t_wall, dets, confirm_seconds, cooldown_sec)

        overlay = None
        if state["pose_done"]:
            overlay = pose_overlay_text(state["pose_verdict"], state["pose_result"])

        annotated = draw_boxes(img.copy(), state["last_dets"], state["detected"], state["motionless_start"], t_wall, confirm_seconds, overlay)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Confirm (sec)", f"{confirm_seconds:.1f}")
    with cols[1]:
        st.metric("Pre/Post", f"{int(pre_sec)}s / {int(post_sec)}s")
    with cols[2]:
        st.metric("Detect interval", f"{detection_interval:.2f}s")

    webrtc_streamer(
        key="camera_fall_detector",
        video_frame_callback=camera_callback,
        media_stream_constraints={"video": True, "audio": False}
    )
