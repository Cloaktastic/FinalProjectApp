# =========================================
# Elderly Fall Detection - Streamlit App
# Unified: Image / Video / Camera
# - Video mode: NO Telegram (local save only)
# - Camera mode: Telegram enabled (message + clip)
# - Confirm fall after 3 seconds (default, configurable)
# - Unified bounding-box drawing across modes
# - Device setting: Auto / CPU / GPU (CUDA)
# =========================================

import os
import cv2
import av
import time
import numpy as np
import streamlit as st
import subprocess
import requests
from PIL import Image
from ultralytics import YOLO
from collections import deque
from streamlit_webrtc import webrtc_streamer

# ----------------------------
# Sources
# ----------------------------
IMAGE = "Image"
VIDEO = "Video"
CAMERA = "Camera"
SOURCES_LIST = [IMAGE, VIDEO, CAMERA]

VIDEO_DIR = "videos"
VIDEOS_DICT = {
    "Video 1": os.path.join(VIDEO_DIR, "01.mp4"),
    "Video 2": os.path.join(VIDEO_DIR, "02.mp4"),
    "Video 3": os.path.join(VIDEO_DIR, "03.mp4"),
}

MODEL_PATH = "models/100_epochs.pt"  # detect both fall & person
TOKEN_FILE = "token.txt"
OUT_DIR = "videos"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Page Layout (more professional, less text)
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
pre_sec = float(st.sidebar.slider("Clip pre (sec)", 5, 30, 15, 1))
post_sec = float(st.sidebar.slider("Clip post (sec)", 5, 30, 15, 1))
detection_interval = float(st.sidebar.slider("Detection interval (sec)", 0.02, 0.30, 0.05, 0.01))
cooldown_sec = float(st.sidebar.slider("Cooldown after confirm (sec)", 2, 30, 10, 1))

st.sidebar.header("Input")
source_radio = st.sidebar.radio("Select source", SOURCES_LIST)

# ----------------------------
# Device resolution
# ----------------------------
def resolve_device(opt: str) -> str:
    if opt == "CPU":
        return "cpu"
    if opt == "GPU (CUDA)":
        return "cuda"
    # Auto
    return "cuda"

# ----------------------------
# Load YOLO Model (cached)
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
    class_names = model.names

except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {MODEL_PATH}")
    st.exception(e)
    st.stop()

# ----------------------------
# Telegram helpers (camera only)
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
    except Exception as e:
        print("Telegram message failed:", e)
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
        if r.status_code != 200:
            print("Telegram video error:", r.text)
    except Exception as e:
        print("Telegram video failed:", e)

# ----------------------------
# Utilities: class helpers
# ----------------------------
def is_fall(name: str) -> bool:
    return name.lower() == "fall"

def is_person(name: str) -> bool:
    return name.lower() in ["person", "people", "human"]

# ----------------------------
# Utilities: draw boxes (unified)
# ----------------------------
def draw_boxes(
    img_bgr: np.ndarray,
    boxes,
    detected_confirmed: bool,
    motionless_start_t: float | None,
    t_now: float,
    confirm_seconds_: float
) -> np.ndarray:
    if boxes is None or len(boxes) == 0:
        return img_bgr

    h, w = img_bgr.shape[:2]
    fall_count, person_count, other_count = 0, 0, 0

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cls_id = int(b.cls[0])
        name = model.names[cls_id]
        conf = float(b.conf[0].cpu().numpy())

        if is_fall(name):
            fall_count += 1
            color = (0, 0, 255)
            label = f"FALL {'CONFIRMED' if detected_confirmed else 'DETECTED'}: {conf:.2f}"
        elif is_person(name):
            person_count += 1
            color = (0, 255, 0)
            label = f"PERSON: {conf:.2f}"
        else:
            other_count += 1
            color = (0, 255, 255)
            label = f"{name.upper()}: {conf:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, label, (x1, max(12, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # show "hold to confirm" timer for fall while pending
        if is_fall(name) and (not detected_confirmed) and (motionless_start_t is not None):
            held = max(0.0, t_now - motionless_start_t)
            remain = max(0.0, confirm_seconds_ - held)
            timer = f"Hold: {held:.1f}s | Remain: {remain:.1f}s"
            cv2.putText(img_bgr, timer, (x1, min(h - 10, y2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # small summary top-right
    summary = f"{person_count} person | {fall_count} fall"
    if other_count:
        summary += f" | {other_count} other"

    (tw, th), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img_bgr, (w - tw - 20, 10), (w - 10, 10 + th + 14), (0, 0, 0), -1)
    cv2.putText(img_bgr, summary, (w - tw - 12, 10 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if detected_confirmed:
        cv2.putText(img_bgr, "⚠ FALL CONFIRMED ⚠", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return img_bgr

# ----------------------------
# Utilities: ffmpeg fix for Telegram/Web playback
# ----------------------------
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

# ----------------------------
# Unified fall-state logic
# ----------------------------
def init_state(buf_maxlen: int):
    return {
        "buf": deque(maxlen=buf_maxlen),  # (t, frame)
        "last_det_t": -1e9,
        "last_boxes": None,

        "motionless_start": None,
        "detected": False,
        "saved": False,
        "t0": None,
        "cooldown_until": -1e9,

        "sent_confirm_msg": False,
    }

def update_confirm_logic(state, t_now: float, boxes, confirm_sec: float, cooldown: float):
    # if in cooldown, skip arming new confirmations
    if t_now < state["cooldown_until"]:
        return

    fall_present = False
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            name = model.names[int(b.cls[0])]
            if is_fall(name):
                fall_present = True
                break

    if not state["detected"]:
        if fall_present:
            if state["motionless_start"] is None:
                state["motionless_start"] = t_now
            else:
                if (t_now - state["motionless_start"]) >= confirm_sec:
                    state["detected"] = True
                    state["t0"] = t_now
                    state["cooldown_until"] = t_now + cooldown
                    state["motionless_start"] = None
        else:
            state["motionless_start"] = None

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
            boxes = res.boxes

            annotated = draw_boxes(
                img_bgr.copy(),
                boxes,
                detected_confirmed=False,
                motionless_start_t=None,
                t_now=0.0,
                confirm_seconds_=confirm_seconds
            )

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            with st.expander("Detections"):
                if boxes is None or len(boxes) == 0:
                    st.write("No detections.")
                else:
                    for i, b in enumerate(boxes):
                        name = model.names[int(b.cls[0])]
                        conf = float(b.conf[0].cpu().numpy())
                        xyxy = b.xyxy[0].cpu().numpy().tolist()
                        st.write(f"{i+1}. **{name}** | conf={conf:.3f} | box={xyxy}")

# ============================================================
# VIDEO MODE (NO TELEGRAM)
# ============================================================
elif source_radio == VIDEO:
    st.subheader("Video")

    source_video_key = st.sidebar.selectbox("Choose video", list(VIDEOS_DICT.keys()))
    video_path = VIDEOS_DICT.get(source_video_key)

    if not os.path.exists(video_path):
        st.error(f"Video not found: {video_path}")
        st.stop()

    with open(video_path, "rb") as f:
        st.video(f.read())

    st.caption("Video mode runs the same confirm logic + drawing, but **does not** send Telegram alerts.")

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

        # buffer size based on video fps
        buf_maxlen = int((pre_sec + post_sec + 2) * fps)
        state = init_state(buf_maxlen)

        out_raw = os.path.join(OUT_DIR, f"annot_video_{int(time.time())}_raw.mp4")
        vw = cv2.VideoWriter(out_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        prog = st.progress(0.0)
        status = st.empty()

        clip_fixed_path = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_now = frame_idx / fps  # stable timestamp for file
            frame_idx += 1

            state["buf"].append((t_now, frame.copy()))

            # detect at interval
            if (t_now - state["last_det_t"]) >= detection_interval and not state["saved"]:
                res = model(frame, verbose=False, conf=confidence_value)[0]
                boxes = res.boxes
                state["last_boxes"] = boxes
                state["last_det_t"] = t_now

                update_confirm_logic(state, t_now, boxes, confirm_seconds, cooldown_sec)

            annotated = draw_boxes(
                frame.copy(),
                state["last_boxes"],
                detected_confirmed=state["detected"],
                motionless_start_t=state["motionless_start"],
                t_now=t_now,
                confirm_seconds_=confirm_seconds
            )

            vw.write(annotated)

            # Save 30s clip locally when confirmed and post_sec elapsed
            if state["detected"] and (not state["saved"]) and (t_now >= (state["t0"] + post_sec)):
                selected = [(t, f) for (t, f) in list(state["buf"]) if (state["t0"] - pre_sec) <= t <= (state["t0"] + post_sec)]
                if selected:
                    clip_raw = os.path.join(OUT_DIR, f"fallclip_video_{int(time.time())}_raw.mp4")
                    vw2 = cv2.VideoWriter(clip_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    for _, fr in selected:
                        vw2.write(fr)
                    vw2.release()

                    clip_fixed_path = clip_raw.replace("_raw.mp4", ".mp4")
                    ffmpeg_fix_h264_yuv420p(clip_raw, clip_fixed_path)

                state["saved"] = True

            if total_frames > 0:
                prog.progress(min(1.0, frame_idx / total_frames))
            status.write(f"{frame_idx}/{total_frames} frames | FPS={fps:.1f} | Confirmed={state['detected']} | ClipSaved={state['saved']}")

        cap.release()
        vw.release()

        out_fixed = out_raw.replace("_raw.mp4", ".mp4")
        ffmpeg_fix_h264_yuv420p(out_raw, out_fixed)

        st.success("Done.")
        st.markdown("**Annotated video**")
        st.video(out_fixed)

        if clip_fixed_path and os.path.exists(clip_fixed_path):
            st.markdown("**30s clip (local only)**")
            st.video(clip_fixed_path)
        else:
            st.info("No confirmed fall (or not held long enough).")

# ============================================================
# CAMERA MODE (TELEGRAM ONLY HERE)
# ============================================================
elif source_radio == CAMERA:
    st.subheader("Camera")

    tg_token, tg_chat = read_telegram_credentials()
    if (tg_token is None) or (tg_chat is None):
        st.warning("Telegram credentials not found in App/token.txt (2 lines: token, chat_id). Camera will still run, but alerts will be skipped.")

    # Session state
    if "fall_state" not in st.session_state:
        # assume up to ~30 fps for buffer sizing
        approx_fps = 30
        buf_maxlen = int((pre_sec + post_sec + 2) * approx_fps)
        st.session_state.fall_state = init_state(buf_maxlen)
        st.session_state.fall_state.update({
            "last_t_wall": None,
            "fps_sum": 0.0,
            "fps_n": 0,
        })

    state = st.session_state.fall_state

    def camera_callback(frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        t_wall = time.time()

        # estimate fps (optional)
        if state["last_t_wall"] is not None:
            dt = t_wall - state["last_t_wall"]
            if 0.005 < dt < 1.0:
                state["fps_sum"] += (1.0 / dt)
                state["fps_n"] += 1
        state["last_t_wall"] = t_wall

        state["buf"].append((t_wall, img.copy()))

        # detect at interval (skip after saved)
        if (t_wall - state["last_det_t"]) >= detection_interval and not state["saved"]:
            res = model(img, verbose=False, conf=confidence_value)[0]
            boxes = res.boxes
            state["last_boxes"] = boxes
            state["last_det_t"] = t_wall

            update_confirm_logic(state, t_wall, boxes, confirm_seconds, cooldown_sec)

            # send confirm message once (camera only)
            if state["detected"] and not state["sent_confirm_msg"]:
                state["sent_confirm_msg"] = True
                if tg_token and tg_chat:
                    tg_send_message(
                        tg_token, tg_chat,
                        f"⚠️ FALL CONFIRMED! Hold {confirm_seconds:.1f}s. Saving 30s clip ({int(pre_sec)}s before + {int(post_sec)}s after)..."
                    )

        annotated = draw_boxes(
            img.copy(),
            state["last_boxes"],
            detected_confirmed=state["detected"],
            motionless_start_t=state["motionless_start"],
            t_now=t_wall,
            confirm_seconds_=confirm_seconds
        )

        # Save clip after post_sec, then send to Telegram
        if state["detected"] and (not state["saved"]):
            t0 = state["t0"]
            if t_wall >= (t0 + post_sec):
                frames_all = list(state["buf"])
                selected = [(t, f) for (t, f) in frames_all if (t0 - pre_sec) <= t <= (t0 + post_sec)]
                if selected:
                    # compute clip fps from real timestamps
                    t_first = selected[0][0]
                    t_last = selected[-1][0]
                    duration = max(0.001, t_last - t_first)
                    n_frames = len(selected)
                    fps_clip = (n_frames - 1) / duration if n_frames > 1 else 10.0
                    fps_clip = max(5.0, min(20.0, fps_clip))

                    h, w = selected[0][1].shape[:2]
                    raw_path = os.path.join(OUT_DIR, f"fall_cam_{int(time.time())}_raw.mp4")
                    vw = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_clip, (w, h))
                    for _, fr in selected:
                        vw.write(fr)
                    vw.release()

                    fixed_path = raw_path.replace("_raw.mp4", ".mp4")
                    ffmpeg_fix_h264_yuv420p(raw_path, fixed_path)

                    # Telegram (camera only)
                    if tg_token and tg_chat:
                        tg_send_video(
                            tg_token, tg_chat, fixed_path,
                            caption=f"📹 Fall detected. 30s clip ({int(pre_sec)}s before + {int(post_sec)}s after)."
                        )
                        tg_send_message(tg_token, tg_chat, "✅ Clip sent.")

                state["saved"] = True

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    # Minimal UI (less text)
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
