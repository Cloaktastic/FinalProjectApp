
#Import All the Required Libraries
import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import subprocess
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import requests
import time
import av
from collections import deque

#Sources
IMAGE = 'Image'
VIDEO = 'Video'
CAMERA = 'Camera'
SOURCES_LIST = [IMAGE, VIDEO, CAMERA]

VIDEO_DIR = 'App/videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR + '/' + '01.mp4',
    'video 2': VIDEO_DIR + '/' + '02.mp4',
    'video 3': VIDEO_DIR + '/' + '03.mp4',
}

#Page Layout
st.set_page_config(
    page_title = "Elderly Fall Detection - Computer Vision Approach",
    page_icon = "🚨"
)

#Header
st.header("YOLOv8n for Fall & Person Detection 🚑👤")

#SideBar
st.sidebar.header("Model Configurations")

confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40))/100

st.sidebar.header("Mode")

source_radio = st.sidebar.radio(
    "Select Source", SOURCES_LIST
)

#Load the YOLO Model
model_path = 'App/models/100_epochs.pt'  # This model can detect both fall and person
try:
    model = YOLO(model_path)
    # Get class names to understand what the model detects
    class_names = model.names
except Exception as e:
    st.error(f"Unable to load model. Check the sepcified path: {model_path}")
    st.error(e)

#Telegram functions
def tg_send_message(token, chat_id, text):
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=15
        )
        if response.status_code != 200:
            return False
        else:
            return True
    except Exception as e:
        print("Telegram message failed:", e)
        return False

def video_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # YOLO predict
    results = model(img, verbose=False)[0]
    boxes = results.boxes

    # Vẽ bounding boxes cho cả fall và person
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Lấy class ID và tên class
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = box.conf[0].cpu().numpy()
            
            # Xác định màu và label dựa trên class
            if class_name.lower() == 'fall':
                # Fall class - đỏ
                box_color = (0, 0, 255)
                label_text = f"FALL: {conf:.2f}"
                print(f"Fall detected! Confidence: {conf:.3f}")
            elif class_name.lower() in ['person', 'people', 'human']:
                # Person class - xanh lá
                box_color = (0, 255, 0)
                label_text = f"PERSON: {conf:.2f}"
                print(f"Person detected! Confidence: {conf:.3f}")
            else:
                # Class khác - vàng
                box_color = (0, 255, 255)
                label_text = f"{class_name.upper()}: {conf:.2f}"
            
            # Vẽ rectangle
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            
            # Vẽ label với confidence
            cv2.putText(img, label_text,
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Trả frame cho WebRTC hiển thị
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def tg_send_video(token, chat_id, path, caption=""):
    try:
        with open(path, "rb") as f:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendVideo",
                data={"chat_id": chat_id, "caption": caption},
                files={"video": (os.path.basename(path), f, "video/mp4")},
                timeout=180
            )
        if response.status_code != 200:
            print("Video error response:", response.text)
        else:
            print("Video sent successfully")
    except Exception as e:
        print("Telegram video failed:", e)

def tg_send_audio(token, chat_id, path, caption=""):
    try:
        with open(path, "rb") as f:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendAudio",
                data={"chat_id": chat_id, "caption": caption},
                files={"audio": (os.path.basename(path), f, "audio/ogg")},
                timeout=180
            )
        if response.status_code != 200:
            print("Audio error response:", response.text)
        else:
            print("Audio sent successfully")
    except Exception as e:
        print("Telegram audio failed:", e)



class VideoTransformer(VideoTransformerBase):
    def __init__(self, tg_token, tg_chat):
        self.tg_token = tg_token
        self.tg_chat = tg_chat
        self.buf = deque(maxlen=40)  # Buffer for storing frames
        self.rec_fps = 3.0
        self.pre_sec = 15.0
        self.post_sec = 15.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr")
        t_now = time.time()
        self.buf.append((t_now, img.copy()))

        # Detect falls (using YOLO model)
        results = model(img, verbose=False)[0]
        boxes = results.boxes
        
        # Check for both fall and person detections
        fall_detected = False
        if boxes:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name.lower() == 'fall':
                    fall_detected = True
                    break
        
        if fall_detected:  # If a fall is detected
            print("Fall detected!")
            self.ensure_and_save_30s_clip(t_now)

        return img

    def ensure_and_save_30s_clip(self, t0):
        # Wait until enough frames are collected
        deadline = t0 + self.post_sec
        while True:
            latest_t = self.buf[-1][0] if self.buf else 0.0
            if latest_t >= deadline:
                break
            time.sleep(0.2)

        frames = [f for (t, f) in list(self.buf) if (t >= t0 - self.pre_sec and t <= t0 + self.post_sec)]
        if not frames:
            return None

        # Save video
        tmp_path = f"videos/face_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(tmp_path, fourcc, self.rec_fps, (frames[0][1].shape[1], frames[0][1].shape[0]))
        for f in frames:
            vw.write(f[1])
        vw.release()

        # Send video to Telegram
        tg_send_video(self.tg_token, self.tg_chat, tmp_path, caption="Fall detected!")

source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        "Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp")
    )
    col1, col2 = st.columns(2)
    with col1:
        # try:
        #     uploaded_image  =Image.open(source_image)
        #     st.image(source_image, caption = "Uploaded Image", use_container_width = True)
        # except Exception as e:
        #     st.error("Error Occured While Opening the Image")
        #     st.error(e)
        if source_image: 
          uploaded_image  =Image.open(source_image)
          st.image(source_image, caption = "Uploaded Image", use_container_width = True)
        else:
          st.text("Please Upload an Image")
    with col2:
        try:
            if st.sidebar.button("Detect Objects"):
                result = model.predict(uploaded_image, conf = confidence_value)
                boxes = result[0].boxes
                
                # Vẽ bounding boxes thủ công để có màu sắc phân biệt
                img_array = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
                
                if boxes is not None and len(boxes) > 0:
                    fall_count = 0
                    person_count = 0
                    
                    for box in boxes:
                        # Lấy tọa độ bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Lấy class ID và tên class
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        conf = box.conf[0].cpu().numpy()
                        
                        # Xác định màu và label dựa trên class
                        if class_name.lower() == 'fall':
                            # Fall class - đỏ
                            box_color = (0, 0, 255)
                            label_text = f"FALL: {conf:.2f}"
                            fall_count += 1
                        elif class_name.lower() in ['person', 'people', 'human']:
                            # Person class - xanh lá
                            box_color = (0, 255, 0)
                            label_text = f"PERSON: {conf:.2f}"
                            person_count += 1
                        else:
                            # Class khác - vàng
                            box_color = (0, 255, 255)
                            label_text = f"{class_name.upper()}: {conf:.2f}"
                        
                        # Vẽ rectangle
                        cv2.rectangle(img_array, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Vẽ label với confidence
                        cv2.putText(img_array, label_text,
                                   (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    
                    # Thêm text tổng kết
                    summary_text = f"Detection Summary: {person_count} person(s), {fall_count} fall(s)"
                    cv2.putText(img_array, summary_text,
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(img_array, "No objects detected",
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Convert back to RGB for display
                result_plotted = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                st.image(result_plotted, caption = "Detected Image", use_container_width = True)

                try:
                    with st.expander("Detection Results"):
                        if boxes is not None and len(boxes) > 0:
                            st.write("**Detections found:**")
                            for i, box in enumerate(boxes):
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                conf = box.conf[0].cpu().numpy()
                                coords = box.xyxy[0].cpu().numpy()
                                st.write(f"{i+1}. {class_name} - Confidence: {conf:.3f} - Coordinates: {coords}")
                        else:
                            st.write("No objects detected in the image.")
                except Exception as e:
                    st.error(e)
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)

elif source_radio == VIDEO:
    source_video = st.sidebar.selectbox(
        "Choose a Video...", VIDEOS_DICT.keys()
    )
    with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        if st.sidebar.button("Detect Video Objects"):
            try:
                # Predict every frame of the video
                results = model(VIDEOS_DICT.get(source_video), save=True, show=True, conf = confidence_value)

                # Get the latest avi file
                DETECT_FOLDER = 'runs/detect'
                number_of_predictions = len(os.listdir(DETECT_FOLDER))
                if number_of_predictions == 1:
                  latest_predict = 'predict'
                else:
                  latest_predict = 'predict' + str(number_of_predictions)
                avi_file =  DETECT_FOLDER + '/' + latest_predict + '/' + os.listdir(DETECT_FOLDER + '/' + latest_predict)[0]

                # Convert from avi to mp4, as streamlit cannot view the video from the avi file
                output_mp4 = f"{latest_predict}.mp4"
                subprocess.run([
                                "ffmpeg", "-i", avi_file,
                                "-ac", "2", "-b:v", "2000k",
                                "-c:a", "aac", "-c:v", "libx264",
                                "-b:a", "160k", "-vprofile", "high",
                                "-bf", "0", "-strict", "experimental",
                                "-f", "mp4", output_mp4
                            ], check=True)

                # Show the video
                video_file = open(output_mp4, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)

            except Exception as e:
                st.sidebar.error("Error Loading Video"+str(e))

# Camera section
elif source_radio == CAMERA:
    # Load Telegram token và chat ID
    with open('App/token.txt', 'r') as file:
        lines = file.readlines()
        tg_token = lines[0].strip()
        tg_chat = lines[1].strip()

    st.header("Camera Stream")
    st.write(
        "📹 Hệ thống sẽ detect **té ngã** và **người**.\n\n"
        "- Vẽ bounding box cho cả **person** (xanh lá) và **fall** (đỏ).\n"
        "- Phát hiện té ngã khi **người nằm bất động 5 giây liên tiếp**.\n"
        "- Ghi lại **30s video** (15s trước + 15s sau thời điểm phát hiện).\n"
        "- Gửi video lên Telegram ở định dạng xem được trên Telegram Web."
    )

    # Khởi tạo state lần đầu
    if "fall_state" not in st.session_state:
        st.session_state.fall_state = {
            "buf": deque(maxlen=900),   # đủ ~30s nếu fps ~30 (30*30 = 900)
            "detected": False,          # đã detect té ngã chưa
            "saved": False,             # đã lưu/gửi video chưa
            "t0": None,                 # thời điểm phát hiện fall
            "pre_sec": 15.0,
            "post_sec": 15.0,
            "last_t": None,             # dùng để ước lượng fps
            "fps_sum": 0.0,
            "fps_n": 0,
            # Thêm các biến để tối ưu bounding box
            "frame_count": 0,           # đếm số frame đã xử lý
            "last_detection": None,     # lưu kết quả detect cuối
            "last_detection_time": 0,   # thời điểm detect cuối
            "detection_interval": 0.05,  # chỉ detect mỗi 0.05 giây (20 lần/giây)
            "last_boxes": None,         # lưu boxes cuối để vẽ
            # Thêm các biến để tracking motionless detection
            "motionless_start": None,   # thời điểm bắt đầu detect motionlessness
            "motionless_duration": 5.0, # cần motionless trong 5 giây
            "is_motionless": False,     # đã detect motionlessness chưa
            "continuous_fall_count": 0, # đếm số lần detect liên tiếp
        }

    state = st.session_state.fall_state

    def video_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        t_now = time.time()

        # --- ƯỚC LƯỢNG FPS THỰC TẾ ---
        if state["last_t"] is not None:
            dt = t_now - state["last_t"]
            if 0.005 < dt < 1.0:  # loại bỏ spike bất thường
                fps_inst = 1.0 / dt
                state["fps_sum"] += fps_inst
                state["fps_n"] += 1
        state["last_t"] = t_now

        # Lưu frame vào buffer (luôn luôn lưu)
        state["buf"].append((t_now, img.copy()))

        # --- TỐI ƯU: CHỈ DETECT MỖI 0.05 GIÂY ---
        should_detect = False
        if not state["detected"]:
            # Chỉ detect nếu đủ thời gian interval hoặc chưa có kết quả nào
            if (t_now - state["last_detection_time"]) >= state["detection_interval"]:
                should_detect = True
        
        # Lưu kết quả detection để vẽ bounding box
        current_boxes = state["last_boxes"]
        
        if should_detect:
            # dùng confidence_value từ sidebar
            results = model(img, verbose=False, conf=confidence_value)[0]
            boxes = results.boxes

            # Lưu kết quả detection
            state["last_detection_time"] = t_now
            state["last_boxes"] = boxes
            
            # Phân loại và xử lý detection results
            fall_detected = False
            person_detected = False
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Lấy class ID và tên class
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    if class_name.lower() == 'fall':
                        fall_detected = True
                        print(f"Fall detected! Confidence: {box.conf[0]:.3f}")
                    elif class_name.lower() in ['person', 'people', 'human']:
                        person_detected = True
                        print(f"Person detected! Confidence: {box.conf[0]:.3f}")
            
            # Logic xử lý fall detection (giữ nguyên logic cũ)
            if fall_detected:
                # Bắt đầu tracking thời gian motionless nếu chưa bắt đầu
                if state["motionless_start"] is None:
                    state["motionless_start"] = t_now
                    state["continuous_fall_count"] = 1
                else:
                    # Tăng số lần detect liên tiếp
                    state["continuous_fall_count"] += 1
                
                # Kiểm tra xem đã đủ 5 giây motionless chưa
                motionless_duration = t_now - state["motionless_start"]
                if motionless_duration >= state["motionless_duration"]:
                    print(f"CONFIRMED FALL! Motionless for {motionless_duration:.2f} seconds")
                    state["detected"] = True
                    state["t0"] = t_now
                    
                    # Send message to group
                    group_message = "⚠️ FALL CONFIRMED! Person motionless for 5 seconds. Recording 30-second clip (15s before + 15s after)..."
                    tg_send_message(tg_token, tg_chat, group_message)
                    

                    
                    # Reset tracking variables
                    state["motionless_start"] = None
                    state["continuous_fall_count"] = 0
            else:
                # Nếu không detect fall, reset tracking
                if state["motionless_start"] is not None:
                    print("Movement detected - resetting motionless timer")
                    state["motionless_start"] = None
                    state["continuous_fall_count"] = 0
            
            # Detection summary logged
            
            current_boxes = boxes

        # --- VẼ BOUNDING BOX TỪ KẾT QUẢ ĐÃ LƯU ---
        if current_boxes is not None and len(current_boxes) > 0:
            # Vẽ bounding boxes lên frame hiện tại
            for box in current_boxes:
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Lấy class ID và tên class
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf = box.conf[0].cpu().numpy()
                
                # Xác định màu và label dựa trên class và trạng thái
                if class_name.lower() == 'fall':
                    # Fall class - đỏ khi chưa confirm, đỏ đậm khi đã confirm
                    if state["detected"]:
                        box_color = (0, 0, 255)  # Đỏ đậm cho confirmed fall
                        label_text = f"FALL CONFIRMED: {conf:.2f}"
                    else:
                        box_color = (0, 0, 255)  # Đỏ nhạt cho fall detected
                        label_text = f"FALL DETECTED: {conf:.2f}"
                elif class_name.lower() in ['person', 'people', 'human']:
                    # Person class - xanh lá hoặc xanh dương
                    box_color = (0, 255, 0)  # Xanh lá cho person
                    label_text = f"PERSON: {conf:.2f}"
                else:
                    # Class khác - vàng
                    box_color = (0, 255, 255)  # Vàng/xanh dương
                    label_text = f"{class_name.upper()}: {conf:.2f}"
                
                # Vẽ rectangle với màu tương ứng
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                
                # Vẽ label với confidence
                cv2.putText(img, label_text,
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # Hiển thị thời gian motionless cho fall detection
                if class_name.lower() == 'fall' and state["motionless_start"] is not None and not state["detected"]:
                    motionless_time = t_now - state["motionless_start"]
                    remaining_time = max(0, state["motionless_duration"] - motionless_time)
                    timer_text = f"Motionless: {motionless_time:.1f}s (Wait {remaining_time:.1f}s)"
                    
                    # Vẽ timer lên frame
                    cv2.putText(img, timer_text,
                               (int(x1), int(y2) + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Vẽ vòng tròn progress
                    progress = min(1.0, motionless_time / state["motionless_duration"])
                    center_x, center_y = int(x2) + 30, int(y1) + 30
                    cv2.circle(img, (center_x, center_y), 15, (50, 50, 50), 2)
                    cv2.circle(img, (center_x, center_y), 15, (0, 255, 255), int(15 * progress), -1)
                
                # Nếu đã confirmed fall, hiển thị thông báo đỏ lớn trên frame
                if state["detected"]:
                    cv2.putText(img, "⚠️ FALL CONFIRMED ⚠️",
                               (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Hiển thị tổng quan detection ở góc trên bên phải
            detection_counts = {"fall": 0, "person": 0, "other": 0}
            for box in current_boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name.lower() == 'fall':
                    detection_counts["fall"] += 1
                elif class_name.lower() in ['person', 'people', 'human']:
                    detection_counts["person"] += 1
                else:
                    detection_counts["other"] += 1
            
            # Vẽ summary box
            summary_text = f"Detections: {detection_counts['person']} person(s), {detection_counts['fall']} fall(s)"
            if detection_counts["other"] > 0:
                summary_text += f", {detection_counts['other']} other(s)"
            
            # Background cho summary
            text_size = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (img.shape[1] - text_size[0] - 20, 10), 
                         (img.shape[1] - 10, 40), (0, 0, 0), -1)
            cv2.putText(img, summary_text,
                       (img.shape[1] - text_size[0] - 10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- SAU KHI DETECT: CHỜ ĐỦ 15s SAU RỒI LƯU CLIP ---
        if state["detected"] and not state["saved"]:
            t0 = state["t0"]
            pre_sec = state["pre_sec"]
            post_sec = state["post_sec"]

            if t_now >= t0 + post_sec:
                frames_all = list(state["buf"])

                # Giữ cả (t, f) để tính FPS chính xác cho đoạn clip
                selected = [
                    (t, f) for (t, f) in frames_all
                    if (t >= t0 - pre_sec and t <= t0 + post_sec)
                ]

                if selected:
                    # Thời gian thực của đoạn clip
                    t_first = selected[0][0]
                    t_last = selected[-1][0]
                    duration = max(0.001, t_last - t_first)  # tránh chia 0
                    n_frames = len(selected)

                    # FPS = số frame / thời gian
                    fps_clip = (n_frames - 1) / duration if n_frames > 1 else 10.0

                    # Giới hạn FPS cho mượt, tránh quá nhanh/chậm
                    fps_clip = max(5.0, min(20.0, fps_clip))

                    print(f"Clip duration ~ {duration:.2f}s, frames = {n_frames}, fps_clip = {fps_clip:.2f}")

                    h, w = selected[0][1].shape[:2]
                    os.makedirs("videos", exist_ok=True)

                    raw_path = f"videos/fall_{int(time.time())}_raw.mp4"

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(raw_path, fourcc, fps_clip, (w, h))
                    for _, frame_img in selected:
                        vw.write(frame_img)
                    vw.release()

                    # --- Convert sang H.264 chuẩn Telegram Web ---
                    fixed_path = raw_path.replace("_raw.mp4", ".mp4")
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", raw_path,
                        "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-profile:v", "baseline",
                        "-level", "3.1",
                        "-movflags", "+faststart",
                        "-an",  # không audio
                        fixed_path
                    ]
                    subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

                    print(f"Saved fall clip -> {fixed_path}")
                    
                    # Send video to group
                    tg_send_video(
                        tg_token,
                        tg_chat,
                        fixed_path,
                        caption="📹 Fall detected! 30-second clip (15s before + 15s after)."
                    )



                    state["saved"] = True
                    
                    # Send completion messages
                    tg_send_message(tg_token, tg_chat, "✅ Clip sent to group. You can stop the camera stream in the app now.")

        # Sau khi đã gửi video, không detect nữa, chỉ trả frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="camera_fall_detector",
        video_frame_callback=video_callback,
        media_stream_constraints={"video": True, "audio": False}
    )