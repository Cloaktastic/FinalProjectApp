# README - Module Person 1 (OpenPose) - Pose & Feature Extraction

1. Mục tiêu của module này là gì?

---

Module này dùng OpenPose (BODY_25) để lấy bộ keypoints (x, y, score) theo từng frame trong một đoạn video “event”, sau đó trích xuất ra một số đặc trưng (features) đơn giản liên quan đến té ngã, gồm:

* v_peak: đỉnh tốc độ thay đổi theo phương thẳng đứng của hông (hip)
* delta_angle: mức thay đổi góc thân người (vector hip -> neck) trước và sau thời điểm “rơi mạnh”
* immobile_duration: thời gian bất động sau thời điểm “rơi mạnh” (ước lượng theo dy và da nhỏ)

Đây không phải là “model phát hiện té ngã” mới, mà là một cách bổ sung tín hiệu (pose-based features) để hỗ trợ hệ thống ra quyết định tốt hơn, giảm báo động nhầm.

2.. Tận dụng module này trong hệ thống như thế nào cho tối ưu?

---

Vì OpenPose chạy chậm, đặc biệt là bản portable chạy CPU, nhóm không chạy OpenPose liên tục.

Cách tối ưu đề xuất:

* Chỉ chạy OpenPose khi hệ thống YOLO phát hiện “event nghi ngờ té ngã”
* Tức là: YOLO chạy realtime; nếu có tín hiệu fall (hoặc điều kiện kích hoạt), ta cắt ra một đoạn video event ngắn (ví dụ 8–12 giây) rồi mới đưa cho OpenPose xử lý
* Như vậy giảm tải rất nhiều, vì thay vì chạy OpenPose 24/7, ta chỉ chạy vài lần khi có event

Ngoài ra có thể tối ưu thêm:

* Giới hạn number_people_max = 1 (module đã set) để nhẹ hơn
* Tắt render/display (module đã set) để nhẹ hơn
* Có thể tăng frame_step ở OpenPose (nếu cần) để giảm số frame xử lý

4. Pipeline hệ thống sau cùng

---

Pipeline tổng (gợi ý triển khai):

(1) Camera stream / Video input
|
(2) YOLOv8 inference realtime
- detect person/fall
- theo dõi điều kiện kích hoạt event (ví dụ fall liên tục trong 3s hoặc fall confidence vượt ngưỡng)
|
(3) Khi kích hoạt event:
- cắt clip event (ví dụ: 2 giây trước + 8 giây sau, có thể rút ngắn để tối ưu)
- lưu clip thành file tmp_event_xxx.mp4
|
(4) Gọi Module Person 1 (OpenPose):
- chạy OpenPose trên tmp_event_xxx.mp4
- xuất keypoints JSON theo frame
- trích xuất features: v_peak, delta_angle, immobile_duration
|
(5) Tầng quyết định cuối (Decision):
- dùng rule đơn giản hoặc classifier (tùy nhóm)
- kết hợp: output YOLO + features OpenPose
- xác định “fall thật” hay “fall giả”
|
(6) Nếu fall thật:
- gửi cảnh báo Telegram + gửi video event
- lưu log và dữ liệu phục vụ đánh giá

Lý do pipeline này hợp lý:

* YOLO đảm bảo realtime
* OpenPose chỉ chạy khi cần (event) nên tổng hệ thống vẫn đáp ứng được tốc độ
* giảm false positive vì có thêm tín hiệu pose và chuyển động theo thời gian

5. Module code hiện tại làm gì?

---

Module chính gồm các hàm sau:

5.1) run_openpose_on_video(video_path, output_dir, openpose_bin)

* Mục tiêu: chạy OpenPose trên video input và xuất ra JSON keypoints theo từng frame.
* output_dir sẽ chứa nhiều file JSON, mỗi file là 1 frame.
* Code có kiểm tra nếu đã có JSON sẵn thì không chạy lại (đỡ tốn thời gian).

Tham số OpenPose đang dùng:

* --video <video_path>           : video đầu vào
* --write_json <output_dir>      : nơi xuất JSON
* --display 0                    : không mở cửa sổ hiển thị
* --model_pose BODY_25           : dùng bộ keypoints BODY_25
* --number_people_max 1          : chỉ lấy 1 người (phù hợp bài toán)
* --render_pose 0                : tắt render keypoints (nhẹ hơn)

Với bản Portable chạy CPU, khuyến nghị thêm:

* --num_gpu 0                    : ép chạy CPU, tránh lỗi cố dùng GPU

5.2) load_keypoints_from_openpose(json_paths)

* Mục tiêu: đọc toàn bộ JSON đã xuất và gom lại thành 1 mảng NumPy dạng (T, 25, 3)

  * T: số frame
  * 25: số khớp BODY_25
  * 3: (x, y, score)

Nếu frame nào không detect được người:

* code sẽ điền NaN cho frame đó (để không bị crash, và các bước sau sẽ dùng nan-safe)

5.3) butter_lowpass_filter(data, cutoff, fs, order=4)

* Mục tiêu: lọc thông thấp Butterworth để làm mượt tín hiệu.
* Dùng để giảm nhiễu do keypoints rung theo frame.

5.4) extract_features_from_keypoints(keypoints, fps=30, use_height_normalization=True)

* Mục tiêu: từ keypoints theo thời gian -> tính 3 feature chính.

Chi tiết các feature:

(1) v_peak:

* Lấy hip.y theo thời gian -> chuẩn hóa (nếu bật) -> lọc mượt -> lấy đạo hàm để ra “tốc độ”
* Lấy giá trị lớn nhất của tốc độ (nanmax) làm v_peak

(2) delta_angle:

* Tính vector thân người = neck - hip
* Tính góc so với trục thẳng đứng (y tăng xuống dưới)
* Lọc mượt góc
* Tìm frame có v_peak lớn nhất (idx_peak)
* Lấy trung bình góc 0.5s trước idx_peak và 0.5s sau idx_peak
* delta_angle = |angle_after - angle_before|

(3) immobile_duration:

* Duyệt từ idx_peak đến cuối clip
* Nếu thay đổi vị trí hông (dy) < ngưỡng và thay đổi góc (da) < ngưỡng
  thì xem như “bất động” và cộng frame
* immobile_duration = immobile_frames * (1/fps)

Lưu ý quan trọng:

* immobile_duration phụ thuộc độ dài clip sau “rơi”. Nếu clip quá ngắn, giá trị này sẽ bị giới hạn theo thời lượng clip.
* Nếu nhóm muốn dùng tiêu chí “nằm im >= 5 giây” thì clip event phải đủ dài sau thời điểm rơi.

5.5) extract_features(video_path, openpose_bin, output_dir, fps=None)

* Hàm cấp cao gọi lần lượt:
  (1) run_openpose_on_video
  (2) load_keypoints_from_openpose
  (3) đọc fps từ video nếu không truyền vào
  (4) extract_features_from_keypoints
* Trả về dict features.

6. Cách chạy trên Windows (Portable + CPU)

---

Chuẩn bị:

* tải OpenPose bản portable (OpenPoseDemo.exe)
* giải nén, ví dụ C:\openpose\
* file chạy: C:\openpose\bin\OpenPoseDemo.exe

Thiết lập trong code:
* copy folder openpose vào ổ C
* openpose_bin = r"C:\openpose\bin\OpenPoseDemo.exe"
* output_dir = nơi lưu json (nên tách theo event hoặc theo video)


7. Dữ liệu vào/ra của module (input/output)

---

Input:

* video event (ví dụ tmp_event_001.mp4)
  Đây là clip đã được cắt từ luồng camera (thường do YOLO kích hoạt)

Output trung gian:

* folder output_dir chứa các file JSON keypoints theo từng frame
  Mục đích: debug, kiểm tra lại keypoints khi cần

Output cuối:

* dict features dạng:
  {
  "v_peak": ...,
  "delta_angle": ...,
  "immobile_duration": ...
  }

8. Ghi chú triển khai thực tế trong dự án nhóm

---

* Không chạy OpenPose liên tục realtime nếu chỉ có CPU.
* Chạy OpenPose theo event là hợp lý nhất.
* Nên thống nhất độ dài event clip (ví dụ 10 giây) để các feature có ý nghĩa ổn định giữa các event.
* Nếu muốn chạy nhanh hơn, có thể tăng frame_step (đổi lại độ mịn thời gian giảm).
* Nếu có nhiều người trong khung hình, code đang giới hạn 1 người. Đây phù hợp nếu camera chủ yếu theo dõi 1 người cao tuổi. Nếu môi trường đông người, cần phương án chọn đúng người (theo tracking hoặc theo bbox YOLO).
