# HỆ THỐNG GIÁM SÁT AN NINH THÔNG MINH PHÁT HIỆN XÂM NHẬP

## 1. Giới thiệu

Hệ thống này là một ứng dụng giám sát thời gian thực sử dụng Computer Vision để phát hiện người xâm nhập vào vùng cấm (ROI - Region of Interest).

Pipeline hoạt động:

```
Video → YOLO Detection → Tracking → ROI Check → Alert → Dashboard
```

Khi phát hiện người đi vào vùng ROI:

* Bounding box chuyển màu (xanh dương)
* Lưu ảnh snapshot
* Gửi email cảnh báo
* Hiển thị log trên dashboard

---

## 2. Công nghệ sử dụng

* Flask – Web server
* OpenCV – Xử lý ảnh/video
* Ultralytics YOLO – Nhận diện đối tượng
* SMTP (Gmail) – Gửi email cảnh báo
* HTML/CSS/JS – Dashboard

---

## 3. Tính năng chính

### Phát hiện xâm nhập

* Nhận diện người bằng YOLOv8
* Tracking ID theo thời gian
* Xác định xâm nhập dựa trên **tâm bounding box**

### Trực quan hóa

* Ngoài ROI → xanh lá
* Trong ROI → xanh dương
* ROI → viền đỏ + overlay mờ

### Cảnh báo

* Gửi email khi:

  * Đủ số frame xác nhận
  * Có chuyển động
  * Không vi phạm cooldown

### Dashboard

* Stream video realtime
* Hiển thị log sự kiện
* Highlight log mới
* Xem ảnh chi tiết (modal)

---

## 4. Cấu trúc project

```
project/
│
├── app.py              # Xử lý video + pipeline chính
├── mail_sender.py      # Module gửi email
├── .env                # Biến môi trường (bảo mật)
├── static/             # Ảnh snapshot
├── templates/
│   └── index.html      # Dashboard
```

---

## 5. Cài đặt

### 5.1 Cài thư viện

```
pip install flask opencv-python ultralytics python-dotenv
```

### 5.2 Mô hình YOLO

Model YOLOv8 nano (`yolov8n.pt`) được sử dụng do:

* Nhẹ, phù hợp xử lý realtime
* Độ chính xác đủ cho bài toán phát hiện người

Có thể tải tại:
https://docs.ultralytics.com/models/

---

### 5.3 Cấu hình `.env`

```env
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password
EMAIL_TO=receiver@gmail.com
MODEL_PATH=yolov8n.pt
```

⚠️ Lưu ý:

* Sử dụng **App Password** của Gmail
* Không hardcode thông tin nhạy cảm trong code

---

## 6. Chạy hệ thống

```
python app.py
```

Truy cập:

```
http://localhost:5000
```

---

## 7. Luồng hoạt động

### 7.1 Detection & Tracking

* YOLO phát hiện người (`class 0`)
* Tracking ID giúp tránh gửi cảnh báo lặp

### 7.2 ROI Logic

```python
center_x = (x1 + x2) // 2
in_roi = center_x > roi_x
```

→ Sử dụng tâm bounding box thay vì cạnh nhằm:

* Giảm sai lệch khi bounding box lớn
* Tránh false positive khi đối tượng chỉ chạm biên ROI

---

### 7.3 Điều kiện gửi cảnh báo

Một cảnh báo chỉ được kích hoạt khi:

* Đối tượng xuất hiện liên tục ≥ `FRAME_CONFIRM`
* Có chuyển động (motion detection)
* Không vi phạm `COOLDOWN`

→ Giúp giảm nhiễu và cảnh báo sai.

---

## 8. Kiến trúc hệ thống

Hệ thống sử dụng đa luồng:

```
Capture Thread
      ↓
Frame Queue
      ↓
Detection Thread
      ↓
Event Queue
      ↓
Email Worker
```

### Điểm quan trọng trong thiết kế

* Tách riêng **detect_frame** và **draw_frame**

  * detect_frame: dùng cho YOLO và motion detection
  * draw_frame: dùng để hiển thị

→ Tránh việc overlay làm sai lệch dữ liệu đầu vào của mô hình.

---

## 9. Kết quả đạt được

* Phát hiện người theo thời gian thực từ webcam
* Phân biệt rõ trạng thái xâm nhập
* Gửi cảnh báo ổn định với cơ chế chống spam
* Dashboard hiển thị trực quan và dễ theo dõi

---

## 10. Hạn chế

* Tracking phụ thuộc YOLO nên chưa hoàn toàn ổn định
* Motion detection còn đơn giản
* Chưa tối ưu cho hệ thống lớn
* Dashboard sử dụng polling (chưa realtime hoàn toàn)

---

## 11. Hướng phát triển

* Tích hợp ByteTrack / DeepSORT để cải thiện tracking
* Sử dụng WebSocket thay polling
* Hỗ trợ nhiều vùng ROI
* Kết nối camera RTSP
* Tối ưu hiệu năng GPU

---

## 12. Kết luận

Hệ thống đã xây dựng thành công một pipeline giám sát có cấu trúc, kết hợp giữa:

* Thị giác máy tính (Computer Vision)
* Xử lý thời gian thực
* Thiết kế hệ thống đa luồng

Mặc dù còn hạn chế, hệ thống đã đạt được mục tiêu đề ra và có khả năng mở rộng trong các ứng dụng thực tế.
