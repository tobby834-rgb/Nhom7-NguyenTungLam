# FILE DEMO, KHÔNG SỬ DỤNG TRONG APP CHÍNH, CHỈ DÙNG ĐỂ TEST VÀ HIỂN THỊ KẾT QUẢ
# SỬ DỤNG VIDEO DEMO, CHƯA SỬ DỤNG CAMERA (CÓ THỂ THAY ĐỔI INPUT BẰNG CAMERA Ở LINE 27)
# THUẦN PYTHON, KHÔNG SỬ DỤNG FLASK, CHỈ DÙNG ULTRALYTICS YOLO VÀ OPENCV ĐỂ TEST THUẬT TOÁN VÀ HIỂN THỊ KẾT QUẢ, CHƯA TÁCH LOGIC GỬI MAIL

import cv2
import time
import smtplib
import threading
from email.message import EmailMessage
from ultralytics import YOLO
import os

# ================= CONFIG =================
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.6
COOLDOWN = 10
MOTION_THRESHOLD = 5000
FRAME_CONFIRM = 5

EMAIL_SENDER = "sender@gmail.com"
EMAIL_PASSWORD = "điền mật khẩu ứng dụng"
EMAIL_RECEIVER = "receiver@gmail.com"

# ==========================================

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture("running_demo.mp4")

last_sent_time = 0
sent_ids = set()
id_counter = {}

prev_frame = None

# ================= EMAIL =================
def send_email(image_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = "🚨 Intruder detected!"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content("Phát hiện người xâm nhập!")

        with open(image_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(image_path)
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print("📧 Email sent!")

    except Exception as e:
        print("❌ Email error:", e)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    h, w = frame.shape[:2]

    # ===== MOTION DETECTION =====
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    motion_detected = False

    if prev_frame is not None:
        delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion = cv2.countNonZero(thresh)

        if motion > MOTION_THRESHOLD:
            motion_detected = True

    prev_frame = gray

    # ===== YOLO TRACKING =====
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > CONF_THRESHOLD and motion_detected:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                track_id = int(box.id[0]) if box.id is not None else -1

                # ===== ROI (vùng cấm bên phải) =====
                if x1 > w // 2:

                    # ===== COUNT FRAME =====
                    if track_id not in id_counter:
                        id_counter[track_id] = 0

                    id_counter[track_id] += 1

                    # ===== VẼ =====
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    current_time = time.time()

                    # ===== ĐIỀU KIỆN GỬI MAIL =====
                    if (
                        id_counter[track_id] > FRAME_CONFIRM
                        and track_id not in sent_ids
                        and (current_time - last_sent_time > COOLDOWN)
                    ):
                        image_path = f"alert_{track_id}.jpg"
                        cv2.imwrite(image_path, frame)

                        threading.Thread(
                            target=send_email,
                            args=(image_path,)
                        ).start()

                        sent_ids.add(track_id)
                        last_sent_time = current_time

                        # log
                        with open("log.txt", "a") as f:
                            f.write(f"Intruder ID {track_id} at {time.ctime()}\n")

    # ===== RESET ID =====
    if len(sent_ids) > 50:
        sent_ids.clear()

    # ===== VẼ VÙNG CẤM =====
    cv2.rectangle(frame, (w//2, 0), (w, h), (255, 0, 0), 2)
    cv2.putText(frame, "RESTRICTED AREA", (w//2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)    

    # ===== HIỂN THỊ =====
    cv2.imshow("Smart Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()