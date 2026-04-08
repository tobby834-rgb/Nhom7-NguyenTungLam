from flask import Flask, Response, render_template, jsonify
import cv2, time, threading, os
from ultralytics import YOLO
from queue import Queue
from dotenv import load_dotenv
from mail_sender import send_email

load_dotenv()

app = Flask(__name__)

# ===== CONFIG =====
MODEL_PATH = os.getenv("MODEL_PATH")

CONF_THRESHOLD = 0.6
FRAME_CONFIRM = 5
COOLDOWN = 10
ID_TTL = 30

# ===== GLOBAL =====
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_queue = Queue(maxsize=10)
event_queue = Queue()

prev_frame = None
logs = []

id_counter = {}
id_last_seen = {}

last_sent_time = 0
output_frame = None

# ===== EMAIL WORKER =====
def email_worker():
    while True:
        img_path = event_queue.get()
        send_email(img_path)
        event_queue.task_done()

# ===== CAPTURE =====
def capture_frames():
    while True:
        success, frame = cap.read()

        if not success:
            time.sleep(0.1)
            continue

        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)

# ===== PROCESS =====
def process_frames():
    global prev_frame, last_sent_time, output_frame

    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        frame = cv2.resize(frame, (640, 360))
        h, w = frame.shape[:2]
        roi_x = int(w * 0.75)

        # ===== TÁCH FRAME =====
        detect_frame = frame.copy()   # dùng cho YOLO + motion
        draw_frame = frame.copy()     # dùng để vẽ

        # ===== MOTION (dùng detect_frame sạch) =====
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        motion_detected = False
        if prev_frame is not None:
            delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            if cv2.countNonZero(thresh) > 5000:
                motion_detected = True

        prev_frame = gray

        # ===== YOLO (dùng detect_frame) =====
        results = model.track(detect_frame, persist=True, verbose=False)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls != 0 or conf < CONF_THRESHOLD:
                    continue

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                # ===== ROI CHECK (center) =====
                center_x = (x1 + x2) // 2
                in_roi = center_x > roi_x

                now = time.time()

                # ===== TTL RESET =====
                if track_id in id_last_seen:
                    if now - id_last_seen[track_id] > ID_TTL:
                        id_counter[track_id] = 0

                id_last_seen[track_id] = now
                id_counter[track_id] = id_counter.get(track_id, 0) + 1

                # ===== COLOR =====
                if in_roi:
                    color = (255, 0, 0)  # xanh dương
                    label = f"INTRUDER {track_id}"
                else:
                    color = (0, 255, 0)  # xanh lá
                    label = f"PERSON {track_id}"

                # ===== DRAW =====
                cv2.rectangle(draw_frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(draw_frame, label,
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

                # ===== ALERT =====
                if (in_roi and
                    id_counter[track_id] > FRAME_CONFIRM and
                    motion_detected and
                    now - last_sent_time > COOLDOWN):

                    img_path = f"static/alert_{track_id}_{int(now)}.jpg"
                    cv2.imwrite(img_path, draw_frame)

                    event_queue.put(img_path)

                    log = {
                        "time": time.strftime("%H:%M:%S"),
                        "id": track_id,
                        "image": img_path
                    }

                    logs.insert(0, log)
                    if len(logs) > 20:
                        logs.pop()

                    last_sent_time = now

        # ===== ROI OVERLAY (CHỈ VẼ, không ảnh hưởng detection) =====
        overlay = draw_frame.copy()
        cv2.rectangle(overlay, (roi_x, 0), (w, h), (0, 0, 255), -1)
        alpha = 0.2
        draw_frame = cv2.addWeighted(overlay, alpha, draw_frame, 1 - alpha, 0)

        # ===== ROI BORDER =====
        cv2.rectangle(draw_frame, (roi_x, 0), (w, h), (0, 0, 255), 2)

        # ===== OUTPUT =====
        _, buffer = cv2.imencode('.jpg', draw_frame)
        output_frame = buffer.tobytes()

        time.sleep(0.02)  # giới hạn FPS

# ===== STREAM =====
def generate_frames():
    while True:
        if output_frame is None:
            time.sleep(0.01)
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               output_frame + b'\r\n')

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    return jsonify(logs)

# ===== MAIN =====
if __name__ == "__main__":
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=process_frames, daemon=True).start()
    threading.Thread(target=email_worker, daemon=True).start()

    app.run(debug=False)