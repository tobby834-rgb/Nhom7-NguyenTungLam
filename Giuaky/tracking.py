import cv2
from ultralytics import YOLO

# ===== CONFIG =====
VIDEO_PATH = "plate_test.mp4"
MODEL_PATH = "yolov8s.pt"

CONF_THRES = 0.35

VEHICLE_CLASSES = [2, 7]  # car, truck
CLASS_NAMES = {
    2: "car",
    7: "truck"
}

# ===================

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# ===== GET FRAME SIZE =====
ret, frame = cap.read()
if not ret:
    print("Không đọc được video")
    exit()

h, w = frame.shape[:2]

ROI_Y1 = int(0.3 * h)
ROI_Y2 = int(0.85 * h)
LINE_Y = int(0.4 * h)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ===== COUNT =====
counted_ids = set()
prev_positions = {}

total_count = 0
count_by_type = {
    "car": 0,
    "truck": 0
}

# ===== LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[ROI_Y1:ROI_Y2, :]

    # ===== BYTE TRACK =====
    results = model.track(roi, persist=True, conf=CONF_THRES)[0]

    if results.boxes is None:
        cv2.imshow("Vehicle Counting", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    ids = results.boxes.id

    if ids is None:
        continue

    ids = ids.cpu().numpy().astype(int)

    # ===== PROCESS =====
    for box, cls, obj_id in zip(boxes, classes, ids):

        cls = int(cls)
        if cls not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box)

        # chuyển về frame gốc
        y1 += ROI_Y1
        y2 += ROI_Y1

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        label = CLASS_NAMES.get(cls, "vehicle")

        # ===== DRAW =====
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} ID:{obj_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # ===== COUNT =====
        prev_cy = prev_positions.get(obj_id, None)

        if prev_cy is not None:

            # đi xuống
            if prev_cy < LINE_Y and cy >= LINE_Y:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_count += 1
                    count_by_type[label] += 1

            # đi lên
            elif prev_cy > LINE_Y and cy <= LINE_Y:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_count += 1
                    count_by_type[label] += 1

        prev_positions[obj_id] = cy

    # ===== LINE =====
    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0,0,255), 2)

    # ===== DISPLAY =====
    cv2.putText(frame, f"Total: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    y_offset = 80
    # for k, v in count_by_type.items():
    #     cv2.putText(frame, f"{k}: {v}", (20, y_offset),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    #     y_offset += 30

    cv2.imshow("Vehicle Counting", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()