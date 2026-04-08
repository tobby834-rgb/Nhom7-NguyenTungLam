import cv2
from ultralytics import YOLO

# ===== CONFIG =====
MODEL_PATH = "yolov10_models\\best.pt"
VIDEO_PATH = "dogs.mp4" 
CONF_THRES = 0.4

# ===== LOAD MODEL =====
model = YOLO(MODEL_PATH)

# ===== LOAD VIDEO =====
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Không mở được video")
    exit()

# ===== LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect
    results = model(frame, conf=CONF_THRES)
    
    # vẽ bbox
    frame = results[0].plot()
    # in ra tên và độ tin cậy
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            print(f"{name}: {conf:.2f}")
    # show
    cv2.imshow("Detection", frame)

    # nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== CLEAN =====
cap.release()
cv2.destroyAllWindows()