import cv2 as cv
import easyocr

video_path = "plate2.mp4"
cap = cv.VideoCapture(video_path)

reader = easyocr.Reader(['en'], gpu=False)

frame_count = 0
last_text = ""

# ===== DETECT =====
def detect_plate(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    edges = cv.Canny(binary, 30, 150)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    plates = []
    img_size = frame.shape[0] * frame.shape[1]

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        ratio = w / h
        area = (w * h) / img_size

        if 1.5 < ratio < 6.5 and 0.005 < area < 0.3:
            plates.append((x, y, w, h))

    return plates

# ===== CROP =====
def crop_plate(img, x, y, w, h, pad=5):
    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(img.shape[1], x+w+pad)
    y2 = min(img.shape[0], y+h+pad)
    return img[y1:y2, x1:x2]

# ===== OCR =====
def recognize_plate(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ⚡ upscale cho dễ đọc
    gray = cv.resize(gray, None, fx=2, fy=2)

    result = reader.readtext(
        gray,
        allowlist="0123456789ABCDEFGHKLMNPQRSTUVXY",
        detail=0
    )

    return result[0] if result else ""

# ===== MAIN =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (800, 450))  # ⚡ tăng nhẹ độ phân giải

    plates = detect_plate(frame)

    for (x, y, w, h) in plates:
        plate_img = crop_plate(frame, x, y, w, h)

        # ⚡ OCR mỗi 10 frame
        if frame_count % 10 == 0:
            last_text = recognize_plate(plate_img)

        # vẽ box
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # vẽ text
        cv.putText(frame, last_text, (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.8, (0,255,0), 2)

        break  # ⚡ chỉ xử lý 1 biển

    cv.imshow("Plate Detection", frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv.destroyAllWindows()