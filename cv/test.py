import cv2 as cv
import numpy as np
import math

# ===== VIDEO =====
vid = cv.VideoCapture("bang_chuyen2.mp4")

# ===== THAM SỐ =====
LINE_1 = 600
DIST_THRESHOLD = 25

MIN_AREA = 80
MAX_AREA = 3000
MIN_CIRCULARITY = 0.7

# ===== BIẾN =====
tracked_objects = {}
counted_ids = set()
object_id = 0
count = 0


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


while True:
    ret, frame = vid.read()
    if not ret:
        break

    # ===== VẼ LINE =====
    cv.line(frame, (LINE_1, 0), (LINE_1, frame.shape[0]), (0, 0, 255), 2)

    # ===== PREPROCESS =====
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv.threshold(
        blur, 0, 255,
        cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # ===== FIND CONTOURS =====
    contours, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    current_centers = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        current_centers.append(center)

        cv.circle(frame, center, radius, (0, 255, 0), 2)
        cv.circle(frame, center, 2, (0, 0, 255), 3)

    # ===== TRACKING =====
    new_tracked = {}

    for center in current_centers:
        matched = False
        best_id = None
        min_dist = float("inf")

        for oid, prev_center in tracked_objects.items():
            d = distance(center, prev_center)
            if d < DIST_THRESHOLD and d < min_dist:
                min_dist = d
                best_id = oid

        if best_id is not None:
            prev_center = tracked_objects[best_id]
            new_tracked[best_id] = center
            matched = True

            # ===== CHECK LINE CROSS =====
            if best_id not in counted_ids and prev_center[0] < LINE_1 <= center[0]:
                count += 1
                counted_ids.add(best_id)
                print(f"[Số hình tròn đã đi qua line đỏ] Count = {count}")

        if not matched:
            new_tracked[object_id] = center
            object_id += 1

    tracked_objects = new_tracked

    # ===== HIỂN THỊ =====
    cv.imshow("Circle Counter", frame)
    cv.imshow("Threshold", thresh)

    if cv.waitKey(10) & 0xFF == ord("q"):
        break

vid.release()
cv.destroyAllWindows()
