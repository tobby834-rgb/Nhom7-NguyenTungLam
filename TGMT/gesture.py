import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# Load model
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

def detect_gesture():
    ret, frame = cap.read()
    if not ret:
        return None

    # lật camera cho tự nhiên
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = recognizer.recognize(mp_image)

    if result.gestures and len(result.gestures) > 0:
        gesture = result.gestures[0][0]
        return gesture.category_name

    return None


def close_camera():
    cap.release()
    cv2.destroyAllWindows()