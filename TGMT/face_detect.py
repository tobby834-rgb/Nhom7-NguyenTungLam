import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

# Load model
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True
)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)


def detect_face_expression():

    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, timestamp)

    if result.face_blendshapes:

        blendshapes = result.face_blendshapes[0]

        smile_score = 0
        blink_score = 0

        for shape in blendshapes:

            if shape.category_name == "mouthSmileLeft":
                smile_score += shape.score

            if shape.category_name == "mouthSmileRight":
                smile_score += shape.score

            if shape.category_name == "eyeBlinkLeft":
                blink_score += shape.score

            if shape.category_name == "eyeBlinkRight":
                blink_score += shape.score

        # Detect smile
        if smile_score > 0.8:
            return "Smile"

        # Detect blink
        if blink_score > 0.6:
            return "Blink"

    return None


def close_camera():
    cap.release()
    cv2.destroyAllWindows()