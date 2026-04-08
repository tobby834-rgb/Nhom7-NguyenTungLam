import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ======================
# LOAD MODELS
# ======================

gesture_base = python.BaseOptions(model_asset_path='gesture_recognizer.task')

gesture_options = vision.GestureRecognizerOptions(
    base_options=gesture_base
)

gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)


face_base = python.BaseOptions(model_asset_path='face_landmarker.task')

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base,
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True
)

face_detector = vision.FaceLandmarker.create_from_options(face_options)


pose_base = python.BaseOptions(model_asset_path='Pose_Landmarker_heavy.task')

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base,
    running_mode=vision.RunningMode.VIDEO
)

pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# ======================
# CAMERA
# ======================

cap = cv2.VideoCapture(0)

# ======================
# CONNECTIONS
# ======================

POSE_CONNECTIONS = [
(11,13),(13,15),
(12,14),(14,16),
(11,12),
(11,23),(12,24),
(23,24),
(23,25),(25,27),
(24,26),(26,28)
]

HAND_CONNECTIONS = [
(0,1),(1,2),(2,3),(3,4),
(0,5),(5,6),(6,7),(7,8),
(5,9),(9,10),(10,11),(11,12),
(9,13),(13,14),(14,15),(15,16),
(13,17),(17,18),(18,19),(19,20),
(0,17)
]

# ======================
# MAIN FUNCTION
# ======================

def get_action(show_window=True):

    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame,1)

    timestamp = int(time.time()*1000)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    action = None

    # ======================
    # HAND GESTURE
    # ======================

    gesture_result = gesture_recognizer.recognize(mp_image)

    if gesture_result.gestures:

        gesture_name = gesture_result.gestures[0][0].category_name

        if gesture_name == "Thumb_Up":
            action = "jump"

        cv2.putText(frame,f"Gesture: {gesture_name}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

    if gesture_result.hand_landmarks:

        h,w,_ = frame.shape

        for hand_landmarks in gesture_result.hand_landmarks:

            points=[]

            for lm in hand_landmarks:

                x=int(lm.x*w)
                y=int(lm.y*h)

                points.append((x,y))

                cv2.circle(frame,(x,y),5,(0,255,0),-1)

            for c in HAND_CONNECTIONS:

                cv2.line(frame,points[c[0]],points[c[1]],(0,255,0),2)

    # ======================
    # FACE DETECTION
    # ======================

    face_result = face_detector.detect_for_video(mp_image,timestamp)

    if face_result.face_landmarks:

        h,w,_ = frame.shape

        for face_landmarks in face_result.face_landmarks:

            for lm in face_landmarks:

                x=int(lm.x*w)
                y=int(lm.y*h)

                cv2.circle(frame,(x,y),1,(255,0,0),-1)

    # ======================
    # FACE EXPRESSION
    # ======================

    if face_result.face_blendshapes:

        blendshapes = face_result.face_blendshapes[0]

        smile_score = 0
        blink_score = 0

        for shape in blendshapes:

            if shape.category_name in ["mouthSmileLeft","mouthSmileRight"]:
                smile_score += shape.score

            if shape.category_name in ["eyeBlinkLeft","eyeBlinkRight"]:
                blink_score += shape.score

        if smile_score > 0.8:
            action = "jump"
            cv2.putText(frame,"Smile",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

        if blink_score > 0.6:
            action = "jump"
            cv2.putText(frame,"Blink",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

    # ======================
    # POSE DETECTION
    # ======================

    pose_result = pose_detector.detect_for_video(mp_image,timestamp)

    if pose_result.pose_landmarks:

        h,w,_ = frame.shape

        for pose_landmarks in pose_result.pose_landmarks:

            points=[]

            for lm in pose_landmarks:

                x=int(lm.x*w)
                y=int(lm.y*h)

                points.append((x,y))

                cv2.circle(frame,(x,y),4,(0,255,255),-1)

            for c in POSE_CONNECTIONS:

                if c[0] < len(points) and c[1] < len(points):

                    cv2.line(frame,points[c[0]],points[c[1]],(0,255,255),2)

            # detect jump (raise hand)

            nose = pose_landmarks[0]
            right_wrist = pose_landmarks[16]

            if right_wrist.y < nose.y:
                action = "jump"

                cv2.putText(frame,"POSE JUMP",
                            (20,120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0,255,255),2)

    # ======================
    # SHOW WINDOW
    # ======================

    if show_window:
        cv2.imshow("Vision Test",frame)
        cv2.waitKey(1)

    return action
if __name__ == "__main__":

    while True:

        action = get_action(show_window=True)

        if action:
            print("Action:", action)