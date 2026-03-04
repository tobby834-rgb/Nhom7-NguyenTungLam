import cv2 as cv
import numpy as np
import os
from mail_sender import send_email
import threading
from mail_sender import open_browser

recog_tool = cv.face.LBPHFaceRecognizer_create()
recog_tool.read("face_recog_model.yml") #doc model da train
labels_dict = np.load("labels_dict.npy", allow_pickle=True).item() #doc labels_dict da luu
face_cascade = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")
email_sent = False
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if frame is not None:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        dgray = cv.GaussianBlur(gray, (5,5), 0)
        faces = face_cascade.detectMultiScale(dgray, 1.3, 5)

        for (x,y,w,h) in faces:
            face_img = dgray[y:y+h, x:x+w]
            name, dotincay = recog_tool.predict(face_img)
            if dotincay < 80:
                name = labels_dict[name]
                color = (0,255,0)
                if not email_sent:
                    threading.Thread(target=send_email, args=(face_img, name)).start()
                    email_sent = True
                    open_browser()
            else:
                name = "Unknown"
                color = (0,0,255)
            
            cv.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv.putText(frame, name, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv.imshow('Face Recognition', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()