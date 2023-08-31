import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
alarm_status = False
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict(r_eye)
        if np.argmax(rpred) == 1:
            lbl = 'Open'
        if np.argmax(rpred) == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        if np.argmax(lpred) == 1:
            lbl = 'Open'
        if np.argmax(lpred) == 0:
            lbl = 'Closed'
        break

    if np.argmax(rpred) == 0 and np.argmax(lpred) == 0:
        score += 1
        if score > 15:
            cv2.putText(frame, "Closed", (10, height-20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            if not alarm_status:
                cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                try:
                    sound.play()
                except:
                    pass
                alarm_status = True
                thicc += 2
                if thicc > 16:
                    thicc = 16
    else:
        score = 0
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if alarm_status:
            thicc -= 2
            if thicc < 2:
                thicc = 2
            if score <= 15:
                alarm_status = False

    cv2.putText(frame, 'Count:'+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if thicc > 2:
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





