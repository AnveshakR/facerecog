from json import load
import cv2
import numpy as np
from math import ceil
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Launching Tensorlow...")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def eye_pred(img):
    img = cv2.resize(img,(200,200))
    img = img_to_array(img)
    img = np.array(img)
    img = img.reshape(1,200,200,1)
    categories = ['closed','open']
    result = model.predict(img)
    return categories[round(result[0][0])]

def detectAndDisplay(frame):
    frame = np.array(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:

        frame = cv2.rectangle(frame, [x,y],[x+w,y+h//2], (0,0,255),4)
        faceROI = frame_gray[y:y+h//2,x:x+w]

        eyes = eyes_cascade.detectMultiScale(faceROI)
        flag = 0
        for (x2,y2,w2,h2) in eyes:
            
            cv2.rectangle(frame,[x+x2-15,y+y2-15],[x+x2+w2+15, y+y2+h2+15],(0,0,255),4)
            status = eye_pred(frame_gray[x+x2-15:x+x2+w2+15,y+y2-15:y+y2+h2+15])
            cv2.putText(frame,status,(x+x2-15,y+y2+h2+30),cv2.FONT_HERSHEY_COMPLEX,0.75,(255,0,0),2)
            # ctr_x = x + x2 + w2//2
            # ctr_y =  y + y2 + h2//2
            # pts = np.array([[ctr_x+h2//2,ctr_y],[ctr_x,ctr_y+w2//2],[ctr_x-h2//2,ctr_y],[ctr_x,ctr_y-w2//2]], np.int32) #t,r,b,l
            # pts = pts.reshape((-1,1,2))
            # frame = cv2.polylines(frame, [pts], True, (255,0,0), 4)
            # for i in pts:
            #     frame = cv2.circle(frame, i[0], 1, (0,255,0), 4)
            # frame = cv2.circle(frame, [x+x2+(w2//2),y+y2+(h2//2)], 1, (0,255,0), 4)
            flag+=1

    cv2.imshow('output', frame)

face_cascade_name = r"C:\Users\anves\Documents\haarcascades\haarcascade_frontalface_alt.xml"
eyes_cascade_name = r"C:\Users\anves\Documents\haarcascades\haarcascade_lefteye_2splits.xml"
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('Error loading eyes cascade')
    exit(0)

print("Loaded.")

model = load_model('model3_gray.h5')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
if not cap.isOpened:
    print('Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        pass
    
    detectAndDisplay(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break