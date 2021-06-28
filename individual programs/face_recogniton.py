import face_recognition
import cv2
from face_recognition.api import face_encodings
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

faces_dir = os.getenv('faces_directory')

known_faces = []
names = []
for filename in os.listdir(faces_dir):
    loadface = face_recognition.load_image_file(faces_dir+filename)
    encoding = face_recognition.face_encodings(loadface)[0]
    known_faces.append(encoding)
    names.append(filename[:len(filename)-4])

video_capture = cv2.VideoCapture(0)
fps = video_capture.get(cv2.CAP_PROP_FPS)
w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output = cv2.VideoWriter("output_{}.avi".format(src[:len(src)-4]),fourcc, fps, (w,h))

face_locations = []
face_encodes = []
face_name = []
process_this_frame = True

ret=True
while True:
    ret, frame = video_capture.read()
    old = frame
    if ret is False:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
    face_encodes = face_recognition.face_encodings(frame,face_locations)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    face_name = []

    for face_encode in face_encodes:

        match = face_recognition.compare_faces(known_faces, face_encode, tolerance=0.55)
        try:
            face_name.append(names[match.index(True)])
        except:
            pass

    for(top, right, bottom, left), name in zip(face_locations, face_name):
        if not name:
            continue

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0,0,0), 1)


    # for face_location in face_locations:

    #     top, right, bottom, left = face_location
    #     frame = cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)

    # combined = cv2.hconcat([old,frame])
    # cv2.imshow("output",cv2.resize(combined,(1280,720)))

    

    cv2.imshow("output",cv2.resize(frame, (int(w/2),int(h/2))))

    #output.write(frame)


    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break

video_capture.release()
cv2.destroyAllWindows()