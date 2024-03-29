import cv2
import face_recognition
import numpy as np
from matplotlib import pyplot as plt
import os
from statistics import mode
from dotenv import load_dotenv

load_dotenv()

# loading values from .env file
faces_dir = os.getenv('faces_directory')
width1 = float(os.getenv('width1'))
width2 = float(os.getenv('width2'))
width3 = float(os.getenv('width3'))
slope1 = float(os.getenv('slope1'))
slope2 = float(os.getenv('slope2'))
slope3 = float(os.getenv('slope3'))
slope4 = float(os.getenv('slope4'))

known_faces = []
names = []

#loading and defining encodings for the faces to be recognized from the defined directory

for filename in os.listdir(faces_dir):
    loadface = face_recognition.load_image_file(faces_dir+ "\\" +filename)
    encoding = face_recognition.face_encodings(loadface)[0]
    known_faces.append(encoding)
    names.append(filename[:len(filename)-4])

face_locations = []
face_encodings = []
face_name = []
eye_heights = []
eye_widths = []


# using 0 for integrated webcam, replace with RTSP/HTTP camera URL if being used
cap = cv2.VideoCapture("blinktest.mp4")


while True:
    ret, frame = cap.read()
    temp_name = ""
    
    if ret is False:
        break

    #converting frame BGR to RGB since opencv loads images in BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #finding and encoding images in the given frame
    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
    face_encodes = face_recognition.face_encodings(frame,face_locations)

    #comparing faces with the previously encoded faces to find a match
    for face_encode in face_encodes:

        match = face_recognition.compare_faces(known_faces, face_encode, tolerance=0.55)
        try:
            temp_name = names[match.index(True)]
            face_name.append(temp_name)
        except:
            pass
    
    if face_name != "":
        try:
            #extracting facial landmarks from detected face
            landmarks = face_recognition.face_landmarks(frame, model='large')

            #drawing lines around the eyes of the detected face, can be commented out if not needed
            cv2.polylines(frame, [np.array(landmarks[0]['left_eye'])], isClosed=True, color=(0,0,255), thickness=2)
            cv2.polylines(frame, [np.array(landmarks[0]['right_eye'])], isClosed=True, color=(0,0,255), thickness=2)
            left_eye = np.array(landmarks[0]["left_eye"])
            right_eye = np.array(landmarks[0]["right_eye"])
    
            #finding the height and width average of both eyes
            left_diff = (left_eye[1][1] + left_eye[2][1])/2 - (left_eye[4][1] + left_eye[5][1])/2
            right_diff = (right_eye[1][1] + right_eye[2][1])/2 - (right_eye[4][1] + left_eye[5][1])/2
            eye_heights.append((left_diff + right_diff)/2)
            eye_widths.append(abs((left_eye[3][0] - left_eye[0][0]) + (right_eye[3][0] - right_eye[0][0]))/2)

        
        except:
            pass

    #converting RGB back to BGR for opencv
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("output",frame)

    #press 'q' or 'Q' to interrupt camera feed
    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break

cap.release()
cv2.destroyAllWindows()

if eye_heights != []:

    # generating a linearly spaced array of numbers equal to the number of frames-
    # -in which a recognized face was detected
    x_axis = np.linspace(0, len(eye_heights)-1, len(eye_heights))
    slopes = []

    # calculating adjacent eye height slopes
    for i in range(len(eye_heights)-1):
        if i == len(eye_heights):
            break
        slopes.append((eye_heights[i]-eye_heights[i+1]))

    avg_slope = np.sum(slopes)/len(slopes) #average slope
    min_slope = min(slopes) #minimum slope

    avg_eye_width = np.sum(eye_widths)/len(eye_widths) #average eye width, used for determining distance of user from camera

    # print(len(slopes))
    # print(abs(avg_slope-min_slope))

    plt.axline(xy1=(0,avg_slope),slope=0)
    plt.axline(xy1=(0,min_slope),slope=0)
    plt.plot(slopes)
    plt.plot(eye_heights)
    plt.scatter(x_axis, eye_heights)
    plt.grid()
    plt.show()

    fig, (ax1,ax2) = plt.subplots(2, sharex=True)
    fig.grid()
    ax1.plot(slopes)
    ax1.set_yticks(np.arange(min(slopes), max(slopes)+1, 0.5))
    ax2.plot(eye_widths)
    ax2.set_yticks(np.arange(min(eye_widths), max(eye_widths)+1, 0.5))
    
    ax1.set_title("slopes")
    ax2.set_title("eye widths")
    plt.show()

    # print(slopes)
    # print(eye_heights)
    # print(eye_widths)
