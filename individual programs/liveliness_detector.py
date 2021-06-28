import cv2
import face_recognition
import numpy as np
from matplotlib import pyplot as plt
import time

time_limit = time.time() + 7

eye_heights = []
eye_widths = []
cap = cv2.VideoCapture(0)


while time.time() <= time_limit:
    ret, frame = cap.read()
    
    if ret is False:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        landmarks = face_recognition.face_landmarks(frame, model='large')

        cv2.polylines(frame, [np.array(landmarks[0]['left_eye'])], isClosed=True, color=(0,0,255), thickness=2)
        cv2.polylines(frame, [np.array(landmarks[0]['right_eye'])], isClosed=True, color=(0,0,255), thickness=2)

        left_eye = np.array(landmarks[0]["left_eye"])
        right_eye = np.array(landmarks[0]["right_eye"])

        left_diff = (left_eye[1][1] + left_eye[2][1])/2 - (left_eye[4][1] + left_eye[5][1])/2
        right_diff = (right_eye[1][1] + right_eye[2][1])/2 - (right_eye[4][1] + left_eye[5][1])/2
        eye_heights.append((left_diff + right_diff)/2)
        eye_widths.append(abs((left_eye[3][0] - left_eye[0][0]) + (right_eye[3][0] - right_eye[0][0]))/2)
    
    except:
        print("Make sure your face is properly visible")
        pass

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("output",frame)


    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break

cap.release()
cv2.destroyAllWindows()

slopes = []
for i in range(len(eye_heights)-1):
    if i == len(eye_heights):
        break
    slopes.append((eye_heights[i]-eye_heights[i+1]))

avg_eye_width = np.sum(eye_widths)/len(eye_widths)

is_real = False

if avg_eye_width <= 23 and min(slopes) <= -0.75:
    is_real = True

if avg_eye_width > 23 and avg_eye_width <= 38 and min(slopes) <= -2.75:
    is_real = True

if avg_eye_width > 38 and avg_eye_width <= 54 and min(slopes) <= -4.75:
    is_real = True

if avg_eye_width > 54 and min(slopes) <= -5.75:
    is_real = True

if is_real:
    print("Real Person")
else:
    print("Fake")

# x_axis = np.linspace(0, len(eye_heights)-1, len(eye_heights))
# if abs(avg_slope-min_slope) > 4:
#     print("Real Person")
# else:
#     print("Photograph")

# print(len(slopes))
# print(abs(avg_slope-min_slope))

# plt.axline(xy1=(0,avg_slope),slope=0)
# plt.axline(xy1=(0,min_slope),slope=0)
# plt.plot(slopes)
# plt.plot(eye_heights)
# plt.scatter(x_axis, eye_heights)
# plt.grid()
# plt.show()

# print(slopes)
# print(eye_heights)
# print(eye_widths)