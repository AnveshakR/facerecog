import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
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
        if (max(left_eye[:,0])-min(left_eye[:,0]) + max(right_eye[:,0])-min(right_eye[:,0]))/2 > 35:

            left_diff = max(left_eye[:,1])-min(left_eye[:,1])
            right_diff = max(right_eye[:,1])-min(right_eye[:,1])
            if (left_diff+right_diff)/2 < 10:
                print("Blink")
            else:
                print("Open")
        else:
            print("Move closer to camera")
    
    except:
        pass

    

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("output",frame)

    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break

cap.release()
cv2.destroyAllWindows()