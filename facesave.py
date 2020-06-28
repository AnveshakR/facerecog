import cv2
import os
import shutil
vid_capture = cv2.VideoCapture(0)

name = input("Enter name of person: ")
data_dir = "data"
pic_dir = os.path.join(data_dir, name)
count = 0
#cv2.imshow("Instructions", cv2.imread('walking.jpg'))

while(count<300):
    ret, frame = vid_capture.read()
    cv2.imshow("Webcam feed", frame)
    if cv2.waitKey(1) &0XFF == ord('y'):
        try:
            os.makedirs(pic_dir)
            print("Directory made")
            cv2.destroyWindow("Instructions")
        except FileExistsError:
            print("Data already exists for this name.")
            break

    cv2.imwrite(os.path.join(pic_dir,"frame%d.jpg" % count), frame)
    count+=1

    if cv2.waitKey(1) &0XFF == ord('x'):
        break
vid_capture.release()
cv2.destroyAllWindows()
count = 0

ch = input("Confirm face? (y/n)")
if (ch=='y' or ch=='Y'):
    print(name, "'s face is registered")
elif(ch == 'n' or ch=='N'):
    shutil.rmtree(os.path.join(pic_dir))
    print("Cancelled")