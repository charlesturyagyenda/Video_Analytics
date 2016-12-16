import cv2
import sys
import os

fp = os.path.join(sys.path[0], "FIG_2016.mp4")
fp2 = os.path.join(sys.path[0], "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture()
cap.open(fp)

face_cascade = cv2.CascadeClassifier(fp2)
count = 0
while True:
    ret, frame = cap.read()

    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3, 8)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    count +=1
    print(count)
    else:
        break

cv2.destroyAllWindows()
cap.release()
