import cv2

cap = cv2.VideoCapture()
cap.open("slow_traffic_small.mp4")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3, 8)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('Video Feed at FLE Instance....',frame)
        cv2.waitKey(25)

    else:
        break

cv2.destroyAllWindows()
cap.release()
