import cv2
import numpy as np

from stack_func import stackImages


# Images
widthImg = 480
heightImg = 640


# parameters
numberPlateCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_russian_plate_number.xml")
minArea = 500
color = (0, 255, 255)
count = 0

# webcam
frame_width = 640
frame_height = 480


cap = cv2.VideoCapture(0)  # '0' is the standard webcam
cap.set(3, frame_width)  # Width
cap.set(4, frame_height)  # Height
cap.set(10, 150)  # Brightness

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlate = numberPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlate:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x, y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=color, thickness=2)
            imgRoi = img[y:y+h, x:x+w]
            cv2.imshow("Region of Interest", img)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Resources/Saved_Plates/Number_Plate_"+str(count)+".jpg", imgRoi)
        count += 1
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Image saved", (150, 256),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
