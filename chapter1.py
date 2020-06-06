import cv2


## Images
# print('Package Imported')
#
# img = cv2.imread('Resources/lena.jpg')
#
# cv2.imshow('Output', img)
# cv2.waitKey(0)  # '0' wait for ever; in milliseconds (1000 = 1sec)

## Video
# cap = cv2.VideoCapture("Resources/test_video.mp4")
#
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

## webcam
cap = cv2.VideoCapture(0)  # '0' is the standard webcam
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(10, 100)  # Brightness


while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

