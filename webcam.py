import cv2


def web_cam():
    ## webcam
    frame_width = 640
    frame_height = 480
    cap = cv2.VideoCapture(0)  # '0' is the standard webcam
    cap.set(3, frame_width)  # Width
    cap.set(4, frame_height)  # Height
    cap.set(10, 150)  # Brightness

    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
