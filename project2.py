import cv2
import numpy as np

from stack_func import stackImages


# Images
widthImg = 480
heightImg = 640

# webcam
frame_width = 640
frame_height = 480

cap = cv2.VideoCapture(0)  # '0' is the standard webcam
cap.set(3, frame_width)  # Width
cap.set(4, frame_height)  # Height
cap.set(10, 150)  # Brightness



def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDialation = cv2.dilate((imgCanny), kernel, iterations=2)
    imgThreshole = cv2.erode(imgDialation, kernel, iterations=1)

    return imgThreshole


def getContours(image):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

        # else:
        #     print("No Picture with area greater than 500")
        #     break

    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)  # the biggest 4 points
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # crop 20 pixels from borders image
    imgCropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgCropped


while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    imgThreshole = preProcessing(img)
    biggest = getContours(imgThreshole)
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)

        imageArray = ([img, imgThreshole],
                      [imgContour, imgWarped])

        cv2.imshow("Final Document", imgWarped)

    else:
        imageArray = ([img, imgThreshole],
                      [img, img])

    stackedImages = stackImages(0.6, imageArray)

    cv2.imshow("Video", stackedImages)
    # cv2.imshow("Final Document", imgWarped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break