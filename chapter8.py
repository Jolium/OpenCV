import cv2
import numpy as np
from stack_func import stackImages


def getContours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print('::::::::::::::::')
        print(area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)
            print(perimeter)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            obj_corners = len(approx)
            print(obj_corners)
            x, y, w, h = cv2.boundingRect(approx)
            if obj_corners == 3:
                obj_type = "Triangle"

            elif obj_corners == 4:
                asp_ratio = w / float(h)
                if 0.95 < asp_ratio < 1.05:
                    obj_type = "Square"
                else:
                    obj_type = "Rectangle"

            elif obj_corners > 4:
                obj_type = "Circle"

            else:
                obj_type = "None"

            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imgContour, obj_type,
                        (x+(w//2)-30, y+(h//2)-20),  # (x_position, y_position)
                        cv2.FONT_HERSHEY_COMPLEX, 0.7,  # font, size
                        (0, 0, 0), 2)  # (color), letter_thickness


path = 'Resources/shapes.png'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)

imgBlank = np.zeros_like(img)

getContours(imgCanny)

imgStack = stackImages(0.7, ([img, imgGray, imgBlur],
                             [imgCanny, imgContour, imgBlank]))

getContours(imgCanny)
# cv2.imshow("Original", img)
# cv2.imshow("Image Gray", imgGray)
# cv2.imshow("Image Blur", imgBlur)
cv2.imshow("Image Stack", imgStack)

cv2.waitKey(0)
