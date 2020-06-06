import cv2
import numpy as np

from stack_func import stackImages


img = cv2.imread("Resources/lena.jpg")

# imgHor = np.hstack((img, img))
# imgVer = np.vstack((img, img))
#
# cv2.imshow("Horizontal", imgHor)
# cv2.imshow("Vertical", imgVer)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgStack = stackImages(0.5, ([img, img, img], [img, imgGray, img], [img, img, img]))

cv2.imshow("Imagw Stack", imgStack)


cv2.waitKey(0)
