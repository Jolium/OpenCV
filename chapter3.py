import cv2
import numpy as np


img = cv2.imread("Resources/lamborghini.jpg")


print(img.shape)  # (492, 875, 3) = height, width, number of channels(BGR)

imgResize = cv2.resize(img, (300, 200))
imgCropped = img[0:200, 200:500]  # [width, height]

cv2.imshow("Image", img)
cv2.imshow("Image Resized", imgResize)
cv2.imshow("Image Cropped", imgCropped)

cv2.waitKey(0)
