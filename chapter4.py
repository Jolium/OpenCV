import cv2
import numpy as np


img = np.zeros((512, 512, 3), np.uint8)
print(img)

# img[:] = 255, 0, 0  # color for all image
# img[0:200, 100:250] = 255, 0, 0  # color for just a part of the image

# cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)
# cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 255), 20)

# cv2.rectangle(img, (0, 0), (250, 350), (0, 0, 255), 2)
# cv2.rectangle(img, (100, 100), (350, 450), (255, 0, 255), cv2.FILLED)

cv2.circle(img, (400, 150), 100, (255, 255, 0), 10)

cv2.putText(img, "OpenCV", (100, 200), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 100, 150), 2)


cv2.imshow("Image", img)


cv2.waitKey(0)
