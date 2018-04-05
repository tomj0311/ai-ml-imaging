import cv2
import numpy as np
import image_resize

#method 1
img = cv2.imread('images/51H1M.png', 0)

cv2.imshow('wing', img)
cv2.waitKey(0)

# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# cv2.imshow('hsv', hsv[:,:,0])

# (thresh, im_bw) = cv2.threshold(hsv[:,:,0], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# cv2.imshow('OTSU', im_bw)
cv2.waitKey(0)

#method 2
filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 3)

# Some morphology to clean up image
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing',closing)
cv2.waitKey(0)