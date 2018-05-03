import cv2
import numpy as np
from matplotlib import pyplot as plt

import hist

large = cv2.imread('images/XsSOP.jpg')

gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 9))
grad = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

cv2.imshow('grad', grad)
cv2.waitKey(0)

_, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 3))
connected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow('connected', connected)
cv2.waitKey(0)

imx, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(thresh.shape, dtype=np.uint8)

index = 1

for idx in range(len(contours)):
  x, y, w, h = cv2.boundingRect(contours[idx])
  mask[y:y+h, x:x+w] = 0
    
  cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)

  r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

  #if w > 8 and h > 8:
  cv2.rectangle(large, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

  index = index + 1

cv2.imshow('rects', large)
cv2.waitKey(0)
cv2.destroyAllWindows()
