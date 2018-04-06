import cv2
import numpy as np
from matplotlib import pyplot as plt

large = cv2.imread('images/659175813_012.tif')
rgb = cv2.pyrDown(large)
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

_, thresh = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))

connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

imx, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

mask = np.zeros(thresh.shape, dtype=np.uint8)

index = 1

for idx in range(len(contours)):
  x, y, w, h = cv2.boundingRect(contours[idx])
  mask[y:y+h, x:x+w] = 0
    
  cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)

  r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

  if r > 0.45 and w > 8 and h > 8:
    cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

    cropped = rgb[y :y +  h , x : x + w]
    # show the portion
    # cv2.imshow(str(index) ,cropped)
    # cv2.waitKey(0)

    index = index + 1

cv2.imshow('rects', rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
