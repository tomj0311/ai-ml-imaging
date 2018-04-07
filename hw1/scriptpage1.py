import sys
import cv2, numpy as np
from matplotlib import pyplot as plt
import json, codecs

import image_resize 

img = cv2.imread(filename='images/659175809_008.tif')
img = image_resize.resize(image=img, width=600)

imgx = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgx = cv2.GaussianBlur(imgx, (5,5), 2)
# imgx = cv2.blur(imgx, (5,5), 2)

kernel = np.ones((1,4), np.uint8) #cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#grad = cv2.morphologyEx(imgx, cv2.MORPH_DILATE, kernel)
grad = cv2.dilate(imgx, kernel, iterations=2)

cv2.imshow('w', grad)
cv2.waitKey(0)

#thresh = cv2.Canny(grad,127,255,apertureSize = 5)
#BEST SUITED FOR COVER PAGES CA
_, thresh = cv2.threshold(grad, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )

cv2.imshow('x', thresh)
cv2.waitKey(0)

imx, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for contour in contours:
  [x, y, w, h] = cv2.boundingRect(contour)

  if h > 5:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)