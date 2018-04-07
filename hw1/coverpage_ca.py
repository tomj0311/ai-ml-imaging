import sys
import cv2, numpy as np
from matplotlib import pyplot as plt
import pytesseract

import image_resize 

img = cv2.imread(filename='images/659175802_001.tif')
img = image_resize.resize(image=img, width=600)

imgx = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgx = cv2.GaussianBlur(imgx, (3,3), 2)

kernel = np.ones((3,7), np.uint8) #cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(imgx, cv2.MORPH_OPEN, kernel)

cv2.imshow('w', grad)
cv2.waitKey(0)

#thresh = cv2.Canny(grad,127,255,apertureSize = 5)
#BEST SUITED FOR COVER PAGES CA
_, thresh = cv2.threshold(grad, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )

cv2.imshow('x', thresh)
cv2.waitKey(0)

imx, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

indx = 1

for contour in contours:
  [x, y, w, h] = cv2.boundingRect(contour)

  if w > 5 and h > 5:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cropped = img[y :y +  h , x : x + w]

    cv2.imshow(str(indx), cropped)
    cv2.waitKey()

    text = pytesseract.image_to_string(cropped, lang = 'eng')
    print(text)

    indx = indx + 1

cv2.imshow('img', img)
cv2.waitKey(0)