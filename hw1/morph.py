import cv2
import numpy as np
import image_resize
import pytesseract

#method 1
#img = cv2.imread('images/IMG_20180331_180458.jpg', 0) handwritten
img = cv2.imread('images/659175802_001.tif', 0)
img = image_resize.resize(img, width=600)

# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# cv2.imshow('hsv', hsv[:,:,0])

# (thresh, im_bw) = cv2.threshold(hsv[:,:,0], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# cv2.imshow('OTSU', im_bw)

#method 2
filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 3)

# Some morphology to clean up image
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

text = pytesseract.image_to_string(closing, lang = 'eng')
print(text)

cv2.imshow('closing',closing)
cv2.waitKey(0)