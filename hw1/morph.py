import cv2
import numpy as np
import image_resize
import pytesseract

#method 1
#img = cv2.imread('images/IMG_20180331_180458.jpg', 0) handwritten
img = cv2.imread('images/51H1M.png', 0)

filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 111, 3)

cv2.imshow('filtered',filtered)
cv2.waitKey(0)

# Some morphology to clean up image
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)

cv2.imshow('opening',opening)
cv2.waitKey(0)

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closing',closing)
cv2.waitKey(0)

text = pytesseract.image_to_string(closing, lang = 'eng')
print(text)
