import cv2
import numpy as np
import image_resize

img = cv2.imread('images/IMG_20180331_180458.jpg',0)
img = image_resize.resize(img,width=600)

edges = cv2.Canny(img,100,200)

cv2.imshow('mat',edges)
cv2.waitKey(0)

