import cv2
import numpy as np
import image_resize

img = cv2.imread('images/659175809_008.tif',0)
img = image_resize.resize(img,width=800)

edges = cv2.Canny(img,100,200)

cv2.imshow('mat',edges)
cv2.waitKey(0)

