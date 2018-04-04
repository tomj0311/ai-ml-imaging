import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('718910447_001.tif',0)
edges = cv2.Canny(img,100,200)

cv2.imshow('mat',edges)
cv2.waitKey(0)

