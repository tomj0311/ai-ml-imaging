import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('718910447_001.tif',0)
edges = cv2.Canny(img,100,200)

cv2.imshow('mat',edges)
cv2.waitKey(0)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()