import cv2
import numpy as np
import image_resize

maxArea = 150
minArea = 10

img = cv2.imread('images/XsSOP.jpg')
img = image_resize.resize(img, height=600)

gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

comp = cv2.connectedComponentsWithStats(thresh)

labels = comp[1]
labelStats = comp[2]
labelAreas = labelStats[:,4]

for compLabel in range(1,comp[0],1):

    if labelAreas[compLabel] > maxArea or labelAreas[compLabel] < minArea:
        labels[labels==compLabel] = 0

labels[labels>0] =  1

se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
dilated = cv2.morphologyEx(comp[1].astype(np.uint8),cv2.MORPH_DILATE,se)

comp = cv2.connectedComponentsWithStats(dilated)

labels = comp[1]
labelStats = comp[2]

for compLabel in range(1,comp[0],1):

    cv2.rectangle(img,(labelStats[compLabel,0],labelStats[compLabel,1]),(labelStats[compLabel,0]+labelStats[compLabel,2],labelStats[compLabel,1]+labelStats[compLabel,3]),(0,0,255),2)

cv2.imshow('comp', img)
cv2.waitKey(0)