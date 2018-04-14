import cv2
import numpy as np
import imutils

# Params
maxArea = 150
minArea = 10

# Read image
img = cv2.imread('images/659175809_008.tif')
img = imutils.resize(img, width=800)

# Convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)

# Keep only small components but not to small
comp = cv2.connectedComponentsWithStats(thresh)

labels = comp[1]
labelStats = comp[2]
labelAreas = labelStats[:,4]

for compLabel in range(1,comp[0],1):

    if labelAreas[compLabel] > maxArea or labelAreas[compLabel] < minArea:
        labels[labels==compLabel] = 0

labels[labels>0] =  1

# Do dilation
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,5))
dilateText = cv2.morphologyEx(labels.astype(np.uint8),cv2.MORPH_DILATE,se)

cv2.imshow('dilated', dilateText)
cv2.waitKey(0)

# Find connected component again
comp = cv2.connectedComponentsWithStats(dilateText)

# Draw a rectangle around the text
labels = comp[1]
labelStats = comp[2]
#labelAreas = labelStats[:,4]

for compLabel in range(1,comp[0],1):

    cv2.rectangle(img,(labelStats[compLabel,0],labelStats[compLabel,1]),
        (labelStats[compLabel,0]+labelStats[compLabel,2],
        labelStats[compLabel,1]+labelStats[compLabel,3]),(0,0,255),2)
    
cv2.imshow('final', img)
cv2.waitKey(0)