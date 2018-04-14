import cv2
import numpy as np
import imutils

image = cv2.imread('images/659175809_008.tif')
image = imutils.resize(image, width=800)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,15), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)

cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

im2, ctrs, hirearchy = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)

cv2.imshow('marked areas',image)
cv2.waitKey(0)