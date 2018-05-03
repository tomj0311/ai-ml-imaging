import sys
import cv2, numpy as np, imutils as im

import image_resize 
import skewCorrection as sc

img = cv2.imread('images/a1.png')
# img = im.resize(img, width=1200)
imgCopy = img.copy()

# blur = cv2.GaussianBlur(img, (5,5), 0)
# ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# a more effective method saving more details
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)

morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 12))
connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

cv2.imshow('connected', connected)
cv2.waitKey(0)

imx, contours,hierarchy = cv2.findContours(connected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('img', gray)
cv2.waitKey(0)

samples =  np.empty((0,400))
labels = []

for cnt in contours:

  [x, y, w, h] = cv2.boundingRect(cnt)
  
  if w > 300 or h > 300:
    continue  

  if w < 5 or h < 5:
    continue

  cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

  roi = connected[y:y+h,x:x+w]
  roistd = cv2.resize(roi,(20, 20))

  cv2.imshow('norm', roistd)
  key = cv2.waitKey(0)

  if key == 27:  # (escape to quit)
    break
  else:
    labels.append(int(ord(chr(key))))
    sample = roistd.reshape((1, 400))
    samples = np.append(samples,sample, 0)

labels = np.array(labels, np.uint8)
labels = labels.reshape((labels.size, 1))
print("training complete")

np.savetxt('alphabets.data', samples)
np.savetxt('alphabetlabels.data', labels)




