import numpy as np
import cv2, sys

samples = np.loadtxt('alphabets.data', np.float32)
labels = np.loadtxt('alphabetlabels.data', np.float32)
labels = labels.reshape((labels.size))
knn = cv2.ml.KNearest_create()
knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

img = cv2.imread('images/a1.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('thres', thresh)
cv2.waitKey(0)

imx, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    
  [x, y, w, h] = cv2.boundingRect(cnt)

  if w > 300 or h > 300:
    continue

  if w < 5 or h < 5:
    continue
    
  roi = thresh[y:y+h,x:x+w]
  roismall = cv2.resize(roi,(20, 20))

  cv2.imshow('roi', img[y:y+h,x:x+w])
  cv2.waitKey(0)

  roismall = roismall.reshape((1, 400))
  roismall = np.float32(roismall)
  retval, result, neighbours, dist = knn.findNearest(roismall, k = 1)

  match = result == labels
  correct = np.count_nonzero(match)
  accuracy = correct * 100 / result.size

  string = chr(int((result[0][0])))
  
  print(string + " accuracy " + str(accuracy) )
  # cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
