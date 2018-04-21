import numpy as np
import cv2, sys

samples = np.loadtxt('alphabets.data', np.float32)
responses = np.loadtxt('alphabetlabels.data', np.float32)
responses = responses.reshape((responses.size))
knn = cv2.ml.KNearest_create()
knn.train(samples, cv2.ml.ROW_SAMPLE, responses)

img = cv2.imread('images/d1.png')
out = np.zeros(img.shape, np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

imx, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    
  [x, y, w, h] = cv2.boundingRect(cnt)

  if w > 300 or h > 300:
    continue

  if w < 5 or h < 5:
    continue
    
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
  roi = thresh[y:y+h,x:x+w]
  roismall = cv2.resize(roi,(20, 20))

  cv2.imshow('roismall', roismall)
  cv2.waitKey(0)

  roismall = roismall.reshape((1, 400))
  roismall = np.float32(roismall)
  retval, results, neigh_resp, dists = knn.findNearest(roismall, k = 1)
  string = str(int((results[0][0])))
  
  print(string)
  # cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
