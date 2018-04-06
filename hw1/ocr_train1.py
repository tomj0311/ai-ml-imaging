import sys
import cv2, numpy as np
from matplotlib import pyplot as plt

import image_resize 

def eqhist_clahe(img):
  
  equ = cv2.equalizeHist(img)
  # Contrast liomited adaptive histogram equalization
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl1 = clahe.apply(img)
  # res = np.hstack((img,cl1)) #stacking images side-by-side
  return cl1

img = cv2.imread(filename='images/IMG_20180331_180458.jpg')
img = image_resize.resize(image=img, width=600)

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
gray = eqhist_clahe(gray)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

edges = cv2.Canny(image=gray, threshold1=50, threshold2=150,apertureSize = 3)

cv2.imshow('l', laplacian)
cv2.imshow('sx', sobelx)
cv2.imshow('sy', sobely)
cv2.waitKey(0)

plt.imshow(gray)
plt.show()

# ret,thresh = cv2.thres hold(gray,100,255,cv2.THRESH_BINARY)
blur = cv2.GaussianBlur(gray,(3,3),2)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,75,10)

plt.imshow(thresh)
plt.show()

imx, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
  # get rectangle bounding contour
  [x, y, w, h] = cv2.boundingRect(contour)

  # Don't plot small false positives that aren't text
  if w < 20 and h < 20:
     continue

  if w > 30:
     continue

  # draw rectangle around contour on original image
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

plt.imshow(img)
plt.show()

sys.exit()

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
  if cv2.contourArea(cnt)>50:
    [x,y,w,h] = cv2.boundingRect(cnt)
    if  h>28:
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
      roi = thresh[y:y+h,x:x+w]
      roistd = cv2.resize(roi,(10,10))

      cv2.imshow('norm',img)
      key = cv2.waitKey(0)

      if key == 27:  # (escape to quit)
        sys.exit()
      elif key in keys:
        responses.append(int(chr(key)))
        sample = roistd.reshape((1,100))
        samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('samples.data',samples)
np.savetxt('responses.data',responses)




