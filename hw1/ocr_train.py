import sys
import cv2, numpy as np

import image_resize 

def eqhist_clahe(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  equ = cv2.equalizeHist(img)
  # Contrast liomited adaptive histogram equalization
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl1 = clahe.apply(img)
  # res = np.hstack((img,cl1)) #stacking images side-by-side
  return cl1

img = cv2.imread('images/IMG_20180331_180458.jpg')
img = image_resize.resize(img, width=600)

imgx = eqhist_clahe(img)

cv2.imshow('eahist', imgx)
cv2.waitKey(0)

# gray = cv2.cvtColor(imgx, cv2.COLOR_BGR2GRAY) Histogram equalized image is alredy gray
blur = cv2.GaussianBlur(imgx,(1,1),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)

imx, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
  # get rectangle bounding contour
  [x, y, w, h] = cv2.boundingRect(contour)

  # Don't plot small false positives that aren't text
  if w < 20 and h < 20:
    continue

  if w > 45 or h > 45:
    continue

  # draw rectangle around contour on original image
  cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

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




