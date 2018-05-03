import sys
import cv2, numpy as np
import imutils
from imutils import contours
from matplotlib import pyplot as plt
import pytesseract

import image_resize 

def greater(a, b):
  momA = cv2.moments(a)        
  (xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

  momB = cv2.moments(b)        
  (xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])
  
  if xa>xb:
    return 1

  if xa == xb:
    return 0
  else:
    return -1

img = cv2.imread(filename='images/a6.jpg')
orig = img.copy()
#img = image_resize.resize(image=img, width=600)

imgx = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
imgx = cv2.GaussianBlur(imgx, (5,5), 2)

kernel = np.ones((3,7), np.uint8) #cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(imgx, cv2.MORPH_OPEN, kernel)

_, thresh = cv2.threshold(grad, 127, 255, 
  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU )

imx, cntrs, hierarchy = cv2.findContours(thresh.copy(), 
  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for method in ("top-to-bottom"):
#     (cntrs, boundingBoxes) = contours.sort_contours(cntrs, method=method)

indx = 1
for cntr in cntrs:
  [x, y, w, h] = cv2.boundingRect(cntr)

  if w * h > 350:
    cropped = img[y :y +  h , x : x + w]
    cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)

    FIND MOMENTS AND CENTROID OF IMAGE AND STORE IN SUCH A WAY 
    #THAT IT IS EASY TO SOR

    cv2.imwrite('cropped/' + str(indx) + '.jpg', cropped)
    cv2.rectangle(img, (x, y), (x + w, y + h), 
        (255, 0, 255), 2)
      
    #text = pytesseract.image_to_string(cropped, lang='eng', config='letters')
    #print(text)
    indx = indx + 1

cv2.imwrite('final.jpg', img)