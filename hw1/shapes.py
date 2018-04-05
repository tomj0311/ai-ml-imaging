import cv2, numpy as np

img = cv2.imread('images/XsSOP.jpg',0)

size = np.size(img)
print(size)

skel = np.zeros(img.shape,np.uint8)

cv2.imshow("img",img)
cv2.waitKey(0)
 
ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
 
while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
 
    zeros = size - cv2.countNonZero(img)
    if zeros == size:
        done = True

thinned = skel
im, cntrs, hirearchy = cv2.findContours(thinned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for contour in cntrs:
  [x, y, w, h] = cv2.boundingRect(contour)

# Don't plot small false positives that aren't text
  if w < 15 and h < 15:
    continue

  cv2.rectangle(thinned, (x, y), (x + w, y + h), (255, 0, 255), 2)

cv2.imshow('win', thinned)
cv2.waitKey(0)

