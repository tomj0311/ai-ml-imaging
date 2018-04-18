import cv2, numpy as np, imutils
import image_resize
import image_ptform

image = cv2.imread('Untitled_01_of_24.jpg')
ratio = 1 #image.shape[0] / 500.0 reduced by half
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blured = cv2.GaussianBlur(gray, (3,3), 0)
edged = cv2.Canny(blured, 0, 50)

cv2.imshow('image', image)
cv2.imshow('edged', edged)
cv2.waitKey(0)

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		screenCnt = approx
		break
	else:
    		screenCnt = approx
 
warped = image_ptform.four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

cv2.imshow("warped", warped)
#cv2.imshow("image", image)
cv2.waitKey(0)

# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# warped = (warped > T).astype("uint8") * 255
 
