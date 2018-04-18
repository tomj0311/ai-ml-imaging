import numpy as np, imutils
import cv2
import image_ptform
 
def templateMatch(image, template):

	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	blured = cv2.GaussianBlur(imageGray, (3,3), 0)
	edged = cv2.Canny(blured, 0, 25)

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
	
	warped = image_ptform.four_point_transform(imageGray, screenCnt.reshape(4, 2) * 1)

	cv2.imshow('warped', warped)
	cv2.waitKey(0)

	h,w = templateGray.shape

	result = cv2.matchTemplate(warped,templateGray, cv2.TM_CCOEFF)

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	top_left = max_loc

	x,y = top_left[0], top_left[1]

	h,w = templateGray.shape
	
	bottom_right = (x + w, y + h)

	cropped = warped[y :y +  h , x : x + w]

	return cropped

image = cv2.imread('images/Untitled_17_of_24.jpg')
template1 = cv2.imread('images/l1.png')
template2 = cv2.imread('images/t2.jpg')

matched = templateMatch(image, template1)

cv2.imshow('matched', matched)
cv2.waitKey(0)
