import numpy as np, imutils
import cv2
import image_ptform

def templateMatch(image, template):

	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	#Perspective transform if the image is tilted
	blured = cv2.GaussianBlur(imageGray, (3,3), 0)
	edged = cv2.Canny(blured, 0, 50)

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


	w,h = templateGray.shape[::-1]

	res = cv2.matchTemplate(warped,templateGray,cv2.TM_CCOEFF_NORMED)

	threshold = 0.5

	loc = np.where( res >= threshold)

	for pt in zip(*loc[::-1]):
		x, y = pt[0], pt[1]
		cropped = warped[y :y +  h , x : x + w]
		break
	
	return cropped

	
image = cv2.imread('images/Untitled_01_of_24.jpg')
template1 = cv2.imread('images/t1.jpg')
# template2 = cv2.imread('images/t2.jpg')

matched = templateMatch(image, template1)

cv2.imshow('mathced', matched)
cv2.waitKey(0)
