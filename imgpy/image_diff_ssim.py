from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import time
import image_resize 

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="first input image")
ap.add_argument("-s", "--second", required=True, help="second input image")

args = {'first': 'C:\\Conversion\\tallpdf.png', 'second': 'C:\\Conversion\\sumatra.png'} #vars(ap.parse_args())
#args = {'first': 'C:\\Conversion\\wordsumatra.png', 'second': 'C:\\Conversion\\wordtallpdf.png'}
#args = {'first': 'C:\\Users\\tojose\\Pictures\\capture22.png', 'second': 'C:\\Users\\tojose\\Pictures\\capture11.png'}

t0 = time.time()

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

resized_imageA = image_resize.resize(imageA, height=300)
resized_imageB = image_resize.resize(imageB, height=300)

grayA = cv2.cvtColor(resized_imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(resized_imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)

t1 = time.time()
total = t1 - t0

diff = (diff * 255).astype("uint8")

print("SSIM - DIFF : {}", 1 - score,3)
print("Time taken {}", total)

# run-commnet below lines to show the difference

thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(resized_imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(resized_imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("first", resized_imageA)
cv2.imshow("second", resized_imageB) 
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)


# usage -  py image_diff_ssim.py --first C:\Conversion\wordsumatra.png --second C:\Conversion\wordtallpdf1.png
