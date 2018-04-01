import os
from PIL import Image
import pytesseract 
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\tojose\\AppData\\Local\\Programs\\Python\\Python36-32\\lib\\site-packages\\pytesseract'
import cv2

image = cv2.imread('a6.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# try threshold to remove noise
# gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# try median blur to remove nose
gray = cv2.medianBlur(gray,3)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

img = Image.open(filename)

text = pytesseract.image_to_string(img)
os.remove(filename)
print(text)
 
cv2.imshow("Output", gray)
cv2.waitKey(0)