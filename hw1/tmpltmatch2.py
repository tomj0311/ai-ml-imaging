import cv2, numpy as np, imutils as im
import glob
import skewCorrection as sc
from matplotlib import pyplot

def matchTemplate(image, template):

  templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  corrected = sc.correctSkew(image)
  imageGray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

  w,h = templateGray.shape[::-1]
  imgw, imgh = imageGray.shape[::-1]

  if imgw < w or imgh < h:
    return None

  res = cv2.matchTemplate(imageGray,templateGray,cv2.TM_CCOEFF_NORMED)

  threshold = 0.6

  loc = np.where( res > threshold)

  cropped = None

  index = 1
  matches = []

  for pt in zip(*loc[::-1]):
    x, y = pt[0], pt[1]
    cropped = corrected[y :y +  h , x : x + w]

    matches.append(cropped)

  return matches

image = cv2.imread('images/Untitled_01_of_24.jpg')

pyplot.imshow(sc.correctSkew(image))
pyplot.show()


# matchs = matchTemplate(image, cv2.imread('images/t4.jpg'))

# index = 1
# for match in matchs:
  
#   cv2.imshow(str(index), match)
#   cv2.waitKey(0)

#   index = index + 1







