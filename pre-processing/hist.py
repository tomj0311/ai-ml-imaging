import cv2, sys
from matplotlib import pyplot as plt

def showHist(img):
  channels = cv2.split(img)

  colors = ("b", "g", "r") 

  plt.xlim([0, 256])
  for(channel, c) in zip(channels, colors):
      histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
      plt.plot(histogram, color=c)

  plt.xlabel("Color value")
  plt.ylabel("Pixels")

  plt.show()

img = cv2.imread("images/XsSOP.jpg", cv2.IMREAD_GRAYSCALE)
showHist(img)
