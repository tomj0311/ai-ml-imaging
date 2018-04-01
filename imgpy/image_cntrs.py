import cv2, sys, math, numpy as np
import imutils
import image_resize as ir

def draw_square_cntrs(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cv2.imshow('tresh', thresh)
    cv2.waitKey(0)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("final", img)
    cv2.waitKey(0)

#img = cv2.imread('718910447_001.tif')
img = cv2.imread('x1.jpg')
im = ir.resize(img, height=800)
draw_square_cntrs(im)
