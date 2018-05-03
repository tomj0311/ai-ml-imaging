import cv2, numpy as np

img = cv2.imread("images/IMG_20180331_180458.jpg")

img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(img_y_cr_cb)

# Applying equalize Hist operation on Y channel.
y_eq = cv2.equalizeHist(y)

img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

cv2.imshow('original', img)
cv2.imshow('equalized',img_rgb_eq)
cv2.waitKey(0)

# equ = cv2.equalizeHist(img)

# #Contrast liomited adaptive histogram equalization
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)

# res = np.hstack((img,cl1)) #stacking images side-by-side

# cv2.imshow('res', res)
# cv2.waitKey(0)