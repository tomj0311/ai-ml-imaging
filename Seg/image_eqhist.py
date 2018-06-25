import cv2, numpy as np

def eqhist(image):
    #img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    y_eq = cv2.equalizeHist(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    return img_rgb_eq
