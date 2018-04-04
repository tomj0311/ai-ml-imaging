import sys

import cv2
import numpy as np
import image_resize

# Input is a color image
def get_contours(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the input image
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)

    # Find the contours in the above image
    im, contours, hierarchy = cv2.findContours(thresh, 2, 1)

    return contours

# old code with lots of contour points
# if __name__=='__main__':
#     img = cv2.imread('718910453_006.tif')
#     img = image_resize.resize(img, width=600)

#     # Iterate over the extracted contours
#     for contour in get_contours(img):
#         # Extract convex hull from the contour
#         hull = cv2.convexHull(contour, returnPoints=False)

#         # Extract convexity defects from the above hull
#         defects = cv2.convexityDefects(contour, hull)

#         if defects is None:
#             continue

#         # Draw lines and circles to show the defects
#         for i in range(defects.shape[0]):
#             start_defect, end_defect, far_defect, _ = defects[i,0]
#             start = tuple(contour[start_defect][0])
#             end = tuple(contour[end_defect][0])
#             far = tuple(contour[far_defect][0])
#             cv2.circle(img, far, 5, [128,0,0], -1)
#             cv2.drawContours(img, [contour], -1, (0,0,0), 3)

#     cv2.imshow('Convexity defects',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__=='__main__':
    img = cv2.imread('d1.jpg')
    img = image_resize.resize(img, width=600)

    for contour in get_contours(img):
        orig_contour = contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)

        if defects is None:
            continue

        for i in range(defects.shape[0]):
            start_defect, end_defect, far_defect, _ = defects[i,0]
            start = tuple(contour[start_defect][0])
            end = tuple(contour[end_defect][0])
            far = tuple(contour[far_defect][0])
            cv2.circle(img, far, 7, [255,0,0], -1)
            cv2.drawContours(img, [orig_contour], -1, (0,255,0), 3)

    cv2.imshow('Convexity defects',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()