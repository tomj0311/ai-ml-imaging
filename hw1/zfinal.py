import cv2, numpy as np


def final_ex(file_name):

    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sharpen image
    # ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Some morphology to clean up image if the image is too dull
    mask = cv2.adaptiveThreshold(gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 2)

    kernel = np.ones((19,15), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    cv2.imshow('opening',opening)
    cv2.waitKey(0)
    
    kernel = np.ones((9,15), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # closing = cv2.bitwise_and(gray, gray, mask=mask)

    cv2.imshow('final', closing)
    cv2.waitKey(0)

    # Apply gaussian filter - further smoothens image - less noise
    # blur = cv2.GaussianBlur(closing, (5,5), 2)
    # binarise final 
    ret, new_img = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  

    cv2.imshow('thresh', new_img)
    cv2.waitKey(0)

    # find connected components
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 3))
    connected = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, morph_kernel)

    cv2.imshow('connected', connected)
    cv2.waitKey(0)

    # find contours
    imx, contours, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    index = 1
    # draw contours
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # # Don't plot small false positives tha1t aren't text
        if w < 15 and h < 15:
            continue

        # #Don't plot large false poistives that arent text
        # if w > 100:
        #   continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cropped = img[y :y +  h , x : x + w]
        cv2.imwrite('cropped/' + str(index) + '.jpg', cropped)
        index = index + 1

    # write original image with added contours to disk
    cv2.imshow('captcha_result', img)
    cv2.waitKey()


file_name = 'images/659175813_012.tif'
final_ex(file_name)