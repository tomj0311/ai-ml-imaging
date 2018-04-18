import cv2, numpy as np
import image_resize


def final_ex(file_name):
    img = cv2.imread(file_name)

    cv2.imshow('imgfinal', img)
    cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # mask = cv2.adaptiveThreshold(gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 2)

    image_final = cv2.bitwise_and(gray, gray, mask=mask)

    # cv2.imshow('imgfinal',image_final)
    # cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,12))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    morphed = cv2.morphologyEx(image_final, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('morphed', morphed)
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # for black text , cv.THRESH_BINARY_INV

    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)

    # cv2.imshow('connected', connected)
    # cv2.waitKey(0)

    im, contours, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # get contours

    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)

        if w < 10 or h < 10:
            continue

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    img = image_resize.resize(img, width=600)
    
    cv2.imshow('result', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


file_name = 'images/Untitled_01_of_24.jpg'
final_ex(file_name)

file_name = 'images/Untitled_09_of_24.jpg'
final_ex(file_name)

file_name = 'images/devex1.jpeg'
final_ex(file_name)
