import cv2
import image_resize


def captch_ex(file_name):
    img = cv2.imread(file_name)
    img = image_resize.resize(img, width=600)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)

    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)

    cv2.imshow('img2gray',img2gray)
    cv2.imshow('mask',mask)
    cv2.imshow('imgfinal',image_final)
    cv2.waitKey(0)

    ret, new_img = cv2.threshold(image_final, 127, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    cv2.imshow('newimg',new_img)
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=5)  # dilate , more the iteration more the dilation

    im, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 15 and h < 15:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    cv2.imshow('result', img)
    cv2.waitKey()


file_name = 'images/a6.jpg'
captch_ex(file_name)