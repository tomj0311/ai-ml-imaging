import cv2, numpy as np, imutils as im
import pytesseract

def get_contour_precedence(contour, cols):
    tolerance_factor = 20
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def final_ex(file_name):

    img = cv2.imread(file_name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, mask = cv2.threshold(gray.copy(), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # mask = cv2.adaptiveThreshold(gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 111, 2)

    image_sharpened = cv2.bitwise_and(gray, gray, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 12))
    morphed = cv2.morphologyEx(image_sharpened, cv2.MORPH_OPEN, kernel)

    cv2.imshow('opening', morphed)
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('closing', morphed)
    cv2.waitKey(0)

    # binarise final 
    ret, thresh = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  

    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

    # find connected components
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 6))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

    cv2.imshow('connected', connected)
    cv2.waitKey(0)

    # find contours
    imx, contours, hierarchy = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  

    contours.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))

    index = 1

    for idx in range(0, len(hierarchy[0])):
        rect = x, y, w, h = cv2.boundingRect(contours[idx])

        if w  < 7:
            continue
        
        if h < 8:
            continue

        # Don't plot small false positives that aren't text
            
        cropped = img[y :y +  h , x : x + w]
        # cv2.imwrite('cropped/' + str(index) + '.png', cropped)
            
        # text = pytesseract.image_to_string(cropped, lang = 'eng')
        # print(text)

        image_with_boxes = cv2.rectangle(img, (x, y+h), (x+w, y), (0,255,0), 2)
            
        index = index + 1

    image_with_boxes = im.resize(image_with_boxes, width=800)

    cv2.imshow('final', image_with_boxes)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

# file_name = 'images/659180599_002.tif'
# final_ex(file_name)
# file_name = 'images/659180602_005.tif'
# final_ex(file_name)
# file_name = 'images/659175810_009.tif'
# final_ex(file_name)
# file_name = 'images/659180598_001.tif'
# final_ex(file_name)
# file_name = 'images/659180603_006.tif'
# final_ex(file_name)
# file_name = 'images/a6.jpg'
# final_ex(file_name)
# file_name = 'images/659175803_002.tif'
# final_ex(file_name)
file_name = 'images/XsSOP.jpg'
final_ex(file_name)

