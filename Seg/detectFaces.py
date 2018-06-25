import cv2, numpy as np
import imutils
import sys
import image_resize as ir

def detectFace(image, cascade):
    img_copy = np.copy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cascade)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5 )

    if (len(faces) == 0):
        return None, None
    else:
        print ("Faces found ", len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img_copy

def pre_process(image):
    equalized = eh.eqhist(image)
    return equalized

def main():
    pass

if __name__ == "__main__":

    hcasc = "haarcascade_frontalface_default.xml"

    image = "./test_images/t2.jpg"  # sys.argv[1]
    
    img = cv2.imread(image)
    img = ir.resize(img, width=1200)

    detected = detectFace(img, hcasc)

    cv2.imshow('detected', detected)
    cv2.waitKey(0)

