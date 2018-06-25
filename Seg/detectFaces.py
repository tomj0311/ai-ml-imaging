import cv2
import sys

def detectFace(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    face_cascade = cv2.CascadeClassifier(cascade)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5 )

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def main():
    pass

if __name__ == "__main__":
    image = "./images/IMG_20180625_145128.jpg"  # sys.argv[1]

    img = cv2.imread(image)

    hcasc = "haarcascade_frontalface_default.xml"

    detected = detectFace(img, hcasc)

    cv2.imshow(detected)
    cv2.waitKey(0)

