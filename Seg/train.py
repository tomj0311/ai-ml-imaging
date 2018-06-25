import cv2
import numpy as np
import image_resize as ir

def detectFace(image):
    hcasc = "haarcascade_frontalface_default.xml"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(hcasc)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5 )

    if (len(faces) == 0):
        return None, None
    else:
        print ("Faces found ", len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)



def startTraining(folderPath):
    import os

    files = os.listdir(folderPath)

    faces = []
    labels = []

    for file in files:
        
        img = cv2.imread(file)
        img = ir.resize(img, width=1200)

        if file.startswith('a'):
            #do training for Anjali
        elif file.startswith('t')
            #do training for thomas
        else:
            #not implemented
    
def main():
    pass

if __name__ == "__main__":
    
    folder_path = "./train_images"

    startTraining(folder_path)
