import sys
import cv2, numpy as np
from matplotlib import pyplot as plt
import json, codecs

import image_resize 

img = cv2.imread(filename='images/IMG_20180331_180458.jpg')
img = image_resize.resize(image=img, width=600)

a = np.arange(15).reshape(3,5)
print(a.tolist())
print(a.shape())

# img = cv2.medianBlur(img,5)

# np_list = img.tolist()
# json.dump(np_list, codecs.open('file1.json', 'w', encoding='utf-8'), sort_keys=True, indent=4)

