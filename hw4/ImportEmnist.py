import numpy as np
import cv2
import cntk cnt
import urllib
import zipfile
import csv
import re
import gzip
import sys

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

def progress(count, blockSize, totalSize):
    percent = int(count * blockSize * 100 / totalSize)
    sys.stdout.write("\r%d%%" % percent + ' complete')
    sys.stdout.flush()

print("Start download EMNIST data sets")

