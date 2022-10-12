import numpy as np
from numpy.linalg import det, lstsq, norm
import cv2
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
from matplotlib import pyplot as plt



fname = './driver_dat/imgs/train/c0/img_100026.jpg'
img = cv2.imread(fname)
resize_img = cv2.resize(img, dsize=(299, 299), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

plt.imshow(img)
plt.show()
plt.imshow(resize_img)
plt.show()
