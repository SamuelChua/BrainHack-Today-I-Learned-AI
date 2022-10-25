from PIL import Image
import numpy as np
from numpy import asarray
import sys
np.set_printoptions(threshold=sys.maxsize)
import cv2
img = cv2.imread(r"C:\Users\intern\Downloads\test_5cm.png")
kernel = np.ones((5,5),np.uint8)
img_dilation = cv2.dilate (img, kernel, iterations = 1)
img_dilation[img_dilation ==0] = 1
# cv2.imshow('orig',img)
# cv2.imshow('dilate', img_dilation)
# print(img_dilation)
# cv2.imshow('hello', img_dilation[:,:,1])
# cv2.waitKey(0)
print( img_dilation[:,:,1])
print('yes')
