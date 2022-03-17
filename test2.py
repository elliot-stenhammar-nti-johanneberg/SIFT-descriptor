import sys
from PIL import Image, ImageFilter, ImageChops
import cv2 
import numpy as np
import math

np.set_printoptions(threshold=sys.maxsize)

def resizeImage(img, width, height):
    resized_img = cv2.resize(img, [width, height])
    return resized_img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    cv2.imshow("gx", gx)
    cv2.imshow("gy", gy)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    mag, ang = cv2.cartToPolar(gx, gy)
    # bin_n = 16 # Number of bins
    # bin = np.int32(bin_n*ang/(2*np.pi))

    return mag


img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resized_img = resizeImage(img, 128, 128)

print(hog(img))