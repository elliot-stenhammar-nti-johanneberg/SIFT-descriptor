import sys
from PIL import Image, ImageFilter, ImageChops
import cv2 
import numpy as np
import math

# np.set_printoptions(threshold=sys.maxsize)

# # Grayscale Image
# image = processImage('digits.jpg')

# # Edge Detection Kernel
kernel1 = np.array(
    [[-1, -1, -1],
     [-1, 8, -1],
     [-1, -1, -1]])


kernel2 = np.array([-1, 0, 1])
kernel3 = np.array([
    [-1],    
    [0],
    [1]
])

def resizeImage(img, width, height):
    resized_img = cv2.resize(img, [width, height])
    return resized_img

def magnitude(imgX, imgY):
    sum = 0
    for i in range(128):
        for j in range(128):
            mag = math.sqrt((imgX[i][j]**2 + imgY[i][j]**2)/2)
            ang = math.atan(imgX[i][j]/imgY[i][j])
            print(imgX[i][j])
            print(imgY[i][j])
            print(ang)
    return sum

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized_img = resizeImage(img, 128, 128)
filtered_imgX = cv2.filter2D(resized_img, -1, kernel2)
filtered_imgY = cv2.filter2D(resized_img, -1, kernel3)
mag = magnitude(filtered_imgX, filtered_imgY)
print(mag)
# cv2.imshow('test', filtered_imgX)
# print(filtered_imgX)
# cv2.imshow('imageX', filtered_imgX)
# cv2.imshow('imageY', filtered_imgY)
# img = magnitude(filtered_imgX, filtered_imgY)
# cv2.imshow('imageMag', img)
cv2.waitKey(0) 
cv2.destroyAllWindows() 