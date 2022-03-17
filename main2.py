import sys
from PIL import Image, ImageFilter, ImageChops
import cv2 
import numpy as np
import math
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

def resizeImage(img, width, height):
    resized_img = cv2.resize(img, [width, height])
    return resized_img

def HoG(imgX, imgY, img):
    mag_array = []
    ang_array = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mag = math.sqrt((imgX[i][j]**2 + imgY[i][j]**2)/2)
            mag_array.append(mag)
            ang = math.degrees(math.atan2(imgY[i][j], imgX[i][j]))
            ang_array.append(ang)
    histogram = np.zeros(8)
    for i in range(len(mag_array)):
        for j in range(8):
            if (ang_array[i] < 180-45*(j) > 180-45*(j+1)):
                histogram[j] += mag_array[i]
    return histogram

def scaleSpace(img):
    img1 = resizeImage(img, 256, 256)
    img2 = resizeImage(img, 128, 128)
    img3 = resizeImage(img, 64, 64)
    img4 = resizeImage(img, 32, 32)
    octaves = [img1, img2, img3, img4]
    all_blurred_imgs = []
    for img in octaves:
        blur = 1
        blurred_imgs = []
        for i in range(5):
            blurred_imgs.append(cv2.GaussianBlur(img, (blur, blur), 0))
            blur += 4
        all_blurred_imgs.append(blurred_imgs)
    return all_blurred_imgs

def DoG(scales):
    all_subtracted_scales = []
    for scale in scales:
        subtracted_scales = []
        for i in range(len(scale)-1):
            subtracted_scales.append(scale[i] - scale[i+1])
        all_subtracted_scales.append(subtracted_scales)
    return all_subtracted_scales

def localMinMax(DoG):
    all_extremas = []
    for octave in DoG:
        all_scale_extremas = []
        for s in range(1, len(octave)-1):
            extremas = []
            scale_active = octave[s]
            scale_up = octave[s-1]
            scale_down = octave[s+1]
            scales = [scale_active, scale_up, scale_down]
            for y in range(len(scale_active)):
                for x in range(len(scale_active[y])):
                    val = scale_active[y][x]
                    extrema = True
                    for scale in scales:
                        for i in range(y-1, y+1):
                            for j in range(x-1, x+1):
                                if scale[i][j] > val or scale[i][j] < val:
                                    extrema = False
                    if extrema:
                        extremas.append([y, x])
            all_scale_extremas.append(extremas)          
        all_extremas.append(all_scale_extremas)
    return all_extremas

def markKeypoints(img, keypoints):
    for point in keypoints:
        x = point[1]
        y = point[0]
        img[y][x] = 0
    return img

img = cv2.imread("test-imgs-old/test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = resizeImage(img, 256, 256)
scales = scaleSpace(img)
dogs = DoG(scales)
cv2.imshow("img_1", dogs[0][0])
cv2.imshow("img_2", dogs[0][3])
# cv2.imshow("img_3", dogs[1][1])
# extremas = localMinMax(dogs)
# img_test = markKeypoints(img, extremas[0][0])
# cv2.imshow("img_test", img_test)
# gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
# gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
# histogram = HoG(gx, gy, img)
# histogram = [round(num) for num in histogram]
# print(histogram)

cv2.waitKey(0) 
cv2.destroyAllWindows() 