import sys
from PIL import Image, ImageFilter, ImageChops
import cv2 
import numpy as np
import math
import matplotlib.pyplot as plt

data = [
    [151381, 131118, 112397, 95366, 62286, 38901, 26237, 17764],
    [114252, 85383, 73598, 61808, 40951, 25074, 21829, 16995],
    [175475, 152671, 130438, 106264, 74168, 63950, 49921, 31874],
    [171422, 150242, 127600, 102844, 79556, 66173, 50050, 25800],
    [179458, 156449, 153101, 142428, 115090, 57157, 33719, 18827],
    [219268, 191713, 168524, 141429, 119334, 90873, 62233, 25060],
    [226590, 203699, 185442, 160059, 125246, 98949, 74223, 37123],
    [160700, 146614, 133306, 119122, 86746, 77325, 60687, 39761],
    [140371, 132350, 120869, 106243, 82472, 55024, 28463, 12730],
    [165364, 151144, 145065, 131688, 108617, 82296, 64218, 33435]
]

# np.set_printoptions(threshold=sys.maxsize)

def compare(dataset, data):
    all_differences = []
    for number in dataset:
        differences = []
        for i in range(8):
            differences.append(number[i] - data[i])
        all_differences.append(differences)
    sum_of_differences = []
    for number in all_differences:
        sum_of_differences.append(abs(sum(number)))
    min_diff = min(sum_of_differences)
    index = sum_of_differences.index(min_diff)
    return index

def convert_to_p(data):
    data_p = []
    for number in data:
        sum_of_array = sum(number)
        angs = []
        for ang in number:
            angs.append(ang/sum_of_array)
        data_p.append(angs)
    return data_p

def convert_to_p_histogram(data):
    data_p = []
    sum_of_array = sum(data)
    for ang in data:
        data_p.append(ang/sum_of_array)
    return data_p

def resizeImage(img, width, height):
    resized_img = cv2.resize(img, [width, height])
    return resized_img

def mag_ang(imgX, imgY, img):
    mag_array = []
    ang_array = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mag = math.sqrt((imgX[i][j]**2 + imgY[i][j]**2)/2)
            mag_array.append(mag)
            ang = math.degrees(math.atan2(imgY[i][j], imgX[i][j]))
            ang_array.append(ang)
    return mag_array, ang_array

def histogram(mag, ang):
    histogram = np.zeros(8)
    for i in range(len(mag)):
        for j in range(8):
            if (ang[i] < 180-45*(j) > 180-45*(j+1)):
                histogram[j] += mag[i]
    return histogram

img = cv2.imread("test-img/9.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("img_before", img)
img = cv2.GaussianBlur(img, (51,51), cv2.BORDER_DEFAULT)
# cv2.imshow("img", img)
img = resizeImage(img, 128, 128)
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
mag, ang = mag_ang(gx, gy, img)
histogram = histogram(mag, ang)
histogram = [round(num) for num in histogram]
print (compare(data, histogram))
# p = convert_to_p_histogram(histogram)
# data_p = convert_to_p(data)
# print (compare(data_p, p))
# print(histogram)
# print(histogram)

cv2.waitKey(0) 
cv2.destroyAllWindows() 