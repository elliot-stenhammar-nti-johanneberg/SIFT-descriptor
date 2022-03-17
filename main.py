import sys
from PIL import Image, ImageFilter, ImageChops
import cv2 
import numpy as np
import math
import matplotlib.pyplot as plt

data = [
    [207674, 178796, 155492, 130223, 90104, 58580, 39518, 24251],
    [181269, 141572, 124869, 107158, 73994, 47698, 38807, 28525],
    [267762, 231494, 202646, 165533, 119703, 97066, 76227, 45689],
    [283240, 245393, 212230, 178322, 135736, 105776, 76944, 44211],
    [243729, 212047, 201579, 186986, 145641, 82097, 50160, 29958],
    [359962, 316004, 277685, 233186, 193122, 145677, 99704, 47396],
    [365412, 319084, 285465, 250468, 194048, 144370, 104267, 60479],
    [272363, 242673, 219540, 195228, 142245, 117995, 91019, 60983],
    [204017, 186861, 168491, 147257, 112809, 77542, 45866, 20633],
    [259321, 229608, 213072, 191952, 153332, 113717, 85885, 48794]
]

# np.set_printoptions(threshold=sys.maxsize)

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
        

def compare(dataset, data):
    # COMAPRE INDIVIDUAL ANGEL VALUES AND ESTIMATE MOST LIKELY NUMBER
    all_differences = []
    for number in dataset:
        differences = []
        for i in range(8):
            differences.append(number[i] - data[i])
        all_differences.append(differences)
    print(all_differences)
    sum_of_differences = []
    for number in all_differences:
        sum_of_differences.append(abs(sum(number)))
    min_diff = min(sum_of_differences)
    index = sum_of_differences.index(min_diff)
    return index

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

img = cv2.imread("dataset/0.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = resizeImage(img, 128, 128)
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
mag, ang = mag_ang(gx, gy, img)
# mag = np.reshape(mag, (-1, img.shape[0]))
# ang = np.reshape(ang, (-1, img.shape[0]))
histogram = histogram(mag, ang)
# histogram = [round(num) for num in histogram]
p = convert_to_p_histogram(histogram)
data_p = convert_to_p(data)
print(p)
print(data_p[6])
# print (compare(data_p, p))
# print(histogram)

cv2.waitKey(0) 
cv2.destroyAllWindows() 