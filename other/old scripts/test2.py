import sys
import cv2 
import numpy as np
import math
# np.set_printoptions(threshold=sys.maxsize)

def formatImage(img):
    img = cv2.resize(img, [27, 27])
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(img, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh

def cropImage(img):
    all_cropped_imgs = []
    for i in range(3):
        cropped_imgs = []
        for j in range(3):
            cropped_img = img[i*9:(i+1)*9, j*9:(j+1)*9]
            cropped_imgs.append(cropped_img)
        all_cropped_imgs.append(cropped_imgs)
    return all_cropped_imgs

def func_HoG(imgs):
    all_histograms = []
    for row in imgs:
        histograms = []
        for img in row:
            mag_array = []
            ang_array = []
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    mag = math.sqrt((gx[i][j]**2 + gy[i][j]**2)/2)
                    mag_array.append(mag)
                    ang = math.degrees(math.atan2(gy[i][j], gx[i][j]))
                    ang_array.append(ang)
            histogram = np.zeros(8)
            for i in range(len(mag_array)):
                for j in range(8):
                    if (ang_array[i] < 180-45*(j) > 180-45*(j+1)):
                        histogram[j] += mag_array[i]
            histogram = [round(num) for num in histogram]
            histograms.append(histogram)
        all_histograms.append(histograms)
    return all_histograms

def compileDataset():
    dataset = []
    for i in range(10):
        img = cv2.imread("dataset/%s.jpg" %(i), cv2.IMREAD_GRAYSCALE)
        img = formatImage(img)
        imgs = cropImage(img)
        HoG = func_HoG(imgs)
        dataset.append(HoG)
    return dataset

def compare(dataset, datas):
    all_differences = []
    for i in range(10):
        differences = []
        for j in range(3):
            for k in range(3): 
                for h in range(8):
                    differences.append(dataset[i][j][k][h] - datas[j][k][h])
        sum_of_differences = abs(sum(differences))
        all_differences.append(sum_of_differences)
    min_diff = min(all_differences)
    index = all_differences.index(min_diff)
    return index




    # for number in dataset:
    #     differences = []
    #     for i in range(8):
    #         
    #     
    # sum_of_differences = []
    # for number in all_differences:
    #     sum_of_differences.append(abs(sum(number)))
    # min_diff = min(sum_of_differences)
    # index = sum_of_differences.index(min_diff)
    # return index

dataset = compileDataset()
img = cv2.imread("test-img/7.jpg", cv2.IMREAD_GRAYSCALE)
img = formatImage(img)
imgs = cropImage(img)
HoG = func_HoG(imgs)
# cv2.imshow("img", imgs[2][2])
# HoG = [round(num) for num in HoG]
print(compare(dataset, HoG))
# print(HoG)

cv2.waitKey(0)