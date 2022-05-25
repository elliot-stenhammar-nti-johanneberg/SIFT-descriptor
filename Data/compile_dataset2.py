# Compile dataset of hog:s. Currently used in main.py.

import cv2
import numpy as np
import math
import sys
import csv
import os

np.set_printoptions(threshold=sys.maxsize)

# Takes an unprocessed (grayscaled) image and removes noise + enlarges contrast between black and white 
def preProcessImg(img):
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(img, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    resize = cv2.resize(img, [28, 28])
    return resize

# Takes a processed image, splits it in 9 parts and creates a histogram of gradients for each. Returns array of histograms.
def HoG(img):
    cropped_imgs = []
    # Crops image into 9 parts
    for i in range(3):
        for j in range(3):
            cropped_img = img[i*9:(i+1)*9, j*9:(j+1)*9]
            cropped_imgs.append(cropped_img)
    # For each part create a HoG for that image
    histograms = []
    for img in cropped_imgs:
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
    return np.array(histograms)

# Opens each image in folder of saved numbers and allows user to label them before saving data to .csv file as a dataset. 
folder = 'Extracted-digits'
all_ors = []
for img in os.listdir(folder):
    img = cv2.imread(folder + "/" + img, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", img)
    label = cv2.waitKey(0) - 48
    print(label)
    cv2.destroyAllWindows()
    img = preProcessImg(img)
    ors = [label] 
    ors = np.append(ors, HoG(img).flatten())
    all_ors.append(ors)

with open('datasets/digits-hog.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter = ' ')
    for ors in all_ors:
        writer.writerow(ors)

cv2.waitKey(0)