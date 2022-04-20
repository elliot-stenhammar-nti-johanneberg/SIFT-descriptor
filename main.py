import cv2 
import numpy as np
import math
import csv
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

targets = []
data = []
dataset = 'Data/Datasets/digits-hog.csv'
# dataset = 'Data/Datasets/digits.csv'
with open(dataset, 'r') as file:
    reader = csv.reader(file)
    for row in file:
        targets.append(int(row[0]))
        row = row.split(" ")
        pxs = []
        for px in row[1:]:
            pxs.append(int(px))
        data.append(pxs)

clf = svm.SVC(kernel='linear')
clf.fit(data, targets)

def preProcessImage(img):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(img, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thresh)
    return invert

def HoG(img):
    cropped_imgs = []
    for i in range(3):
        for j in range(3):
            cropped_img = img[i*9:(i+1)*9, j*9:(j+1)*9]
            cropped_imgs.append(cropped_img)
    
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

def main():
        OriginalImg = cv2.imread("Test-imgs/test2.jpg", cv2.IMREAD_GRAYSCALE)
        ResizedImg = cv2.resize(OriginalImg, [750, 750])
        ProcessedImg = preProcessImage(ResizedImg)

        cnts = cv2.findContours(ProcessedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            # num = cv2.resize(ProcessedImg[y:y+h, x:x+w], [28, 28]).flatten()
            # num = cv2.resize(ResizedImg[y:y+h, x:x+w], [28, 28]).flatten()
            num_HoG = HoG(cv2.resize(ProcessedImg[y:y+h, x:x+w], [28, 28]))
            pred = clf.predict(num_HoG.flatten().reshape(1, -1))
            # pred = clf.predict(num.reshape(1, -1))
            cv2.rectangle(ResizedImg, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(ResizedImg, format(pred[0]), (x+int(w/3), y-5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("test", ResizedImg)

main()
cv2.waitKey(0)
