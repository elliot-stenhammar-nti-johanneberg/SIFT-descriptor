import sys
import cv2 
import numpy as np
import math

# np.set_printoptions(threshold=sys.maxsize)

dataset = [
    [[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0], [38817, 27396, 27035, 26780, 12716, 10871, 7055, 4700], [6595, 6595, 6595, 6595, 1546, 721, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0], [4706, 1442, 0, 0, 0, 0, 0, 0], [94906, 83527, 79350, 77955, 19685, 361, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0,
0, 0, 0, 0, 0, 0, 0]]],
    [185835, 170089, 168712, 166357, 30018, 13132, 12561, 10206],
    [581774, 566874, 451525, 346065, 278861, 263309, 152884, 44857],
    [171422, 150242, 127600, 102844, 79556, 66173, 50050, 25800],
    [179458, 156449, 153101, 142428, 115090, 57157, 33719, 18827],
    [219268, 191713, 168524, 141429, 119334, 90873, 62233, 25060],
    [226590, 203699, 185442, 160059, 125246, 98949, 74223, 37123],
    [160700, 146614, 133306, 119122, 86746, 77325, 60687, 39761],
    [140371, 132350, 120869, 106243, 82472, 55024, 28463, 12730],
    [165364, 151144, 145065, 131688, 108617, 82296, 64218, 33435]
]

def formatImage(img):
    img = cv2.resize(img, [256, 256])
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(img, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return thresh

def cropImage(img):
    all_cropped_imgs = []
    for i in range(4):
        cropped_imgs = []
        for j in range(4):
            cropped_img = img[i*64:(i+1)*64, j*64:(j+1)*64]
            cropped_imgs.append(cropped_img)
        all_cropped_imgs.append(cropped_imgs)
    return all_cropped_imgs

def HoG(imgs):
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

def compare(dataset, datas):

    for number in dataset:
        for row in number:
            for row_data in datas:    
                for histogram in row:
                    for data in row_data:
                        
                

    # all_differences = []
    # for number in dataset:
    #     differences = []
    #     for i in range(8):
    #         differences.append(number[i] - data[i])
    #     all_differences.append(differences)
    # sum_of_differences = []
    # for number in all_differences:
    #     sum_of_differences.append(abs(sum(number)))
    # min_diff = min(sum_of_differences)
    # index = sum_of_differences.index(min_diff)
    # return index

img = cv2.imread("test-img/1.jpg", cv2.IMREAD_GRAYSCALE)
img = formatImage(img)
imgs = cropImage(img)
HoG = HoG(imgs)
# cv2.imshow("img", imgs[2][2])
# HoG = [round(num) for num in HoG]
print(compare(dataset, HoG))
# print(HoG)

cv2.waitKey(0)