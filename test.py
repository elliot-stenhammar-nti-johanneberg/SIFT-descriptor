import cv2 
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

def preProcessImage(img):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(img, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thresh)
    resize = cv2.resize(invert, [28, 28])
    return resize

img = cv2.imread("Data/Extracted-digits/ROI_3.png", cv2.IMREAD_GRAYSCALE)
img = preProcessImage(img)
cv2.imshow("org-img", cv2.resize(img, [100, 100]))
cropped_imgs = []
for i in range(3):
    for j in range(3):
        cropped_img = img[i*9:(i+1)*9, j*9:(j+1)*9]
        cropped_imgs.append(cropped_img)

cv2.imshow("test", cv2.resize(cropped_imgs[6], [100, 100]))
cv2.waitKey(0)
# print(np.array(cropped_imgs).shape)