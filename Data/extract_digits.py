import cv2
import numpy as np
import sys
import os

np.set_printoptions(threshold=sys.maxsize)
        
# Takes an unprocessed (grayscaled) image and removes noise + enlarges contrast between black and white 
def preProcessImage(img):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(img, blur, scale=255)
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thresh)
    return invert

# Takes an processed image and applies a built in function for finding potential targets. Rejects targets if size is unreasonably
# small. 
def findContours(img):
    global ROI_number
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # ROI = ProcessedImg[y:y+h, x:x+w]
        ROI = ResizedImg[y:y+h, x:x+w]
        if ROI.shape[0] > 50: 
            cv2.imwrite('Extracted-digits/ROI_{}.png'.format(ROI_number), ROI)
            ROI_number += 1
            cv2.rectangle(ProcessedImg, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
# Writes found numbers to folder with individual names.
def main():
    folder = "Original-images"
    global ROI_number
    ROI_number = 0 
    for img in os.listdir(folder):
        OriginalImg = cv2.imread(folder + "/" + img, cv2.IMREAD_GRAYSCALE)
        global ResizedImg
        ResizedImg = cv2.resize(OriginalImg, [750, 750])
        global ProcessedImg
        ProcessedImg = preProcessImage(ResizedImg)
        findContours(ProcessedImg)
    return

main()