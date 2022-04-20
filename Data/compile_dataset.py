import cv2
import numpy as np
import sys
import csv
import os

np.set_printoptions(threshold=sys.maxsize)

folder = 'Extracted-digits'
all_pxs = []
for img in os.listdir(folder):
    img = cv2.imread(folder + "/" + img, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("img", img)
    label = cv2.waitKey(0) - 48
    print(label)
    cv2.destroyAllWindows()
    img = cv2.resize(img, [28, 28])
    pxs = [label]
    for row in img:
        for px in row:
            pxs.append(px)
    all_pxs.append(pxs)

with open('Datasets/digits.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter = ' ')
    for pxs in all_pxs:
        writer.writerow(pxs)

# with open('digits.csv', 'r') as file:
#     reader = csv.reader(file)
#     headers = next(reader) 
#     data = list(reader)
#     # data = np.array(data).astype(float) 
#     print(data)