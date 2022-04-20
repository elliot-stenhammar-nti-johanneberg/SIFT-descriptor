import sys
import cv2 
import numpy as np
import math
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv

np.set_printoptions(threshold=sys.maxsize)

targets = []
data = []
with open('Data/digits.csv', 'r') as file:
    reader = csv.reader(file)
    for row in file:
        targets.append(int(row[0]))
        row = row.split(" ")
        pxs = []
        for px in row[1:]:
            pxs.append(int(px))
        data.append(pxs)

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=109)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))




# py -m pip install [packagename]