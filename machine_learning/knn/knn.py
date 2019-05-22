#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/22
# @Author  : Chen

"""
Classification using kNN
"""

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load digits dataset and split it into training/testing sets
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)

# create KNN object
KNN = KNeighborsClassifier(n_neighbors=5)

# train KNN model using training sets
KNN.fit(X_train, y_train)

# make predictions using testing sets
y_test_pred = KNN.predict(X_test)

# show the result
print(classification_report(y_test, y_test_pred))
