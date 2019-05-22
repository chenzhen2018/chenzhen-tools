#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/22
# @Author  : Chen

"""
classification using Descision Tree

"""

# import the necessary packages
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load digits dataset and split it into training/testing sets
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)

# create Descision Tree object
tree = DecisionTreeClassifier()

# train Descision Tree using training sets
tree.fit(X_train, y_train)

# make predictions using testing sets
y_test_pred = tree.predict(X_test)

# show the classification result
print(classification_report(y_test, y_test_pred))

