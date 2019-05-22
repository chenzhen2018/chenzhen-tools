#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/22
# @Author  : Chen

"""
Classification using Logistic Regression
"""

# import the necessary packages
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# load digits dataset and split it into training/testing sets
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
print(X_train.shape, X_test.shape)

# create logistic regression object
clf = LogisticRegression(penalty='l2')

# train the model using the training sets
clf.fit(X_train, y_train)

# make predictions using the testing sets
y_test_pred = clf.predict(X_test)

print(classification_report(y_test, y_test_pred))

