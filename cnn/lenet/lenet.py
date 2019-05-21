#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/18
# @Author  : Chen

"""
implement LeNet with Keras
    Gradient-based learning applied to document recognition: https://ieeexplore.ieee.org/document/726791
"""
# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class LeNet:
    def __init__(self):
        self.net = self.build_net()

    def build_net(self):
        net = Sequential()
        net.add(Conv2D(6,
                       (5, 5),
                       strides=(1, 1),
                       input_shape=(28, 28, 1),
                       padding='valid'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Conv2D(16,
                       (5, 5),
                       strides=(1, 1),
                       padding='valid',
                       activation='relu',
                       kernel_initializer='uniform'))
        net.add(MaxPooling2D(pool_size=(2, 2)))
        net.add(Flatten())
        net.add(Dense(120, activation='relu'))
        net.add(Dense(84, activation='relu'))
        net.add(Dense(10, activation='softmax'))
        return net



