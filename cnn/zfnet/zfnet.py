#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/19
# @Author  : Chen

"""
implement ZF-Net with Keras
"""

# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


class ZFNet:
    def __init__(self):
        self.net = self.build_network()

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(96,
                         (7, 7),
                         strides=(2, 2),
                         input_shape=(224, 224, 3),
                         padding='valid',
                         activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(256,
                         (5, 5),
                         strides=(2, 2),
                         padding='same',
                         activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(384,
                         (3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(384,
                         (3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(256,
                         (3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
        return model

    def get_layer(self):
        for i in range(len(self.net.layers)):
            print(self.net.get_layer(index=i).output)
if __name__ == '__main__':
    zfnet = ZFNet()
    zfnet.get_layer()