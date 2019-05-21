#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/19
# @Author  : Chen

"""
implement vgg with Keras
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
"""

# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

class VGG:
    def __init__(self):
        self.vgg_16 = self.vgg16_network()

    def vgg16_network(self):
        model = Sequential()
        # layer 1: conv-conv-max_pool
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform', input_shape=(224, 224, 3)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer 2: conv-conv-max_pool
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer 3: conv-conv-conv-max_pool
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer 4: conv-conv-conv-max_pool
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # layer 5: conv-conv-conv-max_pool
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
        return model

    def get_layer(self):
        for i in range(len(self.vgg_16.layers)):
            print(self.vgg_16.get_layer(index=i).output)

if __name__ == '__main__':
    vgg16 = VGG()
    vgg16.get_layer()