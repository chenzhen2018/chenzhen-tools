#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/20
# @Author  : Chen

from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D


class Example:
    def __init__(self):
        self.inpt = Input(shape=(224, 224, 3))
        self.net = self.build_network()

    def build_network(self):
        inpt = self.inpt
        x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(inpt)
        ...
        x = Flatten()(x)
        x = Dense(1000)(x)
        return x

    def get_layer(self):
        model = Model(inputs=self.inpt, outputs=self.net)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        print("[INFO] Method 1...")
        model.summary()

        print("[INFO] Method 2...")
        for i in range(len(model.layers)):
            print(model.get_layer(index=i).output)

        print("[INFO] Method 3...")
        for layer in model.layers:
            print(layer.output_shape)


if __name__ == '__main__':
    ex = Example()
    ex.get_layer()