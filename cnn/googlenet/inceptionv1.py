#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/19
# @Author  : Chen

"""
implement inception v1 with keras
    Going deeper with convolutions
    https://arxiv.org/abs/1409.4842
"""

# import the necessary packages
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Dense, Reshape, Concatenate


class InceptionV1:
    def __init__(self):
        self.inpt = Input(shape=(224, 224, 3))
        self.inceptionv1 = self.build_network()

    def inception_module(self, x, filter1x1, filter3x3, filter5x5, filterpool):
        # branch1x1: conv1x1
        branch1x1 = Conv2D(filter1x1[0], (1, 1), strides=(1, 1), padding='same', activation='relu')(x)

        # branch3x3: conv1x1 -> conv3x3
        branch3x3 = Conv2D(filter3x3[0], (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        branch3x3 = Conv2D(filter3x3[1], (3, 3), strides=(1, 1), padding='same', activation='relu')(branch3x3)

        # branch5x5: conv1x1 -> conv5x5
        branch5x5 = Conv2D(filter5x5[0], (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        branch5x5 = Conv2D(filter5x5[1], (5, 5), strides=(1, 1), padding='same', activation='relu')(branch5x5)

        # branchpool: pool -> conv1x1
        branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branchpool = Conv2D(filterpool[0], (1, 1), strides=(1, 1), padding='same', activation='relu')(branchpool)
        x = Concatenate(axis=3)([branch1x1, branch3x3, branch5x5, branchpool])
        return x

    def build_network(self):
        inpt = self.inpt
        # conv, max pool
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inpt)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # conv, max pool
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # inception(3a), inception(3b)
        x = self.inception_module(x, filter1x1=(64,), filter3x3=(96, 128), filter5x5=(16, 32), filterpool=(32,))
        x = self.inception_module(x, filter1x1=(128,), filter3x3=(128, 192), filter5x5=(32, 96), filterpool=(64,))
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # inception 4a, 4b, 4c, 4d, 4e
        x = self.inception_module(x, filter1x1=(192,), filter3x3=(96, 208), filter5x5=(16, 48), filterpool=(64,))
        x = self.inception_module(x, filter1x1=(160,), filter3x3=(112, 224), filter5x5=(24, 64), filterpool=(64,))
        x = self.inception_module(x, filter1x1=(128,), filter3x3=(128, 256), filter5x5=(24, 64), filterpool=(64,))
        x = self.inception_module(x, filter1x1=(112,), filter3x3=(144, 288), filter5x5=(32, 64), filterpool=(64,))
        x = self.inception_module(x, filter1x1=(256,), filter3x3=(160, 320), filter5x5=(32, 128), filterpool=(128,))
        # max pool
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # inception5a, inception5b
        x = self.inception_module(x, filter1x1=(256, ), filter3x3=(160, 320), filter5x5=(32, 128), filterpool=(128,))
        x = self.inception_module(x, filter1x1=(384, ), filter3x3=(192, 384), filter5x5=(48, 128), filterpool=(128,))
        # average pooling
        x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
        # dropout
        x = Dropout(0.4)(x)
        x = Dense(1000, activation='relu')(x)
        x = Dense(1000, activation='softmax')(x)
        x = Reshape((1000,))(x)
        return x

    def get_layer(self):
        model = Model(inputs=self.inpt, outputs=self.inceptionv1)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        print('method 1:')
        model.summary()

        print('method 2:')
        for i in range(len(model.layers)):
            print(model.get_layer(index=i).output)

        print('method 3:')
        for layer in model.layers:
            print(layer.output_shape)

if __name__ == '__main__':
    inceptionv1 = InceptionV1()
    inceptionv1.get_layer()
