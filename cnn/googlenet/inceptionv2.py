#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21
# @Author  : Chen

"""
implement Inception V2 with Keras
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    https://arxiv.org/abs/1502.03167
"""

# import the necessary packages
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate
from keras.layers import MaxPooling2D, AveragePooling2D


class InceptionV2:
    def __init__(self):
        self.inpt = Input(shape=(224, 224, 3))
        self.inceptionv2 = self.build_network()

    def conv2d_bn(self, x, filters, kernel_size, strides, padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x

    def inception_module(self, x, param, final=False, final_layer=False):
        if not final:
            branch1x1 = self.conv2d_bn(x, param[0][0], (1, 1), strides=(1, 1))
            branch3x3 = self.conv2d_bn(x, param[1][0], (1, 1), strides=(1, 1))
            branch3x3 = self.conv2d_bn(branch3x3, param[1][1], (3, 3), strides=(1, 1))
            branch5x5 = self.conv2d_bn(x, param[2][0], (1, 1), strides=(1, 1))
            branch5x5 = self.conv2d_bn(branch5x5, param[2][1], (3, 3), strides=(1, 1))
            branch5x5 = self.conv2d_bn(branch5x5, param[2][1], (3, 3), strides=(1, 1))
            if final_layer:
                branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
            else:
                branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, param[3][0], (1, 1), (1, 1))
            x = Concatenate(axis=3)([branch1x1, branch3x3, branch5x5, branch_pool])
        else:
            branch3x3 = self.conv2d_bn(x, param[1][0], (1, 1), strides=(1, 1))
            branch3x3 = self.conv2d_bn(branch3x3, param[1][1], (3, 3), strides=(2, 2))
            branch5x5 = self.conv2d_bn(x, param[2][0], (1, 1), strides=(1, 1))
            branch5x5 = self.conv2d_bn(branch5x5, param[2][1], (3, 3), strides=(1, 1))
            branch5x5 = self.conv2d_bn(branch5x5, param[2][1], (3, 3), strides=(2, 2))
            branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
            # branch_pool = self.conv2d_bn(branch_pool, (1, 1), strides=(1, 1))
            # 池化后没有使用1x1的卷积
            x = Concatenate(axis=3)([branch3x3, branch5x5, branch_pool])
        return x

    def build_network(self):
        inpt = self.inpt
        # conv -> max pooliing
        x = self.conv2d_bn(inpt, 64, (7, 7), (2, 2))
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        # conv -> max pooling
        x = self.conv2d_bn(x, 64, (1, 1), strides=(1, 1))
        x = self.conv2d_bn(x, 192, (3, 3), strides=(1, 1))
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        # inception3a, 3b, 3c: 28 x 28 x 256
        x = self.inception_module(x, param=[(64, ), (64, 64), (64, 96), (32, )])
        x = self.inception_module(x, param=[(64,), (64, 96), (64, 96), (64,)])
        x = self.inception_module(x, param=[(0,), (128, 160), (64, 96), (0,)], final=True)
        # inception4a, 4b, 4c, 4d, 4e: 14 x 14 x 1024
        x = self.inception_module(x, param=[(224,), (64, 96), (96, 128), (128,)])
        x = self.inception_module(x, param=[(192,), (96, 128), (96, 128), (128,)])
        x = self.inception_module(x, param=[(160,), (128, 160), (128, 160), (96,)])
        x = self.inception_module(x, param=[(96,), (128, 192), (160, 192), (96,)])
        x = self.inception_module(x, param=[(0,), (128, 192), (192, 256), (0,)], final=True)
        # inception5a, 5b: 7x7x1024
        x = self.inception_module(x, param=[(352,), (192, 320), (160, 224), (128,)])
        x = self.inception_module(x, param=[(352,), (192, 320), (192, 224), (128,)], final_layer=True)
        # average pooling
        x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)
        return x

    def get_layer(self):
        model = Model(inputs=self.inpt, outputs=self.inceptionv2)
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
    inceptionv1 = InceptionV2()
    inceptionv1.get_layer()
