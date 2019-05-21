#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21
# @Author  : Chen

"""
implement Inception V3 with Keras

"""

# import the necessary packages
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Concatenate

class InceptionV3:
    def __init__(self):
        self.inpt = Input(shape=(299, 299, 3))
        self.inceptionv3 = self.build_network()

    def conv2d_bn(self, x, filters, kernel_size, strides, padding='same', name=None):
        if name is not None:
            conv_name = name + '_conv'
            bn_name = name + '_bn'
        else:
            conv_name = None
            bn_name = None
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x

    def build_network(self):
        inpt = self.inpt
        # conv -> conv -> conv padded -> pool
        x = self.conv2d_bn(inpt, 32, (3, 3), (2, 2), padding='valid')
        x = self.conv2d_bn(x, 32, (3, 3), (1, 1), padding='valid')
        x = self.conv2d_bn(x, 64, (3, 3), (1, 1))
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        # conv -> conv
        x = self.conv2d_bn(x, 80, (1, 1), (1, 1), padding='valid')
        x = self.conv2d_bn(x, 192, (3, 3), (1, 1), padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x35 x256
        branch1x1 = self.conv2d_bn(x, 64, (1, 1), (1, 1))
        branch3x3 = self.conv2d_bn(x, 64, (1, 1), (1, 1))
        branch3x3 = self.conv2d_bn(branch3x3, 96, (3, 3), (1, 1))
        branch3x3 = self.conv2d_bn(branch3x3, 96, (3, 3), (1, 1))
        branch5x5 = self.conv2d_bn(x, 48, (1, 1), (1, 1))
        branch5x5 = self.conv2d_bn(branch5x5, 64, (5, 5), (1, 1))
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, (1, 1), (1, 1))
        x = Concatenate(axis=3, name='mixed0')([branch1x1, branch3x3, branch5x5, branch_pool])

        # mixed 1: 35 x 35 x 288
        branch1x1 = self.conv2d_bn(x, 64, (1, 1), (1, 1))
        branch3x3 = self.conv2d_bn(x, 64, (1, 1), (1, 1))
        branch3x3 = self.conv2d_bn(branch3x3, 96, (3, 3), (1, 1))
        branch3x3 = self.conv2d_bn(branch3x3, 96, (3, 3), (1, 1))
        branch5x5 = self.conv2d_bn(x, 48, (1, 1), (1, 1))
        branch5x5 = self.conv2d_bn(branch5x5, 64, (5, 5), (1, 1))
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, (1, 1), (1, 1))
        x = Concatenate(axis=3, name='mixed1')([branch1x1, branch3x3, branch5x5, branch_pool])

        # mixed 2: 35 x 35 x288
        branch1x1 = self.conv2d_bn(x, 64, (1, 1), (1, 1))
        branch3x3 = self.conv2d_bn(x, 64, (1, 1), (1, 1))
        branch3x3 = self.conv2d_bn(branch3x3, 96, (3, 3), (1, 1))
        branch3x3 = self.conv2d_bn(branch3x3, 96, (3, 3), (1, 1))
        branch5x5 = self.conv2d_bn(x, 48, (1, 1), (1, 1))
        branch5x5 = self.conv2d_bn(branch5x5, 64, (5, 5), (1, 1))
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, (1, 1), (1, 1))
        x = Concatenate(axis=3, name='mixed2')([branch1x1, branch3x3, branch5x5, branch_pool])

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv2d_bn(x, 384, (3, 3), (2, 2), padding='valid')

        branch3x3dbl = self.conv2d_bn(x, 64, (1, 1), (1, 1))
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, (3, 3), (1, 1))
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, (3, 3), strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Concatenate(axis=3)([branch3x3, branch3x3dbl, branch_pool])

        # mixed 4: 17 x17 x768
        branch1x1 = self.conv2d_bn(x, 192, (1, 1), (1, 1))
        branch7x7 = self.conv2d_bn(x, 128, (1, 1), (1, 1))
        branch7x7 = self.conv2d_bn(branch7x7, 128, (1, 7), (1, 1))
        branch7x7 = self.conv2d_bn(branch7x7, 192, (7, 1), (1, 1))
        branch7x7dbl = self.conv2d_bn(x, 128, (1, 1), (1, 1))
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, (7, 1), (1, 1))
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, (1, 7), (1, 1))
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, (7, 1), (1, 1))
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, (1, 7), (1, 1))
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, (1, 1), (1, 1))
        x = Concatenate(axis=3, name='mixed4')([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 5, 6: 17 x17 x768
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, (1, 1), (1, 1))
            branch7x7 = self.conv2d_bn(x, 160, (1, 1), (1, 1))
            branch7x7 = self.conv2d_bn(branch7x7, 160, (1, 7), (1, 1))
            branch7x7 = self.conv2d_bn(branch7x7, 192, (7, 1), (1, 1))
            branch7x7dbl = self.conv2d_bn(x, 160, (1, 1), (1, 1))
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, (7, 1), (1, 1))
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, (1, 7), (1, 1))
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, (7, 1), (1, 1))
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, (1, 7), (1, 1))
            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            x = Concatenate(axis=3, name='mixed' + str(5 + i))([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, (1, 1), (1, 1))

        branch7x7 = self.conv2d_bn(x, 192, (1, 1), (1, 1))
        branch7x7 = self.conv2d_bn(branch7x7, 192, (1, 7), (1, 1))
        branch7x7 = self.conv2d_bn(branch7x7, 192, (7, 1), (1, 1))

        branch7x7dbl = self.conv2d_bn(x, 192, (1, 1), (1, 1))
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, (7, 1), (1, 1))
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, (1, 7), (1, 1))
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, (7, 1), (1, 1))
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, (1, 7), (1, 1))

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = Concatenate(axis=3, name='mixed7')([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv2d_bn(x, 192, (1, 1), (1, 1))
        branch3x3 = self.conv2d_bn(branch3x3, 320, (3, 3), (2, 2), padding='valid')

        branch7x7x3 = self.conv2d_bn(x, 192, (1, 1), (1, 1))
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, (1, 7), (1, 1))
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, (7, 10), (1, 1))
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, (3, 3), (2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = Concatenate(axis=3, name='mixed8')( [branch3x3, branch7x7x3, branch_pool],)

        return x

    def get_layer(self):
        model = Model(inputs=self.inpt, outputs=self.inceptionv3)
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
    inceptionv1 = InceptionV3()
    inceptionv1.get_layer()