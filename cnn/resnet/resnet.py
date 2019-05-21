#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/21
# @Author  : Chen

"""
implement ResNet with Keras
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
"""

# import the necessary packages
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense
from keras.layers import ZeroPadding2D, add
from keras.layers import GlobalAveragePooling2D,

class ResNet50:
    def __init__(self):
        self.inpt = Input(shape=(224, 224, 3))
        self.resnet50 = self.build_network()

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # 1 x 1
        x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        # 3 x 3
        x = Conv2D(filters2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        # 1 x 1
        x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
        # 另一路
        shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return  x

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """更像是figure5 的二个图: 1x1, 3x3, 1x1"""
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # 1 x 1
        x = Conv2D(filters1, (1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        # 3 x 3
        x = Conv2D(filters2, kernel_size,
                   padding='same',
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        # 1 x 1
        x = Conv2D(filters3, (1, 1),
                   kernel_initializer='he_normal',
                   name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x



    def build_network(self):
        inpt = self.inpt
        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inpt)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(1000, activation='softmax', name='fc1000')(x)
        return x

    def get_layer(self):
        model = Model(inputs=self.inpt, outputs=self.resnet50)
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
    inceptionv1 = ResNet50()
    inceptionv1.get_layer()