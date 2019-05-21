#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/18
# @Author  : Chen

"""
Test for LeNet
"""

# import the necessary packages
from lenet import LeNet
from keras.datasets import mnist
from keras.optimizers import SGD
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

session_config = tf.ConfigProto()  # 配置见第1节
session_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用40%显存
session = tf.Session(config=session_config)

# prepare the train & test dataset and labels
# expand 3D dataset to 4D dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

# convert the labels form integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# define the model
lenet = LeNet()
model = lenet.net

# optimizer
sgd = SGD(momentum=0.9, nesterov=True)

print("[INFO] start training...")
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=128)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x_test, batch_size=128)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("Training Loss and Accuracy")