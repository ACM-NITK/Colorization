#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import random
import tensorflow as tf

from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave

image = img_to_array(load_img('baby.jpg'))
image = np.array(image, dtype=float)


X = rgb2lab(1.0/255*image)[:,:,0]
Y = rgb2lab(1.0/255*image)[:,:,1:]
Y /= 128
X = X.reshape(1, 200, 200, 1)
Y = Y.reshape(1, 200, 200, 2)

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))


model.compile(optimizer='adam', loss='mse')


model.fit(x=X, y=Y, batch_size=1, epochs=1000)

output = model.predict(X)
image = img_to_array(load_img('bab2.jpg'))
image = np.array(image, dtype=float)

Xn = rgb2lab(1.0/255*image)[:,:,0]
Xn = Xn.reshape(1, 200, 200, 1)

out2 = model.predict(Xn)
output *= 128
out2*=128
cur = np.zeros((200, 200, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("img_result.png", lab2rgb(cur))
imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))
cur = np.zeros((200, 200, 3))
cur[:,:,0] = Xn[0][:,:,0]
cur[:,:,1:] = out2[0]
imsave("img_.png", lab2rgb(cur))
imsave("img_g.png", rgb2gray(lab2rgb(cur)))