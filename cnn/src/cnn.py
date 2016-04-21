# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:01:49 2016

@author: jzhao
"""

from keras.models import Sequential
from keras.layers.core import Activation, Dense

model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print('complie done')

# generate dummy data
import numpy as np
data = np.random.random((1000, 784))
labels = np.random.randint(2, size=(1000, 1))

import arff
arff.load('../toydata/vote_train.arff')


# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)

model.predict