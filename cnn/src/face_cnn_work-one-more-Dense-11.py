import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


batch_size = 10
nb_landmarks = 5
nb_epoch = 500
#nb_validation_data = (X_test, _Y_test)
nb_validation_data = None

nb_submodels = 3

# input image dimensions
img_chns, img_rows, img_cols = 3, 20, 20
# number of convolutional filters to use
nb_filters = [32, 64]
# size of pooling area for max pooling
nb_pool_sizes = [(2, 2), (2, 2)]
# convolution kernel size
nb_conv = 3
# activator
nb_activator = 'tanh'
# number of fully connected neurons in the penultimate layer
nb_penu_neurons = 128
# size of output vector, four "corner values" for each landmark
nb_output_size = nb_landmarks * 4

input_img = Input(shape=(img_chns, img_rows, img_cols), dtype='float32', name='input_img')

def submodel(x) :
    x = (Convolution2D(nb_filters[0], nb_conv, nb_conv))(x)
    x = (Activation(nb_activator))(x)
    x = (MaxPooling2D(pool_size=nb_pool_sizes[0]))(x)
    x = (Convolution2D(nb_filters[1], nb_conv, nb_conv))(x)
    x = (Activation(nb_activator))(x)
    x = (MaxPooling2D(pool_size=nb_pool_sizes[1]))(x)
    x = (Dropout(0.25))(x)

    x = (Flatten())(x)
    x = (Dense(nb_penu_neurons))(x)
    x = (Activation(nb_activator))(x)
    x = (Dropout(0.5))(x)
    output_landmarks = (Dense(nb_landmarks * 2))(x)
    return output_landmarks

sub_output = submodel(input_img)
x = sub_output
x = Dense(nb_penu_neurons)(x)
x = (Activation(nb_activator))(x)
x = (Dropout(0.5))(x)

x = (Dense(nb_landmarks * 2))(x)
output_landmarks = (Reshape((nb_landmarks, 2)))(x)

model = Model(input=input_img, output=output_landmarks)

model.compile(loss='mse',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=0, validation_data=nb_validation_data)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

