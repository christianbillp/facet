##% For test
import pandas as pd


#%% Image resizer
from PIL import Image
import numpy as np


def convert(image_path):
    """Converts an image into a grayscale uint8 np array"""
    x = Image.open(image_path).convert('LA')
    x = x.convert('L')
    y = np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))
    
    return y

data = convert("banner-car.png")

# -*- coding: utf-8 -*-

#%%
#%%
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt

#%% Training Specification
batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_x, img_y = 28, 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#%% Model Definition
import numpy as np
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

# Specific for MNIST dataset
input_shape = (28, 28, 1)
num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.set_weights(np.load('trained_weights'))

#%% Train and Evaluate Model
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#%% Dump Weights
np.array(model.get_weights()).dump('trained_weights')

#%% Make predictinos


def predict(item):
    """Uses model to make prediction"""
    data = x_test[item:item+1]    # Must have a list
    result = model.predict(data)
    
    for i, r in enumerate(result[0]):
        print("Probability for: {} is {}".format(i, r))
    print("Correct is: {}".format(np.argmax(y_test[item])))
    
    return result

predict(33)


#%% For testing

from keras.datasets import mnist
img_x, img_y = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



#%%

from PIL import Image
import numpy as np


def get_image(self, image_path):
    image_path = 'banner-car.png'
    x = Image.open(image_path).convert('LA').resize((28, 28))
    image_data = x.convert('L')
    y = np.asarray(image_data.getdata(),dtype=np.uint8).reshape((image_data.size[1],image_data.size[0]))

    return y





#%% Make an image into C array

d = d.astype('int')


print('{', end='')
for line in d:
    print("{", end='')
    for value in line:
        print("{}, ".format(value), end='')
    print("},")
print('}')
#%%
base = 0
n=784

vals=[base+i for i in range(n)]

print("{", end='')
for value in vals:
    print('{}, '.format(value), end='')
print("}")



#%%

