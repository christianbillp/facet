import numpy as np
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from PIL import Image

class Hive():

    def __init__(self):
        print('Initiated')
        self.model = 0
        
    def set_model(self):
        # Specific for MNIST dataset
        input_shape = (28, 28, 1)
        num_classes = 10

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                       activation='relu',
                       input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(64, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
        self.model.set_weights(np.load('trained_weights'))
        print('Model set')
    
    def get_weights(self):
        return self.model.get_weights()
    
    def parse_weights(self):
        print(self.model.get_weights()[0][0])
        
    def predict(self, data):
        prediction = self.model.predict(data)
        
        for i, r in enumerate(prediction[0]):
            print("Probability for: {} is {:.2f}".format(i, r))
        
        return prediction
    
    def get_image(self, image_path):
#        image_path = 'banner-car.png'
        x = Image.open(image_path).convert('LA').resize((28, 28))
        image_data = x.convert('L')
        y = np.asarray(image_data.getdata(),dtype=np.uint8).reshape((image_data.size[1],image_data.size[0]))
        print(y)
    
        return y

h = Hive()
h.set_model()
ulla = h.get_weights()
h.parse_weights()
#%%
i = 77
h = Hive()
h.set_model()
h.predict(x_test[i:i+1])
print("Correct is: {}".format(np.argmax(y_test[i])))
#%%
h = Hive()
h.set_model()
#data = h.get_image('banner-car.png')
data = h.get_image('7.png')
data = 1 - data.astype('float64') / 255
d = data
data = data.reshape((1,28,28,1))
#data = np.expand_dims(data, 1)

h.predict(data)


#%% For test
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