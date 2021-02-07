import os
import numpy as np

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

def get_bytes(l):
    return bytearray(f.read(l))


def get_next_image(f, shape):
    return bytearray(f.read(shape[1] * shape[2]))


def get_label_data(file_in):
    labels = []
    with open(file_in, 'rb') as f:
        magic = bytearray(f.read(2)) # 0x00 0x00
        dtype = f.read(1) # 0x08: unsigned byte
        dimen = f.read(1) # 0x01
        shape = []

        for i in range(ord(dimen)):
            shape.append(int.from_bytes(f.read(4), 'big'))

        for i in range(shape[0]):
            labels.append(int.from_bytes(f.read(1), 'big'))

    return np.reshape(labels, (shape[0]))

def get_image_data(file_in):
    images = []
    with open(file_in, 'rb') as f:
        magic = bytearray(f.read(2)) # 0x00 0x00
        dtype = f.read(1) # 0x08: unsigned byte
        dimen = f.read(1) # 0x03
        shape = []

        for i in range(ord(dimen)):
            shape.append(int.from_bytes(f.read(4), 'big'))

        for i in range(shape[0]):
            data = get_next_image(f, shape)
            images.append(data)
    return np.reshape(images, (shape[0], shape[1] * shape[2]))

X_train = get_image_data('../../../data/train-images-idx3-ubyte')
y_train = get_label_data('../../../data/train-labels-idx1-ubyte')

X_test = get_image_data('../../../data/t10k-images-idx3-ubyte')
y_test = get_label_data('../../../data/t10k-labels-idx1-ubyte')


y_test = get_label_data('../../../data/t10k-labels-idx1-ubyte')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Train lbl matrix shape", y_train.shape)

print("Test matrix shape", X_test.shape)
print("Test lbl matrix shape", y_test.shape)

print(np.unique(y_train, return_counts=True))

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# define the keras model
model = Sequential()
model.add(Dense(512, input_shape=(28*28, ), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the model and saving metrics in history
history = model.fit(X_train, Y_train,
          batch_size=128, epochs=128,
          verbose=2,
          validation_data=(X_test, Y_test))

# saving the model
save_dir = './results'
model_name = '512_128_512.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
