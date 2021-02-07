# imports for array-handling and plotting
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# for testing on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
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

X_test = get_image_data('../../../data/t10k-images-idx3-ubyte')
y_test = get_label_data('../../../data/t10k-labels-idx1-ubyte')
X_test = X_test.astype('float32')
X_test /= 255

# load the model and create predictions on the test set
mnist_model = load_model('./results/512_128_512.h5')
predicted_classes = mnist_model.predict_classes(X_test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

for i, guess in enumerate(predicted_classes):
    if guess != y_test[i]:
        print('{} file:///home/iphands/prog/numclass/data/test_images/all/{}.jpg guessed {} is {}'
              .format(guess == y_test[i], i, guess, y_test[i]))
