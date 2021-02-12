from os import listdir
from os.path import isfile, join
from . import consts as consts
from PIL import Image

import numpy as np

def get_bytes(l):
    return bytearray(f.read(l))


def get_next_image(f, shape):
    return bytearray(f.read(shape[1] * shape[2]))


def get_label_data(file_in, extra=None):
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

    if extra:
        print('Adding {} extra labels'.format(len(extra)))
        labels = labels + extra

    return np.reshape(labels, (len(labels)))

def print_image(data, shape):
    print()
    for y in range(shape[0]):
        if y > 22: return
        for x in range(shape[1]):
            idx = x + (y * shape[1])
            val = data[idx]

            if val == 0x00:
                print('  ', end=' ')
            else:
                print('{:02X}'.format(val), end=' ')
        print()

def defuzz_image(arr):
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            val = arr[y][x]
            if val != 0x00 and val < 0x10:
                arr[y][x] = 0x00
    return arr

def get_my_image_data():
    images = []
    labels = []

    for f in listdir(consts.MY_IMAGES):
        full_path = join(consts.MY_IMAGES, f)
        if isfile(full_path):
            im  = Image.open(full_path)
            arr = np.array(im)
            arr = defuzz_image(arr)
            arr = np.reshape(arr, (28*28))
            lbl = f.split('-')[0]
            # print(lbl)
            # print_image(arr, (28, 28))
            images.append(arr)
            labels.append(lbl)

    return (images, labels)

def get_image_data(file_in, layer_type, extra=None):
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

    if extra:
        print('Adding {} extra images'.format(len(extra)))
        images = images + extra

    # convert to floats and / 255 to get 0.0 - 1.0
    if layer_type == 'dense':
        return np.reshape(images, (len(images), shape[1] * shape[2])).astype('float32') / 255

    # reshape for the conv2d layers
    tmp = np.reshape(images, (len(images), shape[1], shape[2])).astype('float32') / 255
    return np.expand_dims(tmp, -1)
