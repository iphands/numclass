import os
import numpy as np
from PIL import Image

labels = []
image_num = 0

def get_bytes(l):
    return bytearray(f.read(l))


def get_next_image(f, shape):
    return bytearray(f.read(shape[1] * shape[2]))

def save_image(i, data, shape):
    img = Image.fromarray(np.reshape(data, (shape[1], shape[2])), 'L')
    img.save('../../data/images/{}/{}.jpg'.format(labels[i], i))

def print_image(i, data, shape):
    print()
    for y in range(shape[1]):
        for x in range(shape[2]):
            val = data[x + (y * shape[1])]
            if val == 0x00:
                print('  ', end=' ')
            else:
                print('{:02X}'.format(val), end=' ')
        print()


with open('../../data/train-labels-idx1-ubyte', 'rb') as f:
    magic = bytearray(f.read(2)) # 0x00 0x00
    dtype = f.read(1) # 0x08: unsigned byte
    dimen = f.read(1) # 0x03
    shape = []

    for i in range(ord(dimen)):
        shape.append(int.from_bytes(f.read(4), 'big'))

    for i in range(shape[0]):
        labels.append(int.from_bytes(f.read(1), 'big'))

for label in labels:
    d = '../../data/images/{}'.format(label)
    if not os.path.exists(d):
        os.makedirs(d)

with open('../../data/train-images-idx3-ubyte', 'rb') as f:
    magic = bytearray(f.read(2)) # 0x00 0x00
    dtype = f.read(1) # 0x08: unsigned byte
    dimen = f.read(1) # 0x03
    shape = []

    for i in range(ord(dimen)):
        shape.append(int.from_bytes(f.read(4), 'big'))

    for i in range(shape[0]):
        data = get_next_image(f, shape)
        # print_image(i, data, shape)
        save_image(i, data, shape)

