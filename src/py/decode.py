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

    file_name     = '{}.jpg'.format(i)
    file_name_idx = '{}-{}.jpg'.format(labels[i], i)

    file_all = '../../data/alpha/all/{}'.format(file_name)
    file_sorted = '../../data/alpha/{}/{}'.format(labels[i], file_name_idx)

    img.save(file_all)
    os.symlink('../all/{}'.format(file_name), file_sorted)
    # os.symlink('../all/{}'.format(file_name), '../../data/images/all_idx/{}'.format(file_name_idx))


def print_image(i, data, shape):
    print()
    for y in range(shape[1]):
        for x in range(shape[2]):
            val = data[x + (y * shape[1])]
            if val < 0xc0:
                print('  ', end=' ')
            else:
                print('{:02X}'.format(val), end=' ')
        print()


# with open('../../data/train-labels-idx1-ubyte', 'rb') as f:
# with open('../../data/t10k-labels-idx1-ubyte', 'rb') as f:
with open('../../data/emnist-letters-train-labels-idx1-ubyte', 'rb') as f:
    magic = bytearray(f.read(2)) # 0x00 0x00
    dtype = f.read(1) # 0x08: unsigned byte
    dimen = f.read(1) # 0x01
    shape = []

    for i in range(ord(dimen)):
        shape.append(int.from_bytes(f.read(4), 'big'))

    for i in range(shape[0]):
        labels.append(int.from_bytes(f.read(1), 'big'))

for label in labels:
    d = '../../data/alpha/{}'.format(label)
    if not os.path.exists(d):
        os.makedirs(d)


# for d in ['../../data/images/all', '../../data/images/all_idx']:
#     if not os.path.exists(d):
#         os.makedirs(d)


with open('../../data/emnist-letters-train-images-idx3-ubyte', 'rb') as f:
    magic = bytearray(f.read(2)) # 0x00 0x00
    dtype = f.read(1) # 0x08: unsigned byte
    dimen = f.read(1) # 0x03
    shape = []

    for i in range(ord(dimen)):
        shape.append(int.from_bytes(f.read(4), 'big'))

    # for i in range(shape[0]):
    for i in range(1024):
        data = get_next_image(f, shape)
        data = np.asarray(data)
        data = np.reshape(data, (28,28))
        data = np.rot90(data, k=3)
        data = np.fliplr(data)
        data = np.reshape(data, (28*28))
        # print_image(i, data, shape)
        save_image(i, data, shape)
