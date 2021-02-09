import numpy as np

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

def get_image_data(file_in, layer_type, rotate_hack=False):
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

            if rotate_hack:
                data = np.asarray(data)
                data = np.reshape(data, (28,28))
                data = np.rot90(data, k=3)
                data = np.fliplr(data)
                data = np.reshape(data, (28*28))

            images.append(data)

    # convert to floats and / 255 to get 0.0 - 1.0
    if layer_type == 'dense':
        return np.reshape(images, (shape[0], shape[1] * shape[2])).astype('float32') / 255

    # reshape for the conv2d layers
    tmp = np.reshape(images, (shape[0], shape[1], shape[2])).astype('float32') / 255
    return np.expand_dims(tmp, -1)
