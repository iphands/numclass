
def get_bytes(l):
    return bytearray(f.read(l))

def print_next_image(f, shape):
    data = bytearray(f.read(shape[1] * shape[2]))
    print()
    for y in range(shape[1]):
        for x in range(shape[2]):
            val = data[x + (y * shape[1])]
            if val == 0x00:
                print('  ', end=' ')
            else:
                print('{:02X}'.format(val), end=' ')
        print()

with open('../../data/train-images-idx3-ubyte', 'rb') as f:
    magic = bytearray(f.read(2)) # 0x00 0x00
    dtype = f.read(1) # 0x08: unsigned byte
    dimen = f.read(1) # 0x03
    shape = []

    for i in range(ord(dimen)):
        shape.append(int.from_bytes(f.read(4), 'big'))

    for i in range(128):
        print_next_image(f, shape)

