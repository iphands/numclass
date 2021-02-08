TRAIN_IMAGE_PATH = '../../../data/train-images-idx3-ubyte'
TRAIN_LABEL_PATH = '../../../data/train-labels-idx1-ubyte'
TEST_IMAGE_PATH  = '../../../data/t10k-images-idx3-ubyte'
TEST_LABEL_PATH  = '../../../data/t10k-labels-idx1-ubyte'

# # temp testing the flip
# TEST_IMAGE_PATH = '../../../data/train-images-idx3-ubyte'
# TEST_LABEL_PATH = '../../../data/train-labels-idx1-ubyte'
# TRAIN_IMAGE_PATH  = '../../../data/t10k-images-idx3-ubyte'
# TRAIN_LABEL_PATH  = '../../../data/t10k-labels-idx1-ubyte'

RESULT_COUNT = 10

EPOCHS = 1024
LAYERS = [ 28*28 * 2, 28*28, 512, 512 ]
# LAYERS = [ 32 ]

WIN_W = 400
WIN_H = 400
