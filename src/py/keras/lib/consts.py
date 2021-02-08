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

EPOCHS = 128
LAYERS = [ 28*28 * 2, 512, 512 ]
