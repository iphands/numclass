from . import consts as consts
from keras.utils import np_utils


def get_model_name():
    lname  = '_'.join([str(l) for l in consts.LAYERS])
    return '{}-epochs_{}.h5'.format(lname, consts.EPOCHS)


def get_one_hot(data):
    return np_utils.to_categorical(data, consts.RESULT_COUNT)
