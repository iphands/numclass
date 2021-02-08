from . import consts as consts
from keras.utils import np_utils


def get_model_shortname():
    lname  = '_'.join([str(l) for l in consts.LAYERS])
    return './results/layers_{}_epochs_{}/'.format(lname, consts.EPOCHS)

def _do_conv2d(layer, info):
    info.append(str(layer.filters))
    info.append(str(layer.kernel_size)
                .replace('(',  '')
                .replace(', ', 'x')
                .replace(')',  ''))

def get_model_desc(model):
    info = []
    for l in model.layers:
        n = l.name.split('_')[0]
        info.append(n)
        if hasattr(l, 'activation'): info.append(l.activation.__name__)
        if n == 'dense':  info.append(str(l.units))
        if n == 'conv2d': _do_conv2d(l, info)
    return '_'.join(info)

def get_model_name():
    lname  = '_'.join([str(l) for l in consts.LAYERS])
    return '{}-epochs_{}.h5'.format(lname, consts.EPOCHS)


def get_one_hot(data):
    return np_utils.to_categorical(data, consts.RESULT_COUNT)
