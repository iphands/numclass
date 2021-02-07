import numpy as np

from keras.models import load_model

from lib import consts as consts
from lib import data_loader as loader

X_test = loader.get_image_data(consts.TEST_IMAGE_PATH)
y_test = loader.get_label_data(consts.TEST_LABEL_PATH)

mnist_model = load_model('./results/512_512_128_128_64_32-epochs_128.h5')
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
