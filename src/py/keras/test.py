import sys
import numpy as np
import os

from keras.models import load_model

from lib import consts as consts
from lib import data_loader as loader


mnist_model = load_model(sys.argv[1])

X_test = loader.get_image_data(consts.TEST_IMAGE_PATH, mnist_model.layers[0].name)
y_test = loader.get_label_data(consts.TEST_LABEL_PATH)

predicted_classes = mnist_model.predict_classes(X_test)

# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

fail_count = 0

training_images_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'train_images', 'all'))
# just in case, but come on ...
training_images_dir.replace(os.sep, '/')

for i, guess in enumerate(predicted_classes):
    if guess != y_test[i]:
        print('file:///{}/{}.jpg guessed {} is {}'
              .format(training_images_dir, i, guess, y_test[i]))
        fail_count += 1

    if fail_count > 25:
        break
