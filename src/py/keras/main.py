import os

import numpy as np

from lib import data_loader as loader
from lib import utils as utils
from lib import consts as consts

from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(16, (4, 4), padding="same", activation="relu"))
model.add(Conv2D(8,  (3, 3), padding="same", activation="relu"))
model.add(Conv2D(4,  (3, 3), padding="same", activation="relu"))
model.add(Conv2D(4,  (3, 3), padding="same", activation="relu"))

model.add(Flatten())
model.add(Dropout(0.15))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# final layer of 10
model.add(Dense(consts.RESULT_COUNT, activation='softmax'))

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

results_dir = './results/' + utils.get_seconds()
os.mkdir(results_dir)
with open('{}/model_desc.txt'.format(results_dir), 'w') as f:
    f.write(utils.get_model_desc(model))

filepath = results_dir + '/epoch_{epoch:02d}_loss_{val_loss:.16f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')

my_images, my_labels = loader.get_my_image_data()

X_train = loader.get_image_data(consts.TRAIN_IMAGE_PATH, model.layers[0].name, extra=my_images)
y_train = loader.get_label_data(consts.TRAIN_LABEL_PATH, extra=my_labels)

X_nan = loader.get_image_data(consts.NAN_IMAGE_PATH, model.layers[0].name)
y_nan = np.full((len(X_nan,)), 10)

X_train = np.concatenate((X_train, X_nan), axis=0)
y_train = np.concatenate((y_train, y_nan), axis=None)

X_test = loader.get_image_data(consts.TEST_IMAGE_PATH, model.layers[0].name)
y_test = loader.get_label_data(consts.TEST_LABEL_PATH)

history = model.fit(X_train,
                    utils.get_one_hot(y_train),
                    batch_size=64,
                    epochs=consts.EPOCHS,
                    verbose=2,
                    callbacks=[checkpoint],
                    validation_data=(X_test, utils.get_one_hot(y_test)))
