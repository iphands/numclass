import os

from lib import data_loader as loader
from lib import utils as utils
from lib import consts as consts

from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

model = Sequential()
# model.add(Dense(512, input_shape=(28*28, ), activation='relu'))

model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32,  (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64,  (3, 3), padding="same", activation="relu"))
model.add(Conv2D(32,  (6, 6), padding="same", activation="relu"))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='sigmoid'))

# final layer of 10
model.add(Dense(consts.RESULT_COUNT, activation='softmax'))

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print()
print('Model description is: {}'.format(utils.get_model_desc(model)))
print()

filepath = './results/' + utils.get_model_desc(model) + '/epoch_{epoch:02d}_loss_{val_loss:.16f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')

X_train = loader.get_image_data(consts.TRAIN_IMAGE_PATH, model.layers[0].name)
y_train = loader.get_label_data(consts.TRAIN_LABEL_PATH)

X_test = loader.get_image_data(consts.TRAIN_IMAGE_PATH, model.layers[0].name)
y_test = loader.get_label_data(consts.TRAIN_LABEL_PATH)

history = model.fit(X_train,
                    utils.get_one_hot(y_train),
                    batch_size=64,
                    epochs=consts.EPOCHS,
                    verbose=2,
                    callbacks=[checkpoint],
                    validation_data=(X_test, utils.get_one_hot(y_test)))
