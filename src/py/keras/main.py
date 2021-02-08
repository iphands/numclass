import os

from lib import data_loader as loader
from lib import utils as utils
from lib import consts as consts

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

X_train = loader.get_image_data(consts.TRAIN_IMAGE_PATH)
y_train = loader.get_label_data(consts.TRAIN_LABEL_PATH)

X_test = loader.get_image_data(consts.TRAIN_IMAGE_PATH)
y_test = loader.get_label_data(consts.TRAIN_LABEL_PATH)

print('Using filename: {}'.format(utils.get_model_name()))

# define the keras model
model = Sequential()
model.add(Dense(512, input_shape=(28*28, ), activation='relu'))

# Add hidden layers
for l in consts.LAYERS:
    model.add(Dense(l, activation='relu'))
    model.add(Dense(l, activation='sigmoid'))

model.add(Dense(consts.RESULT_COUNT, activation='softmax'))

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = utils.get_model_shortname() + 'model_epoch_{epoch:02d}_loss_{val_loss:.16f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')

history = model.fit(X_train,
                    utils.get_one_hot(y_train),
                    batch_size=128,
                    epochs=consts.EPOCHS,
                    verbose=2,
                    callbacks=[checkpoint],
                    validation_data=(X_test, utils.get_one_hot(y_test)))

# saving the model
save_dir = './results'
model_name = utils.get_model_name()
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
