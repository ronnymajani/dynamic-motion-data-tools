# -*- coding: utf-8 -*-

from utils.preprocessing import *
from data.DataSet import DataSet
from sklearn.model_selection import train_test_split

#%%
MASK_VALUE = -2.0

#%% Prepare Data
folder = 'temp'
dataset = DataSet(folder)
dataset.apply(apply_mean_centering)
dataset.apply(apply_unit_distance_normalization)
dataset.apply(lambda digit: normalize_pressure_value(digit, 512))
dataset.expand(reverse_digit_sequence)
# dataset.apply(lambda digit: convert_xy_to_derivative(digit, normalize=False))
dataset.apply(lambda digit: convert_xy_to_derivative(digit, normalize=True))

#%% Split Train, Test
from keras.preprocessing.sequence import pad_sequences
#data = dataset.as_numpy(MASK_VALUE)[:, :, :2].astype('float32')
data = pad_sequences(dataset.data, dtype='float32', padding='pre', truncating='post', value=MASK_VALUE)
data = data[:, :, :2]
encoder, labels = dataset.get_labels_as_numpy(onehot=True)
labels = labels.astype('float32').todense()
X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(data, labels, shuffle=True, stratify=labels, random_state=42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, shuffle=True, stratify=Y_train_valid, random_state=42)

#%%
from keras import backend as K
K.set_learning_phase(1) #set learning phase
from keras import Sequential
from keras.layers import Masking
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

model = Sequential()
model.add(Masking(mask_value=MASK_VALUE, input_shape=(X_train.shape[1:])))
#model.add(LSTM(128, return_sequences=True))/
model.add(LSTM(256, return_sequences=True))
#model.add(Dropout(0.5))
model.add(LSTM(256))
#model.add(Dropout(0.5))
#model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

#%%
import time
from keras.callbacks import ModelCheckpoint, TensorBoard
import tfcallback
from keras.optimizers import Adam

save_path = 'checkpoints/9'
save_prefix = 'lstm-overfit'
tensorboard_path = "log"

BATCH_SIZE = 32

import os
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_filename = save_path + "/" + save_prefix + "-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
checkpointer = ModelCheckpoint(save_filename, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

tensorboard_dir = tensorboard_path + "/{}".format(time.time())
tensorboard_callback = tfcallback.TB(log_every=1, log_dir=tensorboard_dir, write_graph=False)
tensorboard_callback.write_batch_performance = True

optimizer = Adam()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

print(model.summary())

fit_params = {
        'x': X_train,
        'y': Y_train,
        'epochs': 30,
        'verbose': 1,
        'callbacks': [checkpointer, tensorboard_callback],
        'validation_data': (X_valid, Y_valid),
        'batch_size': BATCH_SIZE
}
model.fit(**fit_params)
#model.fit(x=X_train, y=Y_train, epochs=30, verbose=1, callbacks=[checkpointer, tensorboard_callback], validation_data=(X_valid, Y_valid), batch_size=BATCH_SIZE)

test_score = tuple(model.evaluate(X_test, Y_test))
print("Test Loss: %.3f, Test Acc: %.3f%%" % (test_score[0], test_score[1] * 100))

#%%
continue_params = fit_params.copy()
continue_params['epochs'] = 60
continue_params['initial_epoch'] = 30
model.fit(**continue_params)

#%%
#def predict(digit, model):
#    import utils.preprocessing as preprocessing
#    import numpy as np
#    max_seq_len = 301
#    mask = -2.0
#    digit_len = min(len(digit), max_seq_len)
#    digit = np.array(digit[:digit_len])
#    digit = preprocessing.apply_mean_centering(digit)
#    digit = preprocessing.apply_unit_distance_normalization(digit)
#    digit = preprocessing.normalize_pressure_value(digit)
#    dataz = np.empty((max_seq_len, 3))
#    dataz.fill(mask)
#    dataz[:digit_len, :] = digit[:digit_len, :]
#    dataz = dataz.reshape(1, -1, 3)
#    return model.predict_classes(dataz, verbose=1)[0]