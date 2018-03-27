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
dataset.apply(lambda x: normalize_pressure_value(x, 512))

#%% Split Train, Test
data = dataset.as_numpy(MASK_VALUE)[:, :, :2].astype('float32')
encoder, labels = dataset.get_labels_as_numpy(onehot=True)
labels = labels.astype('float32').todense()
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, shuffle=True, stratify=labels, random_state=42)


#%%

from keras import Sequential
from keras.layers import Masking
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Masking(mask_value=MASK_VALUE, input_shape=(X_train.shape[1:])))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(Dense(10))
model.add(Activation('softmax'))

#%%
import os
save_path = 'checkpoints'
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_filename = save_path + "/naive-lstm-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
checkpointer = ModelCheckpoint(save_filename, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(x=X_train, y=Y_train, epochs=15, verbose=1, callbacks=[checkpointer], validation_split=0.33)

test_score = tuple(model.evaluate(X_test, Y_test))
print("Test Loss: %.3f, Test Acc: %.3f%%" % (test_score[0], test_score[1] * 100))

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
