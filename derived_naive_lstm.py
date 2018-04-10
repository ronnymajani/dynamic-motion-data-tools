# -*- coding: utf-8 -*-

#%%
# Constants
PARAM_NUM_EPOCHS = 30
PARAM_BATCH_SIZE = 300
NUM_SAMPLES = 200

# Paths
dataset_folder_path = 'temp'
tensorboard_logs_path = "logs"
checkpoints_save_folder_path = 'checkpoints'
checkpoints_save_prefix = 'gru-spline'

#%% Prepare Data
# Imports
from utils.preprocessing import *
from data.DataSet import DataSet
from functools import partial

dataset = DataSet(dataset_folder_path)
dataset.apply(apply_mean_centering)
dataset.apply(apply_unit_distance_normalization)
dataset.apply(partial(normalize_pressure_value, max_pressure_val=512))
dataset.apply(partial(spline_interpolate_and_resample, num_samples=NUM_SAMPLES))
dataset.expand(reverse_digit_sequence)
# dataset.apply(lambda digit: convert_xy_to_derivative(digit, normalize=False))
#dataset.apply(partial(convert_xy_to_derivative, normalize=True))

#%% Split Train, Valid, Test
# Imports
import numpy as np
from sklearn.model_selection import train_test_split

data = np.array(dataset.data)
# Convert labels to numpy array and OneHot encode them
encoder, labels = dataset.get_labels_as_numpy(onehot=True)
labels = labels.astype('float32').todense()
# Split Data
X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(data, labels, shuffle=True, stratify=labels, random_state=42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, shuffle=True, stratify=Y_train_valid, random_state=42)


#%% Build Model
# Imports
from keras import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import Adam

# Model
model = Sequential()
model.add(GRU(256, return_sequences=True, input_shape=(X_train.shape[1:])))
model.add(GRU(256))
model.add(Dense(10))
model.add(Activation('softmax'))

# Optimizer
optimizer = Adam()

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
print(model.summary())


#%% Create Folders
# Imports
import os
import time

curr_time = time.time()
tensorboard_dir = os.path.join(tensorboard_logs_path, "{}".format(curr_time))
checkpoints_dir = os.path.join(checkpoints_save_folder_path, "{}".format(curr_time))

# Create Checkpoints save directory if it doesn't exist
if not os.path.exists(checkpoints_save_folder_path):
    os.mkdir(checkpoints_save_folder_path)
if not os.path.exists(tensorboard_logs_path):
    os.mkdir(tensorboard_logs_path)
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)


#%% Record Everything that has been done in this version
# Imports
import pprint
import datetime
    
with open(os.path.join(checkpoints_dir, "summary.txt"), "w") as fd:
    sep = "\n\n----------\n\n"
    timestamp = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
    fd.write(timestamp)
    fd.write(sep)
    fd.write("\n\n".join(dataset.get_recorded_operations()))
    fd.write(sep)
    fd.write("Optimizer: %s\n" % optimizer.__class__)
    fd.write("Number of samples used in spline interpolation and resampling: %d\n" % NUM_SAMPLES)
    fd.write("Batch Size: %d\n" % PARAM_BATCH_SIZE)
    fd.write("Number of Epochs: %d\n" % PARAM_NUM_EPOCHS)
    fd.write("\n")
    model.summary(print_fn=lambda x: fd.write(x + '\n'))
    fd.write(sep)
    
with open(os.path.join(checkpoints_dir, "model.txt"), "w") as fd:
    pp = pprint.PrettyPrinter(indent=2, stream=fd)
    pp.pprint(model.get_config())
    fd.write("\n")


#%% Callbacks
# Imports
import tfcallback
from keras.callbacks import ModelCheckpoint

# Checkpoint for saving best models
save_filename = os.path.join(checkpoints_dir, checkpoints_save_prefix + "-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5")
checkpointer = ModelCheckpoint(save_filename, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

# Tensorboard Callback
tensorboard_callback = tfcallback.TB(log_every=1, log_dir=tensorboard_dir, write_graph=False)
tensorboard_callback.write_batch_performance = True


#%% Model Training
# Train Model
fit_params = {
        'x': X_train,
        'y': Y_train,
        'epochs': PARAM_NUM_EPOCHS,
        'verbose': 1,
        'callbacks': [checkpointer, tensorboard_callback],
        'validation_data': (X_valid, Y_valid),
        'batch_size': PARAM_BATCH_SIZE
}
model.fit(**fit_params)
#model.fit(x=X_train, y=Y_train, epochs=30, verbose=1, callbacks=[checkpointer, tensorboard_callback], validation_data=(X_valid, Y_valid), batch_size=BATCH_SIZE)

#%% Model Evaluation
from utils.evaluation import get_evaluation_metrics, get_confusion_matrix
#Evaluate Model
# Test Score
test_score = tuple(model.evaluate(X_test, Y_test))
print("Test Loss: %.3f, Test Acc: %.3f%%" % (test_score[0], test_score[1] * 100))

# Recall, Precision, F1_Score on Validation set
Y_predicted_valid = model.predict_classes(X_valid, verbose=1)
rpf = get_evaluation_metrics(Y_valid, Y_predicted_valid)

# Confusion Matrix
confmat = get_confusion_matrix(Y_valid, Y_predicted_valid, plot=True)
#%% [optional] continue training for 30 more epochs
#continue_params = fit_params.copy()
#continue_params['epochs'] = 60
#continue_params['initial_epoch'] = 30
#model.fit(**continue_params)
