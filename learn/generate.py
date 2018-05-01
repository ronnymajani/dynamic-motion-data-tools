#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 13:02:48 2018

@author: ronnymajani
"""
# allow the script to access the parent directory so we can import the other modules
# https://stackoverflow.com/a/35273613
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

#%%
# Constants
PARAM_NUM_EPOCHS = 40
PARAM_BATCH_SIZE = 300
NUM_SAMPLES = 50

# Paths
dataset_folder_path = os.path.join("files", "dataset")

#%% Prepare Data
# Imports
from utils.preprocessing import *
from data.DataSet import DataSet
from functools import partial

# Preprocessing
dataset = DataSet()
dataset.load(dataset_folder_path, test_set_percentage=0.1, validation_set_percentage=0.1)
dataset.apply(apply_first_frame_centering)
dataset.apply(apply_unit_distance_normalization)
dataset.apply(partial(spline_interpolate_and_resample, num_samples=NUM_SAMPLES))

#%%
# Create generative dataset
from data.DataSetManipulator import DataSetManipulator

manip = DataSetManipulator(dataset, sequence_length=NUM_SAMPLES)
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = manip.create_dataset_for_generative_models()

#%% Build Model
from models.gen_regularized_1024_gru import GenRegularized1024GRU

model = GenRegularized1024GRU(X_train.shape[1:], manip._maskingValue)
model.batch_size = PARAM_BATCH_SIZE
model.num_epochs = PARAM_NUM_EPOCHS
model.initialize()
print(model)

#%% Save Model Summary
model.save_summary(dataset.get_recorded_operations())
model.save_config()

#%% Train Model
model.train(X_train, Y_train, X_valid, Y_valid)

#%% Model Evaluation
# Test Score
test_score = tuple(model.model.evaluate(X_test, Y_test))
print("Test Loss: %.3f, Test MSE: %.3f" % (test_score[0], test_score[1]))

#%%
from keras.models import load_model

model.model = load_model("files/checkpoints/1525180976.0079334/gen_regularized_1024_gru-02-0.00.hdf5")

#%% Generate a given number
import numpy as np
from utils.plot import show_digit
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

predict_number = 0

gen_digit = np.array([[predict_number, predict_number], [0, 0]])

for i in range(2, NUM_SAMPLES):
    x = np.array([gen_digit])
    x = pad_sequences(x, maxlen=NUM_SAMPLES, dtype='float32', padding='post', truncating='post', value=manip._maskingValue)
    predicted_y = model.model.predict(x)
    gen_digit = np.vstack((gen_digit, predicted_y))

plt.figure()
show_digit(gen_digit[1:])


