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
PARAM_NUM_EPOCHS = 15
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
dataset.load(dataset_folder_path, test_set_percentage=0.3333, validation_set_percentage=0.3333)
dataset.apply(apply_unit_distance_normalization)
dataset.apply(apply_first_frame_centering)
dataset.apply(partial(spline_interpolate_and_resample, num_samples=NUM_SAMPLES))
dataset.expand(reverse_digit_sequence)

#%%
# Create generative dataset
from data.DataSetManipulator import DataSetManipulator

manip = DataSetManipulator(dataset, sequence_length=NUM_SAMPLES)
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = manip.create_dataset_for_generative_models()

#%% Build Model
from models.gen_regularized_64_gru import GenRegularized64GRU

model = GenRegularized64GRU(X_train.shape[1:], manip._maskingValue)
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
print("Test Loss: %.3f, Test Acc: %.3f%%" % (test_score[0], test_score[1] * 100))

#%% Generate a given number
import numpy as np
from utils.plot import show_digit
from keras.preprocessing.sequence import pad_sequences

predict_number = 0
gen_digit = np.array([[predict_number, predict_number]])

for i in range(1, NUM_SAMPLES):
    x = np.array([gen_digit])
    x = pad_sequences(x, maxlen=NUM_SAMPLES, dtype='float32', padding='post', truncating='post', value=manip._maskingValue)
    predicted_y = model.model.predict(x)
    gen_digit = np.vstack((gen_digit, predicted_y))

show_digit(gen_digit[1:])


