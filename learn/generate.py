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
dataset.apply(apply_mean_centering)
dataset.apply(apply_unit_distance_normalization)
dataset.apply(partial(spline_interpolate_and_resample, num_samples=NUM_SAMPLES))
dataset.expand(reverse_digit_sequence)

#%%
# Create generative dataset
from data.DataSetManipulator import DataSetManipulator

manip = DataSetManipulator(dataset, sequence_length=NUM_SAMPLES)
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = manip.create_dataset_for_generative_models()

#%% Model

