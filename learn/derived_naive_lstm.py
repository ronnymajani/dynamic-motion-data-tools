# -*- coding: utf-8 -*-
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

dataset = DataSet()
dataset.load(dataset_folder_path, test_set_percentage=0.333, validation_set_percentage=0)
dataset.apply(apply_mean_centering)
dataset.apply(apply_unit_distance_normalization)
#dataset.apply(partial(normalize_pressure_value, max_pressure_val=512))
dataset.apply(partial(spline_interpolate_and_resample, num_samples=NUM_SAMPLES))
dataset.expand(reverse_digit_sequence)
# dataset.apply(lambda digit: convert_xy_to_derivative(digit, normalize=False))
#dataset.apply(partial(convert_xy_to_derivative, normalize=True))

#%% Split Train, Valid, Test
# Imports
import numpy as np
from sklearn.model_selection import train_test_split

X_train_valid = np.array(dataset.train_data)
X_test = np.array(dataset.test_data)
# Convert labels to numpy array and OneHot encode them
encoder, train_valid_labels, _, Y_test = dataset.onehot_encode_labels()
train_valid_labels = train_valid_labels.astype('float32').todense()
Y_test = Y_test.astype('float32').todense()
# Split Data
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, train_valid_labels, shuffle=True, stratify=train_valid_labels, random_state=42)
#X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, shuffle=True, stratify=Y_train_valid, random_state=42)


#%% Model Training
from models.naive_gru import NaiveGRU

mymodel = NaiveGRU(X_train.shape[1:])
mymodel.batch_size = PARAM_BATCH_SIZE
mymodel.num_epochs = PARAM_NUM_EPOCHS
mymodel.initialize()
print(mymodel)

#%% Save Model Summary
mymodel.save_summary(dataset.get_recorded_operations())
mymodel.save_config()

#%% Train Model
mymodel.train(X_train, Y_train, X_valid, Y_valid)

#%% Model Evaluation
from utils.evaluation import get_evaluation_metrics, get_confusion_matrix
#Evaluate Model
# Test Score
test_score = tuple(mymodel.evaluate(X_test, Y_test))
print("Test Loss: %.3f, Test Acc: %.3f%%" % (test_score[0], test_score[1] * 100))

# Recall, Precision, F1_Score on Validation set
Y_predicted_valid = mymodel.predict_classes(X_valid, verbose=1)
rpf = get_evaluation_metrics(Y_valid, Y_predicted_valid)
print(rpf)

# Confusion Matrix
confmat = get_confusion_matrix(Y_valid, Y_predicted_valid, plot=True)
#%% [optional] continue training for 30 more epochs
#continue_params = fit_params.copy()
#continue_params['epochs'] = 60
#continue_params['initial_epoch'] = 30
#model.fit(**continue_params)
