# -*- coding: utf-8 -*-

# allow the notebook to access the parent directory so we can import the other modules
# https://stackoverflow.com/a/35273613
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
    
import os
dataset_folder_path = os.path.join("files", "dataset")

#%%
from data.DataSet import DataSet
dataset = DataSet()
dataset.load(dataset_folder_path, test_set_percentage=0.2, validation_set_percentage=0.3333)

print("Training Data Len:", len(dataset.train_data))
print("Validation Data Len:", len(dataset.valid_data))
print("Test Data Len:", len(dataset.test_data))

#%%
NUM_SAMPLES = 50
ANGLES_TO_ROTATE = [5, 10, 15, 45, -5, -10, -15, -45]

from utils.preprocessing import *
from functools import partial
dataset.apply(apply_mean_centering)
#dataset.apply(apply_unit_distance_normalization)
dataset.apply(partial(spline_interpolate_and_resample, num_samples=NUM_SAMPLES))
dataset.expand_many(partial(rotate_digit, degrees=ANGLES_TO_ROTATE))
dataset.expand(reverse_digit_sequence)

print("Training Data Len:", len(dataset.train_data))
print("Validation Data Len:", len(dataset.valid_data))
print("Test Data Len:", len(dataset.test_data))

#%%
import numpy as np

X_train = np.array(dataset.train_data)
X_valid = np.array(dataset.valid_data)
X_test = np.array(dataset.test_data)

# Convert labels to numpy array and OneHot encode them
encoder, Y_train, Y_valid, Y_test = dataset.onehot_encode_labels()

print("Training Data Shape:", X_train.shape)
print("Training Labels Shape:", Y_train.shape)
print("Validation Data Shape:", X_valid.shape)
print("Validation Labels Shape:", Y_valid.shape)
print("Test Data Shape:", X_test.shape)
print("Test Labels Shape:", Y_test.shape)

#%%
PARAM_NUM_EPOCHS = 70
PARAM_BATCH_SIZE = 500

from models.regularized_3x512_gru import Regularized3x512GRU
import os.path

mymodel = Regularized3x512GRU(X_train.shape[1:])
mymodel.batch_size = PARAM_BATCH_SIZE
mymodel.num_epochs = PARAM_NUM_EPOCHS
mymodel.initialize()
print(mymodel)
mymodel.save_summary(dataset.get_recorded_operations())
mymodel.save_config()

#%%
mymodel.train(X_train, Y_train, X_valid, Y_valid)

#%% Evaluation
test_score = tuple(mymodel.model.evaluate(X_test, Y_test))
print("Test Loss: %.3f, Test Acc: %.3f%%" % (test_score[0], test_score[1] * 100))

#%% Model Evaluation
from utils.evaluation import get_evaluation_metrics, get_confusion_matrix
# Recall, Precision, F1_Score on Validation set
Y_predicted_valid = mymodel.model.predict_classes(X_valid, verbose=1)
rpf = get_evaluation_metrics(Y_valid, Y_predicted_valid)
print(rpf)

# Confusion Matrix
confmat = get_confusion_matrix(Y_valid, Y_predicted_valid, plot=True)
