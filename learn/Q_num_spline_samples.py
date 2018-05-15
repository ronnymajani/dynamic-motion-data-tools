# -*- coding: utf-8 -*-

# allow the notebook to access the parent directory so we can import the other modules
# https://stackoverflow.com/a/35273613
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
    
dataset_folder_path = os.path.join("files", "dataset")

#%%
from data.DataSet import DataSet
dataset = DataSet()
dataset.load(dataset_folder_path, test_set_percentage=0.2, validation_set_percentage=0.3333)

print("Training Data Len:", len(dataset.train_data))
print("Validation Data Len:", len(dataset.valid_data))
print("Test Data Len:", len(dataset.test_data))

#%% Load Model
from keras.models import load_model

TRAINED_MODEL = os.path.join("files", "checkpoints", "1525696834.4091375", "regularized_3x512_gru-30-0.97.hdf5")
model = load_model(TRAINED_MODEL)

#%%
import numpy as np
from utils.preprocessing import *
from functools import partial
from utils.research_preprocessing import add_occlusions
from models.regularized_3x512_gru import Regularized3x512GRU
import os.path
import pickle

dataset.apply(apply_mean_centering)
dataset.apply(apply_unit_distance_normalization)

ANGLES_TO_ROTATE = [5, 10, 15, 45, -5, -10, -15, -45]

NUM_EPOCHS = 20
PARAM_BATCH_SIZE = 500

NUM_SAMPLES_TO_TRY = [10, 25, 50, 75, 100, 150, 200, 300]

scores_valid = []
scores_test = []


for num_samples in NUM_SAMPLES_TO_TRY:
    curr = dataset.copy()
    curr.apply(partial(spline_interpolate_and_resample, num_samples=num_samples))
    curr.expand(reverse_digit_sequence)
    
    X_train = np.array(dataset.train_data)
    X_valid = np.array(dataset.valid_data)
    X_test = np.array(dataset.test_data)
    # Convert labels to numpy array and OneHot encode them
    encoder, Y_train, Y_valid, Y_test = dataset.onehot_encode_labels()
    # evaluate
    
    mymodel = Regularized3x512GRU(X_train.shape[1:])
    mymodel.batch_size = PARAM_BATCH_SIZE
    mymodel.num_epochs = NUM_EPOCHS
    mymodel.disable_callbacks()
    mymodel.initialize()
    mymodel.train(X_train, Y_train, X_valid, Y_valid)
    
    score_valid = mymodel.model.evaluate(X_valid, Y_valid)[1] * 100.0
    score_test = mymodel.model.evaluate(X_test, Y_test)[1] * 100.0
    print("num samples: %d,   valid acc: %.3f%%,   test acc: %.3f%%" % (num_samples, score_valid, score_test))
    scores_valid.append(score_valid)
    scores_test.append(score_test)
    
    with open(os.path.join("files", "pickles", "Q_spline_num_samples.pkl"), 'wb') as fd:
        pickle.dump([scores_valid, scores_test], fd)
    




#%%
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

fig = plt.figure(1, (7,4))
ax = fig.add_subplot(1,1,1)
ax.plot(NUM_SAMPLES_TO_TRY, scores_valid, c='blue', label="Validation Set")
ax.scatter(NUM_SAMPLES_TO_TRY, scores_valid, c='blue')

ax.plot(NUM_SAMPLES_TO_TRY, scores_test, c='red', label="Test Set")
ax.scatter(NUM_SAMPLES_TO_TRY, scores_test, c='red')

fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(xticks)
plt.xlabel("Number of Frames to Sample from Spline")
plt.ylabel("Prediction Accuracy")
#plt.legend()

