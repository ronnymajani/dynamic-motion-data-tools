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

dataset.apply(clean_repeat_points)

NUM_SAMPLES = 50
ANGLES_TO_ROTATE = [5, 10, 15, 45, -5, -10, -15, -45]


DROPOUTS_TO_TRY = np.linspace(0.0, 0.99, 20)

scores = []

for drop in DROPOUTS_TO_TRY:
    curr = dataset.copy()
    curr.apply(partial(add_occlusions, dropout_percentage=drop))
    curr.apply(apply_mean_centering)
    curr.apply(apply_unit_distance_normalization)
    curr.apply(partial(spline_interpolate_and_resample, num_samples=NUM_SAMPLES))
    curr.expand_many(partial(rotate_digit, degrees=ANGLES_TO_ROTATE))
    curr.expand(reverse_digit_sequence)
    X_train = np.array(curr.train_data)
    # Convert labels to numpy array and OneHot encode them
    encoder, Y_train, _, _ = curr.onehot_encode_labels()
    # evaluate
    score = model.evaluate(X_train, Y_train)[1]
    print("occlusion percentage: %.2f%%,   acc: %.3f%%" % (drop*100.0, score*100.0))
    scores.append(score)

import pickle
with open(os.path.join("files", "pickles", "Q4_scores.pkl"), 'wb') as fd:
    pickle.dump(scores, fd)


#%%
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
percentages = np.array(scores) * 100.0
dropouts = np.array(DROPOUTS_TO_TRY) * 100.0

fig = plt.figure(1, (7,4))
ax = fig.add_subplot(1,1,1)
ax.plot(dropouts, percentages, c='orange', label="36 Users")
ax.scatter(dropouts, percentages, c='orange')
fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
ax.xaxis.set_major_formatter(xticks)
ax.yaxis.set_major_formatter(xticks)
plt.xlabel("Occlusion Percentage")
plt.ylabel("Prediction Accuracy")
#plt.legend()

