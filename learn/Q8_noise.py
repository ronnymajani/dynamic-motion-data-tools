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

#%%
NUM_SAMPLES = 50
ANGLES_TO_ROTATE = [5, 10, 15, 45, -5, -10, -15, -45]

from utils.preprocessing import *
from functools import partial

#%% Load Model
from keras.models import load_model

TRAINED_MODEL = os.path.join("files", "checkpoints", "1525696834.4091375", "regularized_3x512_gru-30-0.97.hdf5")
model = load_model(TRAINED_MODEL)

#%%
import numpy as np
from utils.research_preprocessing import add_noise

MEANS_TO_TRY = [0.0, 100, -100, 250, -250]
STD_TO_TRY = np.linspace(0.0, 1000, 20)

scores = {}

for mean in MEANS_TO_TRY:
    scores[mean] = []
    for std in STD_TO_TRY:
        curr = dataset.copy()
        curr.apply(partial(add_noise, mean=mean, std=std))
        curr.apply(apply_mean_centering)
        curr.apply(apply_unit_distance_normalization)
        curr.apply(clean_repeat_points)
        curr.apply(partial(spline_interpolate_and_resample, num_samples=NUM_SAMPLES))
        curr.expand(reverse_digit_sequence)
        X_train = np.array(curr.train_data)
        # Convert labels to numpy array and OneHot encode them
        encoder, Y_train, _, _ = curr.onehot_encode_labels()
        # evaluate
        score = model.evaluate(X_train, Y_train)[1]
        print("mean/std %.2f (+/- %.3f),   acc: %.3f%%" % (mean, std, score*100.0))
        scores[mean].append(score)

import pickle
with open(os.path.join("files", "pickles", "Q8_scores"), 'wb') as fd:
    pickle.dump(scores, fd)
    


#%%
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

drawing_pad_width = 5105.0
drawing_pad_height = 3713.0
drawing_area_root = np.sqrt(drawing_pad_width * drawing_pad_height)
tried_std = ((np.array(STD_TO_TRY) / drawing_area_root) * 100.0)

fig = plt.figure(1, (7,4))
ax = fig.add_subplot(1,1,1)
fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
ax.xaxis.set_major_formatter(xticks)
ax.yaxis.set_major_formatter(xticks)
plt.xlabel("Standard Deviation of Noise sampling")
plt.ylabel("Prediction Accuracy")

for mean in MEANS_TO_TRY:
    percentages = np.array(scores[mean]) * 100.0
    ax.plot(tried_std, percentages, label="%.0f%%"%((mean/drawing_area_root)*100.0))
    ax.scatter(tried_std, percentages)
    
plt.legend(title="Mean")


#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

means = [0.0]
stds = np.linspace(0.0, 1000, 5)

for mean in means:
    for std in stds:
        sigma = std
        x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
        plt.plot(x,mlab.normpdf(x, mean, sigma), label=str(std))
        plt.show()

plt.xlabel("Mean")
plt.ylabel("Value of Noise")
plt.legend(title="Standard Deviation")

