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
import matplotlib.pyplot as plt
import numpy as np
DIGIT_IDX = 1627
#DIGIT_IDX = 326
digit = dataset.train_data[DIGIT_IDX]

#%% create original digit images
padding = 10
frame_len = len(digit[0])
data = np.array(digit, dtype=np.float32).reshape(-1, frame_len)
x = data[:, 0]
y = data[:, 1]
x = x.flatten()
y = y.flatten()
y_max = y.max() + padding
y_min = y.min() - padding
x_max = x.max() + padding
x_min = x.min() - padding  
c = np.arange(len(x))

for i in range(len(x)):
    ax = plt.gca()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    # Preserve the aspect ratio and scale between the X and Y axis
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.scatter(x[:i], y[:i], c=c[:i], cmap="inferno", alpha=0.43)
    plt.scatter(x[i], y[i], c='red', alpha=1.0)
    plt.savefig("files/figs/%d.png"%i)
    plt.clf()
    
#%% preprocessing graphics
from utils.plot import show_digit
from utils.preprocessing import *
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

original_digit = digit
show_digit(digit, padding=10, show_lines=False)
plt.savefig("files/figs/pp_org.pdf")
plt.clf()

mean_centered_digit = apply_mean_centering(digit)
show_digit(mean_centered_digit, padding=10, show_lines=False)        
plt.axhline(0, xmin=-1000, xmax=1000, color='grey', alpha=0.1)
plt.axvline(0, ymin=-1000, ymax=1000, color='grey', alpha=0.1)
plt.savefig("files/figs/pp_mean_centering.pdf")
plt.clf()

unit_digit = apply_unit_distance_normalization(mean_centered_digit)
show_digit(unit_digit, padding=0.03, show_lines=False)
plt.axhline(0, xmin=-2, xmax=2, color='grey', alpha=0.1)
plt.axvline(0, ymin=-2, ymax=2, color='grey', alpha=0.1)
plt.savefig("files/figs/pp_unit.pdf")
plt.clf()

NUM_SAMPLES = 50
spline_digit = spline_interpolate_and_resample(unit_digit, num_samples=NUM_SAMPLES)
show_digit(spline_digit, padding=0.03, show_lines=False)
plt.axhline(0, xmin=-2, xmax=2, color='grey', alpha=0.1)
plt.axvline(0, ymin=-2, ymax=2, color='grey', alpha=0.1)
plt.savefig("files/figs/pp_spline.pdf")
plt.clf()

#%% augmentation graphics
reverse_digit = reverse_digit_sequence(spline_digit)
show_digit(reverse_digit, padding=0.03, show_lines=False)
plt.savefig("files/figs/ag_reverse.pdf")
plt.clf()

ANGLES_TO_ROTATE = [10, -20, -45]
rotated_digits = rotate_digit(spline_digit, degrees=ANGLES_TO_ROTATE)
for i, rd in enumerate(rotated_digits):
    show_digit(rd, padding=0.03, show_lines=False)
    plt.savefig("files/figs/ag_rotate_%d.pdf"%i)
    plt.clf()


    