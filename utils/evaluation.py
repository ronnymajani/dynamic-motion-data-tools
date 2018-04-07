# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(Y_true, Y_predicted, y_true_is_one_hot=True, y_predicted_is_one_hot=False, plot=False):
    """ Calculates the confusion matrix for the given model and data.
    @param[in] Y_true: the correct "true" labels
    @param[in] Y_predicted: the labels predicted by the model 
    @param[optional] y_true_is_one_hot: if True, treat Y_true as onehot encoded, and decode it.
    @param[optional] y_predicted_is_one_hot: if True, treat Y_predicted as onehot encoded, and decode it.
    @param[optional] plot: if True, plot the confusion matrix.
    @returns the confusion matrix
    """
    if y_true_is_one_hot:
        Y_true = Y_true.argmax(axis=1)
    if y_predicted_is_one_hot:
        Y_predicted = Y_predicted.argmax(axis=1)
    
    confmat = confusion_matrix(Y_true, Y_predicted)
    
    if plot:
        plt.matshow(confmat, cmap=matplotlib.cm.seismic)
        
    return confmat
        
    
