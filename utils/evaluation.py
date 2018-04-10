# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import utils.plot

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
        utils.plot.show_mat(confmat, xlabel="True", ylabel="Predicted", title="Confusion Matrix", 
                            show_grid=True, show_colorbar=True, uniform_ticks=True, hide_ticks=True)
    return confmat
        
    

def get_failed_predictions_indices(Y_true, Y_predicted, y_true_is_one_hot=True, y_predicted_is_one_hot=False):
    """ Returns the indices of the the digits that were misclassified """
    if y_true_is_one_hot:
        Y_true = Y_true.argmax(axis=1)
    if y_predicted_is_one_hot:
        Y_predicted = Y_predicted.argmax(axis=1)
        
    Y_true = Y_true.flatten()
    Y_predicted = Y_predicted.flatten()
        
    return np.ravel(np.not_equal(Y_true, Y_predicted))
    

def get_failed_predictions(X, Y_true, Y_predicted, y_true_is_one_hot=True, y_predicted_is_one_hot=False):
    """
    @returns 3 element ordered tuple:
        1- A list of the failed digits
        2- A list of the failed digits' labels
        3- A list of the (wrong) labels the model predicted for these digits
    """
    if y_true_is_one_hot:
        Y_true = Y_true.argmax(axis=1)
    if y_predicted_is_one_hot:
        Y_predicted = Y_predicted.argmax(axis=1)
    
    Y_true = Y_true.reshape((-1, 1))
    Y_predicted = Y_predicted.reshape((-1, 1))
    
    failed_indices = get_failed_predictions_indices(Y_true, Y_predicted, False, False)
    return X[failed_indices], Y_true[failed_indices], Y_predicted[failed_indices]


def get_random_failure(X_fail, Y_fail_true, Y_fail_predicted, plot=True):
    """ Get a random failure from the given list of miscassified digits
    @param[in] X_fail: list of failed digits
    @param[in] Y_fail_true: list of correct labels for the misclassified digits
    @param[in] Y_fail_predicted: list of wrong predictions for the given digits
    @param[optional] plot: If True, the function will plot the failed digit
    @returns the randomly selected misclassified digit, its correct label, and the predicted label 
    """
    rand_idx = np.random.randint(0, X_fail.shape[0])
    x = X_fail[rand_idx]
    y = Y_fail_true[rand_idx]
    predicted = Y_fail_predicted[rand_idx]
    # plot failure
    if plot:
        utils.plot.show_digit(x, label=y, show_lines=False)
    # print information
    print("Label: %d", y)
    print("Predicted: %d", predicted)
    return x, y, predicted
    
def get_evaluation_metrics(Y_true, Y_predicted, y_true_is_one_hot=True, y_predicted_is_one_hot=False):
    if y_true_is_one_hot:
        Y_true = Y_true.argmax(axis=1)
    if y_predicted_is_one_hot:
        Y_predicted = Y_predicted.argmax(axis=1)
        
    vals = np.array(precision_recall_fscore_support(Y_true, Y_predicted))
    return pd.DataFrame(vals.T, columns=["recall", "precision", "f1 score", "#"])
    
