# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import utils.plot
import utils.support
from models.model_template import ModelTemplate
import copy

def get_confusion_matrix(Y_true, Y_predicted, y_true_is_one_hot=True, y_predicted_is_one_hot=False, plot=False, as_percentage=True):
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
    
    if as_percentage:
        confmat = (confmat / confmat.sum(axis=1).reshape(-1, 1)) * 100.0
    
    if plot:
        utils.plot.show_mat(confmat, xlabel="True", ylabel="Predicted", title="Confusion Matrix", 
                            show_grid=True, show_colorbar=True, uniform_ticks=True, hide_ticks=True, 
                            show_vals=True, show_vals_as_int=False)
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

def cross_validate_model(X, Y, model, n_folds, random_state=None):
    """ Evaluate a given model using crossvalidation 
    @param[in] X: data to split into test/valid sets
    @param[in] Y: labels to split into test/valid sets. They should NOT be onehot encoded
    @param[in] model: An uninitialized model object that implements the ModelTemplate class from models/model_template.py
    @param[in] n_folds: Number of K-folds to split the training_valid data into K different train/valid splits
    @param[in] random_state: The random state to be used when shuffling and permutating the data
    @returns list of accuracies of n cross folds evaluated after training
    """
    if not isinstance(model, ModelTemplate):
        raise ValueError("the model argument must be an instance of ModelTemplate!")

    np.random.seed(random_state)
    encoder = OneHotEncoder()
    encoder.fit(Y.reshape(-1, 1))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=random_state)
    
    # we manually permutate and shuffle the data since we don't want to separate user samples from eachother
    # meaning that all the samples produced by a user should be next to eachother,
    # the reason for this is so when we split the data we can ensure that each user is contained within only one set
    # either the train set or the test set, not both. Although the order with which a users samples appear relative to eachother
    # can be random,what matters is only that all those samples are consecutive (eg: user 1's data samples are all the data points
    # between index 200 and 299 inclusive, user 8's data is all the points from index 300 to 399, etc.)
    if len(X)//100 != len(X)/100:
        raise ValueError("Dataset is corrupt, not all users have produced the same number of samples!")
    random_indices = utils.support.get_random_indices_for_dataset(len(X)//100, 100, shuffle=True)
    X = X[random_indices]
    Y = Y[random_indices]
    
    cvscores = []
    i = 1    
    for train, valid in kfold.split(X, Y):
        print("\n....................\nCross validation fold [%d]\n....................\n" % i)
        model_copy = copy.deepcopy(model)
        model_copy.disable_callbacks()
        model_copy.initialize()
        
        train_labels = encoder.transform(Y[train].reshape(-1, 1))
        valid_labels = encoder.transform(Y[valid].reshape(-1, 1))
        
        # even though we are passing thevalidation set to the training function
        # it is not being used a swe disabled callback
        model_copy.train(X[train], train_labels, X[valid], valid_labels)
        scores = model_copy.evaluate(x=X[valid], y=valid_labels)
        print("%s: %.2f%%" % (model_copy.model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        i += 1
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return cvscores
    
    
    
    
