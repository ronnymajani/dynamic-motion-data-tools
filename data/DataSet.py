# -*- coding: utf-8 -*-

import os
import warnings
import copy
import numpy as np
from utils.decorators import preprocessingOperation
from data.contract import DataSetContract
from data.DigitSet import DigitSet
from sklearn.preprocessing import OneHotEncoder
import functools

#todo: add functions for loading and storing information about dataset
#todo: add functions for loading a unified dataset file instead of many digitsets

class DataSet(object):
    def __init__(self, folder=None):
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self._is_dt = True
        self._applied_operations = []
        if folder is not None:
            self.load(folder)
    
    def load(self, folder, test_set_percentage=0.3333):
        """ Load a set of digitsets in the given folder, and split them into train and test sets
        @param[in] test_set_percentage: The percentage of the data that will be assigned to the test set
        @param[optional] random_seed: The seed to use for randomization
        """
        if self.train_data is not None or self.test_data is not None:
            warnings.warn("Loading a new dataset into a non empty DataSet object")
        if test_set_percentage >= 1.0:
            raise ValueError("Test Set percentage too high!")
        files = [os.path.join(folder, file) for file in os.listdir(folder)]
        # split files into train and test
        files = np.random.permutation(files)
        num_users = len(files)
        split_index = round(num_users * (1.0 - test_set_percentage))
        train_files = files[:split_index]
        test_files = files[split_index:]
        self._load_train_data(train_files)
        self._load_test_data(test_files)
        
    def _load_train_data(self, files):
        self.train_data = []
        self.train_labels = []
        for file in files:
            digitset = DigitSet(file) 
            self.train_data += digitset.data
            self.train_labels += digitset.labels
    
    def _load_test_data(self, files):
        self.test_data = []
        self.test_labels = []
        for file in files:
            digitset = DigitSet(file) 
            self.test_data += digitset.data
            self.test_labels += digitset.labels
            
    def apply(self, operation, apply_to_test_set=True):
        """Apply a given digit operation to each digit in the digitset
        This function is for operations that work on individual digits
        @param[optional] apply_to_test_set: if True, the given operation will also be applied to the test set
        """
        res = []
        for digit in self.train_data:
            res.append(operation(digit))
        self.train_data = res
        
        if apply_to_test_set:
            res = []
            for digit in self.test_data:
                res.append(operation(digit))
            self.test_data = res
            
        # save name of applied operation
        optype = "modify:train"
        if apply_to_test_set:
            optype += "|test"
        self._record_operation(optype, operation)
        
        
    def expand(self, operation, apply_to_test_set=True):
        """ Apply a given digit operation to each digit in the dataset and append the result
        to the dataset
        @param operation: a function to apply to each digit. should take one argument, which is
        a single digit, and should return a digit.
        """
        data_len_pre_expand = len(self.train_data)
        for digit_idx in range(data_len_pre_expand):
            self.train_data.append(operation(self.train_data[digit_idx]))
            self.train_labels.append(self.train_labels[digit_idx])
            
        if apply_to_test_set:
            data_len_pre_expand = len(self.test_data)
            for digit_idx in range(data_len_pre_expand):
                self.test_data.append(operation(self.test_data[digit_idx]))
                self.test_labels.append(self.test_labels[digit_idx])
            
        # save name of applied operation
        optype = "expand:train"
        if apply_to_test_set:
            optype += "|test"
        self._record_operation(optype, operation)
    
    def get_labels_as_numpy(self, onehot=False):
        """ Returns the labels as a numpy ndarray
        if onehot is set to True, it will onehot encode the labels using scikit learn's OneHotEncoder
        and will return both the encoder and the encoded labels
        @returns (ndarray of train labels, ndarray of test labels) if onehot is False
        @returns (OneHotEncoder, ndarray of train labels, ndarray of test labels) if onehot is True
        """
        train_labels = np.array(self.train_labels).reshape(-1, 1)
        test_labels = np.array(self.test_labels).reshape(-1, 1)
        if onehot:
            encoder = OneHotEncoder()
            train_labels = encoder.fit_transform(train_labels)
            test_labels = encoder.fit_transform(test_labels)
            return encoder, train_labels, test_labels
        else:
            return train_labels, test_labels
            
    
    def copy(self):
        """Returns a copy of this dataset"""
        res = DataSet()
        res.train_data = copy.copy(self.train_data)
        res.test_data = copy.copy(self.test_data)
        res.train_labels = copy.copy(self.train_labels)
        res.test_labels = copy.copy(self.test_labels)
        res._is_dt = self._is_dt
        return res
    
    @preprocessingOperation("Convert time feature from 'dt' (time difference between points) to total Elapsed Time")
    def convert_dt_to_t(self, apply_to_test_set=True):
        """ Converts the time feature from 'dt' (the difference between each point and its previous point)
        to 't' (the time elapsed since the first point in this sequence) """
        dt_idx = DataSetContract.DigitSet.Frame.indices['dt']
        
        for digit in self.train_data:
            for i in range(1, len(digit)):
                digit[i][dt_idx] += digit[i-1][dt_idx]
                
        for digit in self.test_data:
            for i in range(1, len(digit)):
                digit[i][dt_idx] += digit[i-1][dt_idx]
                
        self._is_dt = False
        self._record_operation(self.convert_dt_to_t)
        return self
    
    def time_is_dt(self):
        """
        @returns True if the time feature of the digits is measured in 'dt' (the difference between each point and its previous point)
        @returns False if the time feature of the digits is measured in 't' (the time elapsed since the first point in this sequence)
        """
        return self._is_dt
    
    def get_recorded_operations(self):
        """ Returns a list of descriptions of the operations applied so far on this dataset """
        return self._applied_operations.copy()

    def _record_operation(self, operation_type, operation, info=None):
        """ Record an operation applied to the DataSet 
        It's expected that any applied operation should have an attribute 'operation_name'
        Otherwise it will be recorded as '[Unknown Operation: `function name`]'
        """
        misc = ""
            
        # If a partial function was passed using functools, extract the actual function that was used to construt the partial
        if operation.__class__ == functools.partial:
            misc += "\n\t>> args: " + str(operation.args)
            misc += "\n\t>> keywords: " + str(operation.keywords)
            operation = operation.func
        
        if info is not None:
            misc += "\n\t>>> info: " + info
            
        newstr = ""
        # Operation Assigned Type
        newstr += "* [%s]" % operation_type
        # Operation Function's Name
        newstr += " (%s):\n\t" % operation.__name__
        # Operation's Assigned Name
        try:
            newstr += "%s" % operation.operation_name
        except AttributeError:
            newstr += "Unknown Operation!"
        newstr += misc
        # Save operation record
        self._applied_operations.append(newstr)



