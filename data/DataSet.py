# -*- coding: utf-8 -*-

import os
import warnings
import copy
import numpy as np
from utils.decorators import preprocessingFunction
from data.contract import DataSetContract
from data.DigitSet import DigitSet
from sklearn.preprocessing import OneHotEncoder
import functools

#todo: add functions for loading and storing information about dataset
#todo: add functions for loading a unified dataset file instead of many digitsets

class DataSet(object):
    def __init__(self, folder=None):
        self.data = None
        self.labels = None
        self._is_dt = True
        self._applied_operations = []
        if folder is not None:
            self.load(folder)
    
    def load(self, folder):
        """ Load a set of digitsets in the given folder """
        if self.data is not None:
            warnings.warn("Loading a new dataset into a non empty DataSet object")
        files = [os.path.join(folder, file) for file in os.listdir(folder)]
        self.data = []
        self.labels = []
        for file in files:
            digitset = DigitSet(file)
            self.data += digitset.data
            self.labels += digitset.labels
            
    def apply(self, operation):
        """Apply a given digit operation to each digit in the digitset
        This function is for operations that work on individual digits"""
        res = []
        for digit in self.data:
            res.append(operation(digit))
        self.data = res
        # save name of applied operation
        self._record_operation("modify", operation)
        
        
    def expand(self, operation):
        """ Apply a given digit operation to each digit in the dataset and append the result
        to the dataset
        @param operation: a function to apply to each digit. should take one argument, which is
        a single digit, and should return a digit.
        """
        data_len_pre_expand = len(self.data)
        for digit_idx in range(data_len_pre_expand):
            self.data.append(operation(self.data[digit_idx]))
            self.labels.append(self.labels[digit_idx])
            
        # save name of applied operation
        self._record_operation("expand", operation)
            
    def as_numpy(self, mask_value):
        """ Returns the entire digitset as a numpy array in the shape
        [number of samples , maximum sequence length , length of single frame]
        The function also pads sequences less than the maximum sequence length with the given
        masking value.
        """
        max_len = 0
        for digit in self.data:
            max_len = max(max_len, len(digit))
        # create empty numpy array 
        res = np.empty((len(self.data), max_len, len(DataSetContract.DigitSet.Frame.columns)))
        res.fill(mask_value)
        # fill in array
        for i in range(len(self.data)):
            digit = self.data[i]
            res[i, :len(digit), :] = digit
        return res    
    
    def get_labels_as_numpy(self, onehot=False):
        """ Returns the labels as a numpy ndarray
        if onehot is set to True, it will onehot encode the labels using scikit learn's OneHotEncoder
        and will return both the encoder and the encoded labels
        @returns ndarray of labels if onehot is False
        @returns (OneHotEncoder, ndarray of labels) if onehot is True
        """
        labels = np.array(self.labels).reshape(-1, 1)
        if onehot:
            encoder = OneHotEncoder()
            labels = encoder.fit_transform(labels)
            return encoder, labels
        else:
            return labels
            
    
    def copy(self):
        """Returns a copy of this dataset"""
        res = DataSet()
        res.data = copy.copy(self.data)
        res.labels = copy.copy(self.labels)
        res._is_dt = self._is_dt
        return res
    
    @preprocessingFunction("Convert time feature from 'dt' (time difference between points) to total Elapsed Time")
    def convert_dt_to_t(self):
        """ Converts the time feature from 'dt' (the difference between each point and its previous point)
        to 't' (the time elapsed since the first point in this sequence) """
        dt_idx = DataSetContract.DigitSet.Frame.indices['dt']
        for digit in self.data:
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

    def _record_operation(self, operation_type, operation):
        """ Record an operation applied to the DataSet 
        It's expected that any applied operation should have an attribute 'operation_name'
        Otherwise it will be recorded as '[Unknown Operation: `function name`]'
        """
        # If a partial function was passed using functools, extract the actual function that was used to construt the partial
        if operation.__class__ == functools.partial:
            operation = operation.func
            
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
        # Save operation record
        self._applied_operations.append(newstr)



