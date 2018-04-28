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
        self.valid_data = None
        self.valid_labels = None
        self.test_data = None
        self.test_labels = None
        self.encoder = None
        self._is_onehot_encoded = False
        self._is_dt = True
        self._applied_operations = []
        if folder is not None:
            self.load(folder)
            
    def reset(self):
        self.train_data = None
        self.train_labels = None
        self.valid_data = None
        self.valid_labels = None
        self.test_data = None
        self.test_labels = None
        self.encoder = None
        self._is_onehot_encoded = False
        self._is_dt = True
        self._applied_operations = []
    
    def load(self, folder, test_set_percentage=0.3333, validation_set_percentage=0.3333):
        """ Load a set of digitsets in the given folder, and split them into train and test sets
        @param[in] test_set_percentage: The percentage of the data that will be assigned to the test set
        @param[in] validation_set_percentage: The percentage of the non test data that will be assigned to the validation set
        @param[optional] random_seed: The seed to use for randomization
        """
        if self.train_data is not None or self.test_data is not None:
            # reset state
            self.reset()
            # warn of reloading data
            warnings.warn("Loading a new dataset into a non empty DataSet object")
        if test_set_percentage >= 1.0:
            raise ValueError("Test Set percentage too high; should be less than 1!")
        elif test_set_percentage < 0:
            raise ValueError("Test Set percentage should be bigger or equal to 0!")
        if validation_set_percentage >= 1.0:
            raise ValueError("Validation Set percentage too high; should be less than 1!")
        elif validation_set_percentage < 0:
            raise ValueError("Validation Set percentage should be bigger than or equal to 0!")
        
        files = [os.path.join(folder, file) for file in os.listdir(folder)]
        # split files into train, valid and test sets
        files = np.random.permutation(files)
        num_users = len(files)
        test_split_index = round(num_users * (1.0 - test_set_percentage))
        if test_split_index == 0:
            raise ValueError("Test Set percentage too high; no data left for Train set!")
        valid_split_index = round(num_users * (1.0 - validation_set_percentage))
        if valid_split_index == 0:
            raise ValueError("Validation Set percentage too high; no data left for Train set!")
        train_valid_files = files[:test_split_index]
        test_files = files[test_split_index:]
        train_files = train_valid_files[:valid_split_index]
        valid_files = train_valid_files[valid_split_index:]
        # laod the files of each set
        self._load_train_data(train_files)
        self._load_valid_data(valid_files)
        self._load_test_data(test_files)
        
    def _load_train_data(self, files):
        self.train_data = []
        self.train_labels = []
        for file in files:
            digitset = DigitSet(file) 
            self.train_data += digitset.data
            self.train_labels += digitset.labels
            
    def _load_valid_data(self, files):
        self.valid_data = []
        self.valid_labels = []
        for file in files:
            digitset = DigitSet(file) 
            self.valid_data += digitset.data
            self.valid_labels += digitset.labels
    
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
        # Train Set
        res = []
        for digit in self.train_data:
            res.append(operation(digit))
        self.train_data = res
        
        # Validation Set
        res = []
        for digit in self.valid_data:
            res.append(operation(digit))
        self.valid_data = res
        
        # Test Set
        if apply_to_test_set:
            res = []
            for digit in self.test_data:
                res.append(operation(digit))
            self.test_data = res
            
        # save name of applied operation
        optype = "modify:train/valid"
        if apply_to_test_set:
            optype += "|test"
        self._record_operation(optype, operation)
        
        
    def expand(self, operation, apply_to_test_set=True, append_to_end=False):
        """ Apply a given digit operation to each digit in the dataset and append the result
        to the dataset
        @param operation: a function to apply to each digit. should take one argument, which is
          a single digit, and should return a digit.
        @param append_to_end: if True, all generated data will be appended to the end of the training dataset
          else, each generated new digit will be appended right after the original digit that produced it
        """
        # Train Set
        if append_to_end:
            data_len_pre_expand = len(self.train_data)
            for digit_idx in range(data_len_pre_expand):
                self.train_data.append(operation(self.train_data[digit_idx]))
                self.train_labels.append(self.train_labels[digit_idx])
        else:
            new_train_data = []
            new_train_labels = []
            for digit, label in zip(self.train_data, self.train_labels):
                new_train_data.append(digit)
                new_train_data.append(operation(digit))
                new_train_labels.append(label)
                new_train_labels.append(label)
            self.train_data = new_train_data
            self.train_labels = new_train_labels
            
        # Validation Set
        if append_to_end:
            data_len_pre_expand = len(self.valid_data)
            for digit_idx in range(data_len_pre_expand):
                self.valid_data.append(operation(self.valid_data[digit_idx]))
                self.valid_labels.append(self.valid_labels[digit_idx])
        else:
            new_valid_data = []
            new_valid_labels = []
            for digit, label in zip(self.valid_data, self.valid_labels):
                new_valid_data.append(digit)
                new_valid_data.append(operation(digit))
                new_valid_labels.append(label)
                new_valid_labels.append(label)
            self.valid_data = new_valid_data
            self.valid_labels = new_valid_labels    
        
        # Test Set
        if apply_to_test_set:
            if append_to_end:
                data_len_pre_expand = len(self.test_data)
                for digit_idx in range(data_len_pre_expand):
                    self.test_data.append(operation(self.test_data[digit_idx]))
                    self.test_labels.append(self.test_labels[digit_idx])
            else:
                new_test_data = []
                new_test_labels = []
                for digit, label in zip(self.test_data, self.test_labels):
                    new_test_data.append(digit)
                    new_test_data.append(operation(digit))
                    new_test_labels.append(label)
                    new_test_labels.append(label)
                self.test_data = new_test_data
                self.test_labels = new_test_labels
            
        # save name of applied operation
        optype = "expand:train/valid"
        if apply_to_test_set:
            optype += "|test"
        self._record_operation(optype, operation)
    
    def onehot_encode_labels(self):
        """ Onehot encodes this datasets labels using scikit learn's OneHotEncoder
        and will return both the encoder and the encoded labels. If the dataset is already onehot encoded,
        this function will just return the previously onehot encoded labels and their encoder
        @returns (OneHotEncoder, ndarray of train labels, ndarray of validation labels, ndarray of test labels) if onehot is True
        """
        if self._is_onehot_encoded:
            warnings.warn("The labels of this dataset are already onehot encoded!")
        else:
            self._is_onehot_encoded = True
            train_labels = np.array(self.train_labels).reshape(-1, 1)
            valid_labels = np.array(self.valid_labels).reshape(-1, 1)
            test_labels = np.array(self.test_labels).reshape(-1, 1)
            self.encoder = OneHotEncoder()
            self.train_labels = self.encoder.fit_transform(train_labels)
            if len(valid_labels) > 0:
                self.valid_labels = self.encoder.transform(valid_labels)
            if len(test_labels) > 0:
                self.test_labels = self.encoder.transform(test_labels)
        return self.encoder, self.train_labels, self.valid_labels, self.test_labels
    
    def labels_are_onehot_encoded(self):
        """ Returns True if the labels of this dataset have been OneHot encoded """
        return self._is_onehot_encoded        
    
    def copy(self):
        """Returns a copy of this dataset"""
        res = DataSet()
        res.train_data = copy.copy(self.train_data)
        res.train_labels = copy.copy(self.train_labels)
        res.valid_data = copy.copy(self.valid_data)
        res.valid_labels = copy.copy(self.valid_labels)
        res.test_data = copy.copy(self.test_data)
        res.test_labels = copy.copy(self.test_labels)
        res.encoder = copy.copy(self.encoder)
        res.encoder = self._is_onehot_encoded
        res.applied_operations = copy.copy(self._applied_operations)
        res._is_dt = self._is_dt
        return res
    
    @preprocessingOperation("Convert time feature from 'dt' (time difference between points) to total Elapsed Time")
    def convert_dt_to_t(self, apply_to_test_set=True):
        """ Converts the time feature from 'dt' (the difference between each point and its previous point)
        to 't' (the time elapsed since the first point in this sequence) """
        dt_idx = DataSetContract.DigitSet.Frame.indices['dt']
        
        # Train Set
        for digit in self.train_data:
            for i in range(1, len(digit)):
                digit[i][dt_idx] += digit[i-1][dt_idx]
                
        # Validation Set
        for digit in self.valid_data:
            for i in range(1, len(digit)):
                digit[i][dt_idx] += digit[i-1][dt_idx]
                
        # Test Set
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



