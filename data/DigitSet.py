#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import copy
import json
from contract import DataSetContract
import warnings

class DigitSet():
    def __init__(self, filename=None):
        self.name = None
        self.json = None
        self.data = None
        self.labels = None
        self._is_dt = True  # specifies whether the time feature is dt or elapsed time since begin of sequence
        
        if filename is not None:
            self.load(filename)
            
    def __getitem__(self, key):
        return self.data[key], self.labels[key]
    
    def get(self, digit_index, as_data_frame=False):
        """ Retrieve the digit at the given index
        @param digit_index: the index of the digit to retrieve 
        @param as_data_frame: if True we return the digit as a pandas dataframe
        @returns the digit located at the given index in this digitset's data"""
        digit = self.data[digit_index]
        if as_data_frame:
            cols = DataSetContract.DigitSet.Frame.columns
            if not self.time_is_dt():
                dt_idx = DataSetContract.DigitSet.Frame.indices['dt']
                cols = list(cols)
                cols[dt_idx] = 't'
            return pd.DataFrame(digit, columns=cols)
        else:
            return digit
        
    def as_numpy(self, mask_value):
        """ Returns the entire digitset as a numpy array in the shape
        [number of samples , maximum sequence length , length of single frame]
        The function also pads sequences less than the maximum sequence length with the given
        masking value.
        """
        # first find max sequence length
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
            
    def apply(self, operation):
        """Apply a given digit operation to each digit in the digitset
        This function is for operations that work on individual digits"""
        res = []
        for digit in self.data:
            res.append(operation(digit))
        self.data = res
            
    def load(self, filename):
        """Load digitset data from the given file"""
        if self.name is not None:
            warnings.warn("Loading a new digitset file into a non empty DigitSet object")
        self.name = filename
        self.json = DigitSet.load_json(filename)
        self.data, self.labels = DigitSet.extract_data_and_labels(self.json)
        
    def copy(self):
        """Returns a copy of this digitset"""
        res = DigitSet()
        res.name = self.name
        res.json = copy.deepcopy(self.json)
        res.data = copy.copy(self.data)
        res.labels = copy.copy(self.labels)
        res._is_dt = self._is_dt
        return res
    
    def convert_dt_to_t(self):
        """ Converts the time feature from 'dt' (the difference between each point and its previous point)
        to 't' (the time elapsed since the first point in this sequence) """
        dt_idx = DataSetContract.DigitSet.Frame.indices['dt']
        for digit in self.data:
            for i in range(1, len(digit)):
                digit[i][dt_idx] += digit[i-1][dt_idx]
        self._is_dt = False
        return self
    
    def time_is_dt(self):
        """
        @returns True if the time feature of the digits is measured in 'dt' (the difference between each point and its previous point)
        @returns False if the time feature of the digits is measured in 't' (the time elapsed since the first point in this sequence)
        """
        return self._is_dt

    @staticmethod
    def load_json(filename):
        """ Loads the JSON file of the digitset specified by the given filename """
        with open(filename) as fp:
            return json.load(fp)

    @staticmethod
    def extract_data_and_labels(digitset_json):
        """ Extracts and returns the Data and it's corresponding Labels from the given digitset JSON file
        @returns data: The raw sequence data where each row corresponds to a sample
        @returns labels: The corresponding labels of the raw data
        """
        digitset = digitset_json[DataSetContract.DigitSet.DIGITS]
        data = []
        labels = []
        for digit_idx in DataSetContract.DigitSet.digits:
            for sample in digitset[digit_idx]:
                data.append(np.array(sample))
                labels.append(int(digit_idx))
        return data, labels

