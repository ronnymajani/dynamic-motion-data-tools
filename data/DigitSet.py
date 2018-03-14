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
            return pd.DataFrame(digit, columns=DataSetContract.DigitSet.Frame.columns)
        else:
            return digit
            
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
        return res
    
    def convert_t_to_dt(self):
        """ Converts the time feature from 't' (the time elapsed since the first point in this sequence)
        to 'dt' (the difference between each point and its previous point)"""
        for digit in self.data:
            for i in range(len(digit)-1, 0, -1):
                digit[i][3] -= digit[i-1][3]

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

