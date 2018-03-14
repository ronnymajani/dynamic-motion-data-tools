#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
from contract import DataSetContract

#%%

def load_digitset(filename):
    """ Loads the JSON file of the digitset specified by the given filename """
    with open(filename) as fp:
        return json.load(fp)


def extract_data_and_labels(digitset):
    """ Extracts and returns the Data and it's corresponding Labels from the given digitset JSON file
    @returns data: The raw sequence data where each row corresponds to a sample
    @returns labels: The corresponding labels of the raw data
    """
    digitset = digitset[DataSetContract.DigitSets.DIGITS]
    data = []
    labels = []
    for digit_idx in DataSetContract.DigitSets.Digits:
        for sample in digitset[digit_idx]:
            data.append(sample)
            labels.append(int(digit_idx))
    return data, labels


