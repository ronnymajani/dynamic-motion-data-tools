#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from data.contract import DataSetContract
from utils.decorators import preprocessingOperation


@preprocessingOperation("Add Occlusions")
def add_occlusions(digit, dropout_percentage):
    """ Adds occlusions to the given digit (drops random frames)
    @param digit: The digit to apply the occlusion to
    @param dropout_percentage: the percentage of the frames to dropout
    """
    if not isinstance(digit, np.ndarray):
        digit = np.array(digit)
    
    occlusion_idx = round(len(digit) * (1 - dropout_percentage))
    new_dig_idx = np.random.permutation(len(digit))
    new_dig_idx = new_dig_idx[:occlusion_idx]
    new_dig_idx.sort()
    new_dig = digit[new_dig_idx]
    
    return new_dig


@preprocessingOperation("Add Noise")
def add_noise(digit, mean=0, std_deviation=0.1):
    """ Adds noise to the given digit by drawing random values from a normal distribution
    @param digit: The digit to apply the occlusion to
    @param mean: The mean of the normal distribution noise is generated from
    @param std_deviation: The standard deviation of the normal distribution noise is generated from
    """
    if not isinstance(digit, np.ndarray):
        digit = np.array(digit)
        
    noise = np.random.normal(mean, std_deviation, digit.shape)
    return digit + noise
    



