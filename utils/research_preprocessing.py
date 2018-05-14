#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from data.contract import DataSetContract
from utils.decorators import preprocessingOperation


@preprocessingOperation("Add Occlusions")
def add_occlusions(digit, dropout_percentage, min_len=5):
    """ Adds occlusions to the given digit (drops random frames)
    @param digit: The digit to apply the occlusion to
    @param dropout_percentage: the percentage of the frames to dropout
    @param min_len: the minimum length that the resulting digit should be.
        The function won't apply occlusion to a digit if the result is smaller than
        the specified minimum.
    """
    if not isinstance(digit, np.ndarray):
        digit = np.array(digit)
  
    occlusion_idx = int(round((len(digit) * (1 - dropout_percentage))))
    if occlusion_idx < min_len:
        occlusion_idx = min_len
    
    new_dig_idx = np.random.permutation(len(digit))
    new_dig_idx = new_dig_idx[:occlusion_idx]
    new_dig_idx.sort()
    new_dig = digit[new_dig_idx]

    return new_dig


@preprocessingOperation("Add Noise")
def add_noise(digit, mean=0, std=0.1):
    """ Adds noise to the given digit by drawing random values from a normal distribution
    @param digit: The digit to apply the occlusion to
    @param mean: The mean of the normal distribution noise is generated from
    @param std: The standard deviation of the normal distribution noise is generated from
    """
    if not isinstance(digit, np.ndarray):
        digit = np.array(digit)
        
    noise = np.random.normal(mean, std, digit.shape)
    return digit + noise
    



