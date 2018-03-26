#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from data.contract import DataSetContract


def normalize_pressure_value(digit, max_pressure_val=512, inplace=False):
    """ Normalizes the pressure value to the range [0.0, 1.0] given the maximum pressure value possible """
    p_idx = DataSetContract.DigitSet.Frame.indices['P']
    if not isinstance(digit, np.ndarray) or inplace == False:
        digit = np.array(digit)
    digit[:, p_idx] /= max_pressure_val
    return digit


def apply_mean_centering(digit, inplace=False):
    """ Translates the coordinates X, Y so their mean is aligned with the origin (0,0)
    @param digit: The digit to apply the transformationt to
    @param inplace: If True, the operation is performed inplace
                    (this only works if digit is already a numpy array)
                    If False, a copy of the given data is made
    @returns The resulting digit as a copy if inplace is False
    @returns The given digit itself after applying the transformation
    """
    if not isinstance(digit, np.ndarray) or inplace == False:
        digit = np.array(digit)
        
    x_idx = DataSetContract.DigitSet.Frame.indices['X']
    y_idx = DataSetContract.DigitSet.Frame.indices['Y']
    
    x = digit[:, x_idx]
    y = digit[:, y_idx]
    
    mean_x, mean_y = x.mean(), y.mean()
    
    digit[:, x_idx] = x - mean_x
    digit[:, y_idx] = y - mean_y
    
    return digit


def apply_unit_distance_normalization(digit, inplace=False):
    """ Scales the coordinates (using minMax scaling) so the maximum euclidian distance to any of the points
    in the given digit is set to 1.0
    @param digit: The digit to apply the transformationt to
    @param inplace: If True, the operation is performed inplace
                    (this only works if digit is already a numpy array)
                    If False, a copy of the given data is made
    @returns The resulting digit as a copy if inplace is False
    @returns The given digit itself after applying the transformation
    """
    if not isinstance(digit, np.ndarray) or inplace == False:
        digit = np.array(digit)
    
    x_idx = DataSetContract.DigitSet.Frame.indices['X']
    y_idx = DataSetContract.DigitSet.Frame.indices['Y']
    
    x = digit[:, x_idx]
    y = digit[:, y_idx]
    
    mean_x, mean_y = x.mean(), y.mean()
    
    squared_euclidian_distance = np.square(x - mean_x) + np.square(y - mean_y)
    max_distance = squared_euclidian_distance.max()
    scale = np.sqrt(max_distance)
    
    digit[:, x_idx] = x / scale
    digit[:, y_idx] = y / scale
    return digit










