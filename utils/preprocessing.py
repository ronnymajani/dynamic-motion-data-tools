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
    """ Scales the coordinates (using minMax scaling) so the maximum euclidian distance from the 
    mean (center of the points) to any of the points in the given digit is set to 1.0 
    and  the minimum is set to -1.0
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



def convert_xy_to_derivative(digit, normalize=False, inplace=False):
    """ Converts the given digit's X, Y data into derivatives (change of X, Y values each timestep)
    @param digit: The digit to generate the derivative from
    @param normalize: If set to True, the generated derivative vectors will be normalized to unit length,
    stripping them of any magnitude information.
    @param inplace: If True, the operation is performed inplace
                    (this only works if digit is already a numpy array)
                    If False, a copy of the given data is made
    @returns The given digit with its X Y values replaced with their derivatives
    """
    if not isinstance(digit, np.ndarray) or inplace == False:
        digit = np.array(digit)

    x_idx = DataSetContract.DigitSet.Frame.indices['X']
    y_idx = DataSetContract.DigitSet.Frame.indices['Y']

    digit_xy = digit[:, [x_idx, y_idx]]  # select only the X and Y values    
    rotated_digit = np.roll(digit_xy, -1)  # the digit frames one time step ahead
    derivative = rotated_digit - digit_xy
    derivative = derivative[:-1]  # drop last element as it is connecting the first and last frame
    
    if normalize:
        dx = derivative[:, 0]
        dy = derivative[:, 1]
        
        squared_dist = np.square(dx) + np.square(dy)
        unit = np.sqrt(squared_dist)
        
        dx /= unit
        dy /= unit
        
    digit[:-1, [x_idx, y_idx]] = derivative
    return digit
    









