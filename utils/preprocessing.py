#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from data.contract import DataSetContract
from utils.decorators import preprocessingOperation
from scipy import interpolate, ndimage


@preprocessingOperation("Pressure Values Normalization")
def normalize_pressure_value(digit, max_pressure_val=512, inplace=False):
    """ Normalizes the pressure value to the range [0.0, 1.0] given the maximum pressure value possible """
    p_idx = DataSetContract.DigitSet.Frame.indices['P']
    if not isinstance(digit, np.ndarray) or inplace == False:
        digit = np.array(digit)
    digit[:, p_idx] /= max_pressure_val
    return digit


@preprocessingOperation("Mean Centering")
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


@preprocessingOperation("Unit Distance Normalization")
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


@preprocessingOperation("Convert (X, Y) points to Discrete Derivatives (dx, dy)")
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
    

@preprocessingOperation("Reverse Order of Digit Sequences")
def reverse_digit_sequence(digit, inplace=False):
    """ Reverses the order of the frames in the given digit
    @returns a reversed sequence version of the given digit
    """
    if not isinstance(digit, np.ndarray) or inplace == False:
        digit = np.array(digit)
    return digit[::-1]


@preprocessingOperation("B-Spine Interpolation and Resampling (Warning: Deletes time and pressure features!)")
def spline_interpolate_and_resample(digit, num_samples):
    """ Performs B-Spline interpolation on the given digit, then resamples new uniformly distributed points from the
    newly calculated spline, and sets returns a new digit with the resampled X, Y values.
    @note: this function completely ignores and disposes of the other features (time and pressure)
        and the resulting digit only contains X and Y
    @param[in] digit: The digit to interpolate and resample from
    @param[in] num_samples: The number of samples to take from the calculated spline.
    @returns a 2D array, with its first column set to the resulting X values,
        and its Y column set to the corresponding Y values
    """
    if not isinstance(digit, np.ndarray):
        digit = np.array(digit)
        
    x_idx = DataSetContract.DigitSet.Frame.indices['X']
    y_idx = DataSetContract.DigitSet.Frame.indices['Y']
    # Delete identical points to not get an error from the spline function
    # https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs/47949170#47949170
    x = digit[:, x_idx]
    y = digit[:, y_idx]
    x = x.flatten()
    y = y.flatten()
    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    xp = np.r_[x[okay], x[-1], x[0]]
    yp = np.r_[y[okay], y[-1], y[0]]
    jump = np.sqrt(np.diff(xp)**2 + np.diff(yp)**2) 
    smooth_jump = ndimage.gaussian_filter1d(jump, 5, mode='wrap')  # window of size 5 is arbitrary
    limit = 2*np.median(smooth_jump)    # factor 2 is arbitrary
    xn, yn = xp[:-1], yp[:-1]
    xn = xn[(jump > 0) & (smooth_jump < limit)]
    yn = yn[(jump > 0) & (smooth_jump < limit)]
    # Generate B-Spline
    tck, u = interpolate.splprep([xn, yn], s=0)
    xi, yi = interpolate.splev(np.linspace(0, 1, num_samples), tck)
    return np.array([xi, yi]).T


