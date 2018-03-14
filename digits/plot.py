#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage



def show_digit(digit, show_points=True, show_lines=True, padding=100):
    """ Displays the given digit
    @param digit: A sequence of [X,Y,P,t] points that represent a dynamically drawn handwritten digit
    """
    # extract X, Y coordinates
    data = np.array(digit, dtype=np.float32).reshape(-1, 4)
    x, y, p, t = np.split(data, 4, axis=1)
    x = x.flatten()
    y = y.flatten()
    # Delete identical points to not get an error from the spline function
    # https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs/47949170#47949170
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
    xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
    # plot
    # invert Y axis (our coordinates assume X, Y is the top left corner)
    ax = plt.gca()
    y_max = max(y.max(), yi.max()) + padding
    y_min = min(y.min(), yi.min()) - padding
    x_max = max(x.max(), xi.max()) + padding
    x_min = min(x.min(), xi.min()) - padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    plt.xlabel("X")
    plt.ylabel("Y")
    
    if show_points:
        plt.plot(x, y, 'o')
    if show_lines:
        plt.plot(xi, yi, '-')




