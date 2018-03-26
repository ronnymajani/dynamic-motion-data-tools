#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage



def show_digit(digit, label=None, show_points=True, show_lines=True, use_time_as_color=False, padding=0.1):
    """ Displays the given digit
    @param digit: A sequence of [X,Y,P,t] points that represent a dynamically drawn handwritten digit
    @param label: The label of the current digit. If given, the plot's window will be set to the label
    @param show_points: If True, the function will explicitly display the actual points in the digit
    @param show_lines: If True, the function will draw a spline between the points
    @param use_time_as_color: If True, and show_points is True, the drawn points will use the time values
                              of the digits to determine which color from the colormap to use, otherwise the
                              colors will be uniformly distributed. You can only use this if the time values 
                              of the given digit is the time elapsed since the first point in the sequence.
                              This won't work if the time feature represents 'dt' (the time difference between
                              each two points). This features means that the faster a number was drawn, the more
                              the points will have similar colors. In other words this feature visualizes the speed
                              of the drawing using the colors of the points, and is useful to compare digits as well
    @param padding: how much padding to add around the drawn shapes
    """
    #todo: clear plot option?
    #todo: use subplots?
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
    ax = plt.gca()
    # Set the Y and X axis limits
    # we invert the Y axis (our coordinates assume X, Y is the top left corner)
    y_max = max(y.max(), yi.max()) + padding
    y_min = min(y.min(), yi.min()) - padding
    x_max = max(x.max(), xi.max()) + padding
    x_min = min(x.min(), xi.min()) - padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    # Preserve the aspect ratio and scale between the X and Y axis
    ax.set_aspect('equal', adjustable='box')
    # Set the X axis to be drawn on the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    # Set axes labels
    plt.xlabel("X")
    plt.ylabel("Y")
    # Set label as title
    if label is not None:
        plt.gcf().canvas.set_window_title("Digit: %d" % label)
    # Show the points of the digit
    if show_points:
        if use_time_as_color:
            c = t.flatten()
        else:
            c = np.arange(len(x))
        plt.scatter(x, y, c=c, cmap="inferno")
    # Interpolate between points and draw a connecting line (spline)
    if show_lines:
        plt.plot(xi, yi, '-', c="#1f77b4ff")





