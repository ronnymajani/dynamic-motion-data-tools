#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import spline_interpolate_and_resample



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
    frame_len = len(digit[0])
    data = np.array(digit, dtype=np.float32).reshape(-1, frame_len)
    x = data[:, 0]
    y = data[:, 1]
    x = x.flatten()
    y = y.flatten()
    
    ax = plt.gca()
    
    if show_lines:
        xi, yi = spline_interpolate_and_resample(data, 1000)
        y_max = max(y.max(), yi.max()) + padding
        y_min = min(y.min(), yi.min()) - padding
        x_max = max(x.max(), xi.max()) + padding
        x_min = min(x.min(), xi.min()) - padding
    else:
        y_max = y.max() + padding
        y_min = y.min() - padding
        x_max = x.max() + padding
        x_min = x.min() - padding   
    # Set the Y and X axis limits
    # we invert the Y axis (our coordinates assume X, Y is the top left corner)
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
            c = data[:, 3].flatten()
        else:
            c = np.arange(len(x))
        plt.scatter(x, y, c=c, cmap="inferno")
    # Interpolate between points and draw a connecting line (spline)
    if show_lines:
        plt.plot(xi, yi, '-', c="#1f77b4ff")





