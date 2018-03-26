#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import digits.plot as plot
from digits.preprocessing import *
from data.DigitSet import DigitSet
from data.DataSet import DataSet



#%%
folder = 'temp'
dataset = DataSet(folder)
dataset.apply(apply_mean_centering)
dataset.apply(apply_unit_distance_normalization)
dataset.apply(lambda x: normalize_pressure_value(x, 512))

#%%
filename = "temp/10.43_23.03.2018_digitset.json"
digitset = DigitSet(filename)
scaled = digitset.copy()
# Apply transformations
scaled.apply(apply_mean_centering)
scaled.apply(apply_unit_distance_normalization)
scaled.apply(lambda x: normalize_pressure_value(x, 512))
if scaled.time_is_dt():
    scaled.convert_dt_to_t()

#%%
digit, label = digitset[6]
plot.show_digit(digit, label=label, 
                show_lines=True, show_points=True, 
                use_time_as_color=(not digitset.time_is_dt()), 
                padding=100)

#%%
# plot random digit
digit, label = scaled[6]
plot.show_digit(digit, label=label, 
                show_lines=True, show_points=True, 
                use_time_as_color=(not scaled.time_is_dt()), 
                padding=0.1)

#%%
#if __name__ == '__main__':
#    filename = "temp/01.15_14.03.2018_digitset.json"
#    digitset = loader.load_digitset(filename)
#    data, labels = loader.extract_data_and_labels(digitset)
#    plot.show_digit(data[-7])
    
#%%
# convert dt to t
#digit, label = digitset[30]
#digit = digit.copy()
#for i in range(1, len(digit)):
#    digit[i][3] += digit[i-1][3]
#    
