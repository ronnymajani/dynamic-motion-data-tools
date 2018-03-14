#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import digits.plot as plot
from digits.preprocessing import *
from data.DigitSet import DigitSet

#%%
filename = "temp/01.15_14.03.2018_digitset.json"
digitset = DigitSet(filename)
# Apply transformations
digitset.apply(apply_mean_centering)
digitset.apply(apply_unit_distance_normalization)
digitset.apply(lambda x: normalize_pressure_value(x, 512))
# digitset.convert_t_to_dt()
# plot random digit
scaled_digit, label = digitset[-7]
plot.show_digit(scaled_digit, label=label, padding=0.1)

#%%
#if __name__ == '__main__':
#    filename = "temp/01.15_14.03.2018_digitset.json"
#    digitset = loader.load_digitset(filename)
#    data, labels = loader.extract_data_and_labels(digitset)
#    plot.show_digit(data[-7])
    
#%%
# convert dt to t
#digit = digitset[30].copy()
#for i in range(len(digit)-1, 0, -1):
#    digit[i][3] -= digit[i-1][3]