#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import digits.plot as plot
from digits.preprocessing import apply_mean_centering, apply_unit_distance_normalization
from data.DigitSet import DigitSet

#%%
filename = "temp/01.15_14.03.2018_digitset.json"
digitset = DigitSet(filename)
digit = digitset[0]
digitset.apply(apply_mean_centering)
digitset.apply(apply_unit_distance_normalization)
scaled_digit = digitset[0]
#scaled_digit = apply_unit_distance_normalization(apply_mean_centering(digit))
plot.show_digit(scaled_digit, padding=0.1)
#plot.show_digit(digit)


#if __name__ == '__main__':
#    filename = "temp/01.15_14.03.2018_digitset.json"
#    digitset = loader.load_digitset(filename)
#    data, labels = loader.extract_data_and_labels(digitset)
#    plot.show_digit(data[-7])
    
#%%