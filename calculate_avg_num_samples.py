#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:01:13 2018

@author: ronnymajani
"""

dataset_folder_path = 'temp'
#%%
from data.DataSet import DataSet
dataset = DataSet()
dataset.load(dataset_folder_path, test_set_percentage=0)

#%%
print(len(dataset.train_data))
print(len(dataset.test_data))

#%%
total = 0
for digit in dataset.train_data:
    total += len(digit)
avg = total / len(dataset.train_data)
print(avg)
