#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:01:13 2018

@author: ronnymajani
"""
import os, sys
# so the script can access the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: %s PATH/TO/DATASET" % sys.argv[0])
		exit(-1)
		
	dataset_folder_path = sys.argv[1]

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
