# -*- coding: utf-8 -*-

# allow the notebook to access the parent directory so we can import the other modules
# https://stackoverflow.com/a/35273613
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
    
import os
dataset_folder_path = os.path.join("files", "dataset")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: %s PATH/TO/DATASET" % sys.argv[0])
		exit(-1)
		
	dataset_folder_path = sys.argv[1]

	#%%
	from data.DataSet import DataSet
	dataset = DataSet()
	dataset.load(dataset_folder_path, test_set_percentage=0, validation_set_percentage=0)

	#%%
    print(len(dataset.train_data))
    print(len(dataset.valid_data))
    print(len(dataset.test_data))

	#%%
    import numpy as np
    total = []
    for digit in dataset.train_data:
        total.append(len(digit))
    avg = np.mean(total)
    std = np.std(total)
    
    print(avg)
    print(std)

