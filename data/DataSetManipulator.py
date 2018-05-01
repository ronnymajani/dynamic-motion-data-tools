# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import time

class DataSetManipulator(object):
    DEFAULT_MASKING_VALUE = -99
    
    def __init__(self, dataset, sequence_length, masking_value=DEFAULT_MASKING_VALUE):
        self.dataset = dataset
        self._sequenceLength = sequence_length
        self._maskingValue = masking_value
        
    def _create_data_for_generative_model(self, X, Y):
        """ @note: Why values must not be OneHotEncoded! """
        new_X = []
        new_Y = []
        for dig_idx, dig in enumerate(X):
            prefix = np.array([Y[dig_idx],Y[dig_idx]])
            new_sub_seq = []
            new_sub_seq_label = []
            for i in range(len(dig)):
                new_sub_seq.append(np.vstack((prefix, dig[:i])))  # create a subsequence from all previous elements in the sequence
                new_sub_seq_label.append(dig[i])  # append next element in sequence as the label for this subsequence
            new_X += new_sub_seq
            new_Y += new_sub_seq_label
        return new_X, new_Y
        
    def create_dataset_for_generative_models(self):
        """ Creates and returns a dataset for generative models. 
        - For each digit in the orignial dataset of length `n` time steps,
          a set of n new subsequences.
        - Each subsequence is equal to the previous subsequence concatted with 
          the next element that follows in the original sequence.
        - All subsequences are prepended with the label of the data. 
        - In the end, all the generated subsequences are padded with a masking value.
        - The new labels of each subsequence are equal to the the next consecutive element
          in the original sequence
        """
        #TODO: if Y is onehot encoded, decode it first
        
        X_train, Y_train = self._create_data_for_generative_model(self.dataset.train_data, self.dataset.train_labels)
        X_valid, Y_valid = self._create_data_for_generative_model(self.dataset.valid_data, self.dataset.valid_labels)
        X_test, Y_test = self._create_data_for_generative_model(self.dataset.test_data, self.dataset.test_labels)
        
        #TODO: mask values (with Keras maybe?)
#        pad = lambda data: pad_sequences(data, maxlen=self._sequenceLength, padding='post', truncating='post', value=self._maskingValue)
#        X_train = pad(X_train)
#        X_valid = pad(X_valid)
#        X_test = pad(X_test)
        
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
