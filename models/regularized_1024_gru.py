#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:04:33 2018

@author: ronnymajani
"""


from keras import Sequential
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Nadam
import keras.regularizers

from .model_template import ModelTemplate


class Regularized1024GRU(ModelTemplate):
    NAME = "Regularized 1024 GRU"
    PREFIX = "regularized_1024_gru"
    
    def __init__(self, input_shape, **kwargs):
        ModelTemplate.__init__(self, input_shape, **kwargs)
        self.name = self.__class__.NAME
        self.prefix = self.__class__.PREFIX
    
    def _build(self):
        # Model
        self.model = Sequential()
        self.model.add(GRU(1024, return_sequences=True, reset_after=True,
                           input_shape=self.input_shape))
        self.model.add(Dropout(0.5))
        self.model.add(GRU(1024, return_sequences=True, reset_after=True))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        # Optimizer
        self.optimizer = Nadam(lr=0.0005, schedule_decay=0.15)
        # Compile Model
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['categorical_accuracy'])
        
    def train(self, X_train, Y_train, X_valid=None, Y_valid=None):
        # Validation Data
        if X_valid is None or Y_valid is None:
            validation_data = None
        else:
            validation_data = (X_valid, Y_valid)
        # Train Model
        self.fit_params = {
                'x': X_train,
                'y': Y_train,
                'epochs': self.num_epochs,
                'verbose': 1,
                'callbacks': self.callbacks,
                'validation_data': validation_data,
                'batch_size': self.batch_size
        }
        self.model.fit(**self.fit_params)
        
    
    
    
        

    
    
