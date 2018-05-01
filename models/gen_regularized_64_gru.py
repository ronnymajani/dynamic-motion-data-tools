#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:04:33 2018

@author: ronnymajani
"""


from keras import Sequential
from keras.layers import Masking
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Nadam
from .model_template import ModelTemplate


class GenRegularized64GRU(ModelTemplate):
    NAME = "Gen Regularized 64 GRU"
    PREFIX = "gen_regularized_64_gru"
    
    def __init__(self, input_shape, masking_value, **kwargs):
        ModelTemplate.__init__(self, input_shape, **kwargs)
        self.name = self.__class__.NAME
        self.prefix = self.__class__.PREFIX
        self.callback_monitored_value = 'val_acc'
        self.masking_value = masking_value
    
    def _build(self):
        # Model
        self.model = Sequential()
        self.model.add(Masking(mask_value=self.masking_value, input_shape=self.input_shape))
        self.model.add(GRU(64, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(GRU(64))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32))
        self.model.add(Activation('relu'))
        self.model.add(Dense(2))
        # Optimizer
        self.optimizer = Nadam(lr=0.002, schedule_decay=0.15)
        # Compile Model
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])
        
    def train(self, X_train, Y_train, X_valid, Y_valid):
        # Train Model
        self.fit_params = {
                'x': X_train,
                'y': Y_train,
                'epochs': self.num_epochs,
                'verbose': 1,
                'callbacks': self.callbacks,
                'validation_data': (X_valid, Y_valid),
                'batch_size': self.batch_size
        }
        self.model.fit(**self.fit_params)
        
    
    
    
        

    
    
