#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:05:54 2018

@author: ronnymajani
"""
import os
import time, datetime
import io
import pprint
import warnings
from keras.callbacks import ModelCheckpoint
from .tfcallback import TensorBoardCallback

class ModelTemplate(object):
    DEFAULT_CHECKPOINTS_SAVE_PATH = os.path.join("files", "checkpoints")
    DEFAULT_TENSORBOARD_LOGS_PATH = os.path.join("files", "tflogs")
    
    def __init__(self, input_shape, checkpoints_save_path=None, tensorboard_logs_path=None):
        # generic attributes
        self.name = "Model Template"
        self.prefix = "model"
        self.timestamp = time.time()
        self._use_callbacks = True
        # folders and paths
        if checkpoints_save_path is None:
            self.checkpoints_save_path = ModelTemplate.DEFAULT_CHECKPOINTS_SAVE_PATH
        else:
            self.checkpoints_save_path = checkpoints_save_path
            
        if tensorboard_logs_path is None:
            self.tensorboard_logs_path = ModelTemplate.DEFAULT_TENSORBOARD_LOGS_PATH
        else:
            self.tensorboard_logs_path = tensorboard_logs_path
            
        # model attributes
        self.model = None
        self.optimizer = None
        self.batch_size = 32
        self.num_epochs = 30
        self.callbacks = []
        self.input_shape = input_shape
        self.fit_params = None
        self.callback_monitored_value = 'val_categorical_accuracy'
        
    def disable_callbacks(self):
        if self.model is not None:
            warnings.warn("Disabling callbacks after model has been compiled!")
        self._use_callbacks = False
        self.callbacks = []
        
    def enable_callbacks(self):
        self._use_callbacks = True
        self.__setup_checkpoints_folder()
        self._setup_callback_folders()
        self._setup_callbacks()
        
    def initialize(self):
        if self._use_callbacks:
            self.__setup_checkpoints_folder()
            self._setup_callback_folders()
            self._setup_callbacks()
        self._build()
        
    def evaluate(self, **kwargs):
        return self.model.evaluate(**kwargs)
    
    def predict_classes(self, **kwargs):
        return self.model.predict_classes(**kwargs)
        
    # OVERRIDE THIS METHOD
    def _build(self):
        raise NotImplementedError
    
    # OVERRIDE THIS METHOD
    def train(self, X_train, Y_train, X_valid=None, Y_valid=None):
        raise NotImplementedError
        
    def save_summary(self, recorded_operations, filename="summary.txt"):
        """ Save summary of current model in a file """
        self.__setup_checkpoints_folder()  # setup checkpoints folder if it doesn't exist
        output_filename = os.path.join(self.checkpoints_dir, filename)
        with open(output_filename, "w") as fd:
            sep = "\n\n----------\n\n"
            timestamp = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            fd.write(self.name)
            fd.write("\n" + timestamp)
            fd.write(sep)
            fd.write("\n\n".join(recorded_operations))
            fd.write(self.__str__())
            print("Summary saved to", output_filename)
            
    def save_config(self, filename="model.txt"):
        self.__setup_checkpoints_folder()  # setup checkpoints folder if it doesn't exist
        output_filename = os.path.join(self.checkpoints_dir, filename)
        with open(output_filename, "w") as fd:
            fd.write(self.get_model_config())
            fd.write("\n")
            print("Model config saved to", output_filename)
            
    def save(self, filename="model_manual_save.hdf5"):
        self.__setup_checkpoints_folder()  # setup checkpoints folder if it doesn't exist
        output_filename = os.path.join(self.checkpoints_dir, filename)
        self.model.save(output_filename)
            
    def __str__(self):
        res = ""
        sep = "\n\n----------\n\n"
        res += sep
        res += "Optimizer: %s\n" % self.optimizer.__class__
        res += "Batch Size: %d\n" % self.batch_size
        res += "Number of Epochs: %d\n" % self.num_epochs
        res += "\n"
        res += self._get_model_summary()
        res += sep
        return res
    
    # OVERRIDEABLE
    def _setup_callback_folders(self):
        """ Create all needed folders, and save their paths so they can be used later """
        self.tensorboard_dir = os.path.join(self.tensorboard_logs_path, "{}".format(self.timestamp))
        if not os.path.exists(self.tensorboard_logs_path):
            os.mkdir(self.tensorboard_logs_path)

    def __setup_checkpoints_folder(self):
        self.checkpoints_dir = os.path.join(self.checkpoints_save_path, "{}".format(self.timestamp))
        # Create Checkpoints save directory if it doesn't exist
        if not os.path.exists(self.checkpoints_save_path):
            os.mkdir(self.checkpoints_save_path)
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
    
    # OVERRIDEABLE
    def _setup_callbacks(self):
        """ Setup necessary callbacks """
        # Checkpoint for saving best models
        save_filename = os.path.join(self.checkpoints_dir, self.prefix + "-{epoch:02d}-{%s:.2f}.hdf5"%self.callback_monitored_value)
        checkpointer = ModelCheckpoint(save_filename, monitor=self.callback_monitored_value, verbose=1, save_best_only=True, mode='max')
        self.callbacks.append(checkpointer)
        
        # Tensorboard Callback
        tensorboard_callback = TensorBoardCallback(log_every=1, log_dir=self.tensorboard_dir, write_graph=False)
        tensorboard_callback.write_batch_performance = True
        self.callbacks.append(tensorboard_callback)
    
    def _get_model_summary(self):
        """ Return Keras model summary as a string """
        outputBuf  = io.StringIO()
        self.model.summary(print_fn=lambda x: outputBuf.write(x + '\n'))
        res = outputBuf.getvalue()
        outputBuf.close()
        return res
            
    def get_model_config(self):
        """ Get Keras model configuration as a JSON string """
        outputBuf  = io.StringIO()
        pp = pprint.PrettyPrinter(indent=2, stream=outputBuf)
        pp.pprint(self.model.get_config())
        res = outputBuf.getvalue()
        outputBuf.close()
        return res
    
    
