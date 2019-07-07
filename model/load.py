# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:57:04 2019

@author: Devesh Kaushik
"""

from keras.models import load_model


def init(): 
	best_model_file = "cnn_model.h5"
    loaded_model = load_model(best_model_file)
	print("Loaded Model from disk")

	return loaded_model