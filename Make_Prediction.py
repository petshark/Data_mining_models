# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:18:45 2021

@author: carlo
"""

input_parameters = {'Category_to_be_predicted':'Good', 'Director':'Quentin Tarantino', 'Duration':120, 'Description':'A very bloody good movie', 'Genre':'Drama, Horror'}
models = Train_Models(input_parameters)






Prediction_Controller(input_parameters, models)
