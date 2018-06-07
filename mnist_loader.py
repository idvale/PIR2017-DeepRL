#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:19:34 2018

@author: paul
"""

import numpy as np
#Library to handle .csv file
import pandas as pd


CSV_COLUMN_NAMES = ['Number']+['pix_{}_{}'.format(i,j) for i in range(28) for j in range(28)]

def number_to_array(number):
    v=np.zeros((1,10),dtype=int)
    v[0,number]=1
    return v
    
    
#### Load the MNIST data
def load_data(label_name='Number'):
    test=pd.read_csv("./Data/mnist_test.csv",names=CSV_COLUMN_NAMES)
    #test_features, test_label= test, test.pop(label_name)
    training_set=pd.read_csv("./Data/mnist_train.csv",names=CSV_COLUMN_NAMES)
    #training_set_features, training_set_label= training_set, training_set.pop(label_name)
    
    test=test.values
    training_set=training_set.values
    
    test_label=np.zeros((1,10))
    for k in range(test.shape[0]):
        test_label=np.concatenate((test_label,number_to_array(test[k][0])))
    test_label=test_label[1:,:]
    test=test[:,1:]
    test=np.concatenate((test,test_label),axis=1)
    
    training_label=np.zeros((1,10))
    for k in range(training_set.shape[0]):
        training_label=np.concatenate((training_label,number_to_array(training_set[k][0])))
    training_label=training_label[1:,:]
    training_set=training_set[:,1:]
    training_set=np.concatenate((training_set,training_label),axis=1)
      
    return test,training_set

