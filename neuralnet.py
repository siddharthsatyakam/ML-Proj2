# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:07:46 2019

@author: siddh
"""

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""
    path = "C://Users//siddh//Documents//Assignment2//basecode//mnist_all.mat"
    mat = loadmat(path)  # loads the MAT object as a Dictionary
    
    "train_data = np.zeros(shape=(6000,28,28))"
    train_label = np.zeros(shape=(50000,1))
    train_label = np.asmatrix(train_label)
    
    test_label = np.zeros(shape=(10000,1))
    test_label = np.asmatrix(test_label)
    
    validation_label = np.zeros(shape=(10000,1))
    validation_label = np.asmatrix(validation_label)
    
    train_data = np.zeros(shape=(60000,784))
    train_data = np.asmatrix(train_data)
    t=0
    while(t<=9):
        termOfMat = "train" + str(t)
        testbed = mat[termOfMat]
        dimension = np.shape(testbed)
        
        for i in range(0,dimension[0]):
            for j in range(0,784):
                train_data[i,j] = testbed[i,j]
            train_label[i,1] = t
        t=t+1

    test_data = np.zeros(shape=(10000,784))
    test_data = np.asmatrix(test_data)
    t=0
    while(t<=9):
        termOfMat = "test" + str(t)
        testbed = mat[termOfMat]
        dimension = np.shape(testbed)
        
        for i in range(0,dimension[0]):
            for j in range(0,784):
                test_data[i,j]=testbed[i,j]
            test_label[i,1] = t
        t=t+1
    
    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    np.random.shuffle(train_data)
    
    temp_data = train_data[:50000]
    validation_data = train_data[50000:]
    train_data = temp_data
    
    np.random.shuffle(train_label)
    
    tempv_label = train_label[:50000]
    validation_label = train_label[50000:]
    train_label = tempv_label
    
    # Feature selection
    # Your code here.
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    feature_trained = bestfeatures.fit_transform(train_data,train_label)
    

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label
