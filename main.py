# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16
Keras Implementation of Deep Multiple Graph Convolution Neural Network (DMGCN) model in:
Hu Yang, Wei Pan, Zhong Zhuang.
@author: Hu Yang (hu.yang@cufe.edu.cn)
"""

from __future__ import print_function
from train.cv import *
import time
###############################################
# load dataset
input_path = "data/Breast_A/"

# the number of graphs
m = 7
x, y, graphs = load_data(input_path, ngraphs=m, begin_graph=7)  # load_data()
top_k = np.cast['int32'](x.shape[1] * 0.1, np.int)

random.seed(1)
randomset = np.random.rand(len(y))
train_mask = randomset < 0.65
test_mask = (randomset >= 0.65)
start_time = time.time()

n_layers = 1
n_nurons = 16
activation_function = "relu"
dropout_size = 0.2
tuning_param = 0.001
learning_rate = 0.001
n_epoch = 200

"""test train function
Fit the model using x, y and graphs of features and then use the fitted model to predict X and graphs. 
"""
train(x, graphs, y, train_mask, test_mask, dropout_size, tuning_param, learning_rate)

"""test cross validation train function 
Fit the model using x, y and graphs of features and return cross validation error. 
"""
cvtrain(x, graphs, y, train_mask, test_mask, dropout_size, tuning_param, learning_rate)