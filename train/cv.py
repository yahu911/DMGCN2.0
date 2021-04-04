"""
Created on Wed Oct 16
Keras Implementation of Deep Multiple Graph Convolution Neural Network (DMGCN) model in:
Hu Yang, Wei Pan, Zhong Zhuang.
@author: Hu Yang (hu.yang@cufe.edu.cn)
"""

from __future__ import print_function
import random
from train.train import *

"""cross-validation"""
def cvtrain(x, list_adj, y, train_mask, test_mask
            , dropout_size=0.2, tuning_param=0.001, learning_rate=0.001
            , n_layers=1, n_nurons=16, activation_function="relu"
            , n_epoch=200, top_k=100, pool=False, cv_num=5):
    which_train = np.where(train_mask)
    which_train = which_train[0]
    which_train_index = np.array(range(0, len(which_train)))

    random.shuffle(which_train)
    cv_scores = list()

    for train_index, test_index in k_fold_cross_validation(which_train_index, cv_num):
        cv_train_mask = np.zeros(len(train_mask), dtype=bool)
        cv_test_mask = np.zeros(len(train_mask), dtype=bool)

        cv_train_mask[which_train[train_index]] = np.ones(len(train_index), dtype=bool)
        cv_test_mask[which_train[test_index]] = np.ones(len(test_index), dtype=bool)

        test_loss, test_acc, epoch, y_prediction, y_true = train(x, list_adj, y
                                                                  , train_mask, test_mask
                                                                  , dropout_size, tuning_param
                                                                  , learning_rate, n_layers=1
                                                                  , n_nurons=16, activation_function="relu"
                                                                  , n_epoch=200, top_k=100
                                                                  , pool=False)
        cv_scores.append(test_acc)

    return np.mean(cv_scores)
