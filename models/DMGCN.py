# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16
Keras Implementation of Deep Multiple Graph Convolution Neural Network (DMGCN) model in:
Hu Yang, Wei Pan, Zhong Zhuang.
@author: Hu Yang (hu.yang@cufe.edu.cn)
"""
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

K.set_image_dim_ordering('tf')

from keras.layers import Input, Dropout, Dense, Concatenate, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
from layer.gcn import GCN
from layer.SAGPools import SAGPools

'''
   Structure of Deep Multiple Graph Convolutional Neural Networks
'''
def DMGCN(x, graphs, y, n_layers, n_neurons
          , activation_function, dropout_size
          , tuning_param, learning_rate, top_k=100, pool=False):

    layer_out_list = list()
    x0 = Input(shape=(x.shape[1],))
    y_dim = 1

    if (len(y.shape) != 1):
        y_dim = y.shape[1]

    for i in range(0, len(graphs) + 1):

        layer_name = 'H_' + str(i) + "_"
        if (i == 0):
            H0 = x0
            for j in range(0, n_layers):
                layer_name_temp = layer_name + str(j + 1) + '_dnn_layer'
                H0 = Dense(n_neurons, activation=activation_function
                           , kernel_regularizer=l2(tuning_param), name=layer_name_temp)(H0)
                H0 = Dropout(dropout_size)(H0)
            layer_out_list.append(H0)
        else:
            graph_i = graphs[i - 1]
            graph_i = tf.convert_to_tensor(graph_i)
            Hi = x0
            n_nurons_i = x.shape[1]
            for j in range(0, n_layers):

                if (j == n_layers-1):  n_nurons_i = n_neurons

                if (pool==True):
                    # the pooling layer outputs the score of X and adj_i
                    layer_name_temp = layer_name + str(j + 1) + '_pooling_layer'
                    Hi_score, graph_pool_i = SAGPools(1, graphs=graph_i, topk=top_k, activation="sigmoid"
                                                 , kernel_regularizer=l2(tuning_param), name=layer_name_temp)(Hi)

                    # the single gcn layer outputs the Hi
                    layer_name_temp = layer_name + str(j + 1) + '_gcn_layer'
                    Hi = GCN(n_nurons_i, graphs=graph_pool_i, activation=activation_function
                             , kernel_regularizer=l2(tuning_param), name=layer_name_temp)(Hi)
                    Hi = Dropout(dropout_size)(Hi)
                    Hi = Multiply()([Hi_score, Hi])
                else:
                    # the single gcn layer outputs the Hi
                    layer_name_temp = layer_name + str(j + 1) + '_gcn_layer'
                    Hi = GCN(n_nurons_i, graphs=graph_i, activation=activation_function
                             , kernel_regularizer=l2(tuning_param), name=layer_name_temp)(Hi)
                    Hi = Dropout(dropout_size)(Hi)

            layer_out_list.append(Hi)

    H = Concatenate(axis=-1)(layer_out_list)
    layer_name_temp = layer_name + '_concatenate_layer'
    H = Dense(n_neurons, activation=activation_function, kernel_regularizer=l2(tuning_param)
              , name=layer_name_temp)(H)

    z = Dense(y_dim, activation="softmax")(H)

    model = Model(inputs=[x0], outputs=z)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
    print(model.summary())

    return model
