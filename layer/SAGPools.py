# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16
Keras Implementation of Deep Multiple Graph Convolution Neural Network (DMGCN) model in:
Hu Yang, Wei Pan, Zhong Zhuang.
@author: Hu Yang (hu.yang@cufe.edu.cn)
"""

from keras.layers import Layer
from keras import activations, initializers, constraints
from keras import regularizers
import keras.backend as K
import tensorflow as tf


class SAGPools(Layer):

    def __init__(self, output_dim, graphs, topk,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.graphs = graphs
        self.topk = topk
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        super(SAGPools, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(SAGPools, self).build(input_shape)

    def call(self, x):
        xl = tf.matmul(tf.cast(x, tf.float32), tf.cast(self.graphs, tf.float32))
        xl = K.dot(xl, self.kernel)

        if self.bias:
            xl += self.bias

        scores = tf.matmul(tf.cast(self.graphs, tf.float32), tf.cast(self.kernel, tf.float32))

        scores = tf.squeeze(scores)
        topk = tf.nn.top_k(scores, k=self.topk)

        values = tf.squeeze(topk.values)
        indices = tf.squeeze(topk.indices)

        indices = tf.convert_to_tensor(indices)
        indices = tf.expand_dims(indices, axis=1)

        values = tf.convert_to_tensor(values)

        shape = x.shape[1]
        shape = tf.cast(shape, tf.int32)
        shape = tf.convert_to_tensor(shape)

        scatter = tf.scatter_nd(indices, tf.ones_like(values), [shape])
        scatter = tf.cast(scatter, tf.float32)

        subgraph = tf.multiply(tf.cast(self.graphs,tf.float32),tf.cast(scatter,tf.float32))
        subgraph = tf.multiply(tf.cast(tf.transpose(scatter), tf.float32), tf.cast(subgraph, tf.float32))

        return [xl, subgraph]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[1],input_shape[1])]