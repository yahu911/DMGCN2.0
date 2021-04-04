# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:11:53 2019

@author: Hu Yang
@e-mail: hu.yang@cufe.edu.cn

"""
import numpy as np
from random import shuffle
import scipy.sparse as sp

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

#load data sets, such as: gene expression profile, and adjacency matrix of gene networks
def load_data(path="data/input", graph_name="weighted_Adj", ngraphs=10, begin_graph=1):
    X = np.genfromtxt("{}/X".format(path), dtype=np.float32)
    adj_list = list()
    for i in range(0, ngraphs):
        adj = np.genfromtxt("{}/{}.{}".format(path, graph_name, i+begin_graph), dtype=np.float32)
        adj = sp.csr_matrix(adj, dtype=np.float32)
        adj = preprocess_adj(adj)
        adj = adj.todense()
        adj_list.append(adj)    
    y = np.genfromtxt("{}/labels".format(path), dtype=np.int)
    return X, y, adj_list

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = np.array(adj.sum(1))
        d[d==0] = 1
        d = sp.diags(np.power(d, -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = np.array(adj.sum(1))
        d[d==0] = 1
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)+ pow(10.0, -9)))

def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))

def k_fold_cross_validation(items, k, randomize=False):

    if randomize:
        items = list(items)
        shuffle(items)

    slices = [items[i::k] for i in range(k)]

    for i in range(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation


def get_features_weight(weights, n_features, n_graphs, n_layers, n_nurons):

    w_features = np.zeros((n_graphs+1, n_features))
    w_attention = np.zeros((n_graphs, n_features))
    w_graphs = np.zeros(n_graphs+1)

    #the weight only coming from features in the first layer
    #w_features[0, :] = np.sum(abs(weights[0]), axis=1)

    #the weight of graph coming from the concancated layer
    group_weights = abs(weights[(n_graphs*4+2)*n_layers])

    for i in range(0, n_graphs+1):

        # the weight coming from features and graph in the first layer
        idx = i*4
        w_features[i, :] = np.sum(abs(weights[idx]), axis=1)

        if( idx > 0 ):
            idx = i * 4 - 2
            w_attention[i-1,:] =abs(weights[idx]).squeeze()

        # For the last layer
        start = i * n_nurons
        w_graphs[i] = np.sum(group_weights[range(start, start + n_nurons), :])
        # w_graphs[i] = np.sum(w_features[i,:])

    w_features_vector = np.sum(w_features, axis=0)
    w_attention_vector = np.sum(w_attention, axis=0)

    return w_features_vector, w_attention_vector, w_graphs

def save_txt(file_path, content_list):
    with open(file_path, 'ab') as f:
        np.savetxt(f, content_list, delimiter=',')
    return 1

