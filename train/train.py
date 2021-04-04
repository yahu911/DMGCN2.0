"""
Created on Wed Oct 16
Keras Implementation of Deep Multiple Graph Convolution Neural Network (DMGCN) model in:
Hu Yang, Wei Pan, Zhong Zhuang.
@author: Hu Yang (hu.yang@cufe.edu.cn)
"""

from __future__ import print_function
from models.DMGCN import DMGCN
from utils.utils import *
from keras import backend as Ks
import time

PATIENCE = 100


def train(x, list_adj, y, train_mask, test_mask
            , dropout_size=0.2, tuning_param=0.001, learning_rate=0.001
            , n_layers=1, n_nurons=16, activation_function="relu"
            , n_epoch=200, top_k=100, pool=False):

    """Fit the model using X, y and graphs of features and then use the fitted model to predict X and graphs.
    ----------
    x : array-like, shape=(n_samples, dim_input)
        Training samples.
    graphs : list of arrays, one graph is an array， shape=(dim_input, dim_input)
    y : array-like, shape=(n_samples, n_classes)
        Training labels (formatted as a binary matrix, as returned by a standard One Hot Encoder, see
         http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html for more details).
    n_layers : integer, the number of layers
    n_neurons : integer, the number of neurons of the output of each gcn layer
    activation_function : character, the activation function
    dropout_size : float, the probability of dropping rate
    tuning_param : double, the hyperparameter of l1  or l2 regularizer
    learning_rate : double, the learning rate of Adam algorithm
    top_k : integer, the number of how many nodes retain in each self attention gcn

    Returns
    -------
    labels : array, shape=(n_samples,)
         Array of class indices.
    """

    model = DMGCN(x, list_adj, y, n_layers, n_nurons, activation_function
                  , dropout_size, tuning_param, learning_rate, top_k)

    train_val_loss = 0.0
    train_val_acc = 0.0
    wait = 0
    y_prediction = None
    best_val_loss = 99999

    for epoch in range(1, n_epoch + 1):
        # Log wall-clock time
        t = time.time()

        model.fit(x, y, sample_weight=train_mask, batch_size=x.shape[0], epochs=1, shuffle=True, verbose=0)

        # Predict on full dataset
        y_prediction = model.predict(x, batch_size=x.shape[0])

        train_val_loss = categorical_crossentropy(y_prediction[train_mask], y[train_mask])
        if train_val_loss < best_val_loss:
            best_val_loss = train_val_loss
            wait = 0
        else:
            if wait >= PATIENCE:
                print('Epoch {}: early stopping'.format(epoch))
                print("train_val_loss=" + str(train_val_loss) + "," + "train_val_acc=" + str(train_val_acc))
                break
            wait += 1

    # Testing
    test_loss = categorical_crossentropy(y_prediction[test_mask], y[test_mask])
    test_acc = accuracy(y_prediction[test_mask], y[test_mask])
    print("test_loss=" + str(test_loss) + "," + "test_acc=" + str(test_acc))

    Ks.clear_session()
    del model
    return test_loss, test_acc, epoch, y_prediction[test_mask], y[test_mask]
