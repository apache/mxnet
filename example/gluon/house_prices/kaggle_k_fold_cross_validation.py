# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# This example provides an end-to-end pipeline for a common Kaggle competition.
# The entire pipeline includes common utilities such as k-fold cross validation
# and data pre-processing.
#
# Specifically, the example studies the `House Prices: Advanced Regression
# Techniques` challenge as a case study.
#
# The link to the problem on Kaggle:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

import numpy as onp
import pandas as pd
from mxnet import autograd
from mxnet import gluon
from mxnet import np

# After logging in www.kaggle.com, the training and testing data sets can be downloaded at:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/test.csv
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))

# Get all the numerical features and apply standardization.
numeric_feas = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feas] = all_X[numeric_feas].apply(lambda x:
                                                (x - x.mean()) / (x.std()))
# Convert categorical feature values to numerical (including N/A).
all_X = pd.get_dummies(all_X, dummy_na=True)
# Approximate N/A feature value by the mean value of the current feature.
all_X = all_X.fillna(all_X.mean())

num_train = train.shape[0]

# Convert data formats to NDArrays to feed into gluon.
X_train = all_X[:num_train].as_matrix()
X_test = all_X[num_train:].as_matrix()
y_train = train.SalePrice.as_matrix()

X_train = np.array(X_train)
y_train = np.array(y_train)
y_train.reshape((num_train, 1))

X_test = np.array(X_test)
square_loss = gluon.loss.L2Loss()

def get_rmse_log(net, X_train, y_train):
    """Gets root mse between the logarithms of the prediction and the truth."""
    num_train = X_train.shape[0]
    clipped_preds = np.clip(net(X_train), 1, float('inf'))
    return np.sqrt(2 * np.sum(square_loss(
        np.log(clipped_preds), np.log(y_train))).item() / num_train)

def get_net():
    """Gets a neural network. Better results are obtained with modifications."""
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(50, activation="relu"))
    net.add(gluon.nn.Dense(1))
    net.initialize()
    return net

def train(net, X_train, y_train, epochs, verbose_epoch, learning_rate,
          weight_decay, batch_size):
    """Trains the model."""
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size,
                                            shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate,
                             'wd': weight_decay})
    net.initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            avg_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print(f"Epoch {epoch}, train loss: {avg_loss}")
    return avg_loss

def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay, batch_size):
    """Conducts k-fold cross validation for the model."""
    assert k > 1
    fold_size = X_train.shape[0] // k

    train_loss_sum = 0.0
    test_loss_sum = 0.0
    for test_idx in range(k):
        X_val_test = X_train[test_idx * fold_size: (test_idx + 1) *
                                                   fold_size, :]
        y_val_test = y_train[test_idx * fold_size: (test_idx + 1) * fold_size]
        val_train_defined = False
        for i in range(k):
            if i != test_idx:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    X_val_train = np.concatenate([X_val_train, X_cur_fold], axis=0)
                    y_val_train = np.concatenate([y_val_train, y_cur_fold], axis=0)
        net = get_net()
        train_loss = train(net, X_val_train, y_val_train, epochs, verbose_epoch,
                           learning_rate, weight_decay, batch_size)
        train_loss_sum += train_loss
        test_loss = get_rmse_log(net, X_val_test, y_val_test)
        print(f"Test loss: {test_loss}")
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k

# The sets of parameters. Better results are obtained with modifications.
# These parameters can be fine-tuned with k-fold cross-validation.
k = 5
epochs = 100
verbose_epoch = 95
learning_rate = 0.3
weight_decay = 100
batch_size = 100

train_loss, test_loss = \
    k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train,
                       learning_rate, weight_decay, batch_size)
print(f"{k}-fold validation: Avg train loss: {train_loss}, Avg test loss: {test_loss}")

def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
          weight_decay, batch_size):
    """Trains the model and predicts on the test data set."""
    net = get_net()
    _ = train(net, X_train, y_train, epochs, verbose_epoch, learning_rate,
                 weight_decay, batch_size)
    preds = net(X_test).asnumpy()
    test['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate,
      weight_decay, batch_size)
