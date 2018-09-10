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


import mxnet as mx
import logging
from mxnet.contrib.svrg_optimization.svrg_module import SVRGModule


def create_lin_reg_network(train_features, train_labels, feature_dim, batch_size, update_freq, ctx, logger):
    # fit a linear regression model with mxnet SVRGModule
    print("Fitting linear regression with mxnet")
    train_iter = mx.io.NDArrayIter(train_features, train_labels, batch_size=batch_size, shuffle=True,
                                   data_name='data', label_name='label')
    data = mx.sym.Variable("data")
    label = mx.sym.Variable("label")
    weight = mx.sym.Variable("fc_weight", shape=(1, feature_dim))
    net = mx.sym.dot(data, weight.transpose())
    bias = mx.sym.Variable("fc_bias", shape=(1,), wd_mult=0.0, lr_mult=10.0)
    net = mx.sym.broadcast_plus(net, bias)
    net = mx.sym.LinearRegressionOutput(data=net, label=label)
    mod = SVRGModule(symbol=net, context=ctx, data_names=['data'], label_names=['label'], logger=logger,
                     update_freq=update_freq)
    return train_iter, mod


def create_metrics(metrics):
    metric = mx.metric.create(metrics)
    return metric


def create_logger():
    logger = logging.getLogger('sgd_svrg')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler('experiments.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


################################################################################
# Functions below are for benchmark purpose to calcuate expectation, variance of
# gradients per epoch for each parameter. These calculations will be helpful when
# benchmarking SVRG optimization with other optimization techniques, such as SGD.
# Currently it only calculates the expectation, variance for single context but
# can be extended to multi-context in later iterations.
################################################################################

def accumulate_grad(grad_dict, mod):
    param_names = mod._exec_group.param_names

    for index, name in enumerate(param_names):
        if name not in grad_dict:
            grad_dict[name] = mod._exec_group.grad_arrays[index][0].copy()
        else:
            grad_dict[name] = mx.ndarray.concat(grad_dict[name], mod._exec_group.grad_arrays[index][0], dim=0)


def calc_expectation(grad_dict, num_batches):
    """Calculates the expectation of the gradients per epoch for each parameter w.r.t number of batches

    Parameters
    ----------
    grad_dict: dict
        dictionary that maps parameter name to gradients in the mod executor group
    num_batches: int
        number of batches

    Returns
    ----------
    grad_dict: dict
        dictionary with new keys mapping to gradients expectations

    """
    for key in grad_dict.keys():
        grad_dict[str.format(key+"_expectation")] = mx.ndarray.sum(grad_dict[key], axis=0) / num_batches

    return grad_dict


def calc_variance(grad_dict, num_batches, param_names):
    """Calculates the variance of the gradients per epoch for each parameter w.r.t number of batches

    Parameters
    ----------
    grad_dict: dict
        dictionary that maps parameter name to gradients in the mod executor group
    num_batches: int
        number of batches
    param_names: str
        parameter name in the module

    Returns
    ----------
    grad_dict: dict
        dictionary with new keys mapping to gradients variance

    """
    for i in range(len(param_names)):
        diff_sqr = mx.ndarray.square(mx.nd.subtract(grad_dict[param_names[i]],
                                                    grad_dict[str.format(param_names[i]+"_expectation")]))
        grad_dict[str.format(param_names[i] + "_variance")] = mx.ndarray.sum(diff_sqr, axis=0) / num_batches
