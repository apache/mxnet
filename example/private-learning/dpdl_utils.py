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

# This utility module allows one to access the mxnet.gluon.ParameterDict
# and its corresponding gradients as a list of ndarray

# Functions to update, manipulate and reset this list of ndarrays back and forth
# are included in this module

# Author: Yu-Xiang Wang

from mxnet import nd
import mxnet as mx


def initialize_grad(params,ctx=mx.cpu()):
    """ initialize a grad object with format just like those in paramter """
    a=[]
    for param in params.values():
        a.append(nd.zeros(param.shape).as_in_context(ctx))
    return a


def reset_grad(grads):
    for grad in grads:
        grad[:] = 0


def accumuate_grad(grads, params, thresh):  # accumuate the thresholded gradient
    tmp=grad_norm_in_params(params)

    if tmp > thresh:
        factor = thresh / tmp
    else:
        factor = 1.0

    for grad, param in zip(grads, params.values()):
        grad[:] += param.grad() * factor

def accumulate_params(param_cumu, params, n):
    for param2,param in zip(param_cumu, params.values()):
        param2[:] = param2 *(n-1)/n + param.data() /n

def iir_filter(mm,gg,beta,order):# helps to accumuate the gradients and second momeent of adam
    for m,g in zip(mm,gg):
        m[:] = beta*m + (1-beta)*(g**order)


def extract_grad(params, grads):
    """ get the gradient attached to "params" and assign to "grads" """
    for param,grad in zip(params.values(), grads):
        grad[:] = param.grad()

def grad_norm_in_params(params):
    """Calculate the Euclidean norm of the parameters in grad list grads """
    a=0
    for item in params.values():
        a += nd.sum(item.grad() ** 2).asscalar()
    return a ** 0.5


def grad_norm(grads):
    """Calculate the Euclidean norm of the parameters in grad list grads """
    a=0
    for item in grads:
        a += nd.sum(item ** 2).asscalar()
    return a ** 0.5


def grad_rescale(grads, k):
    """scale the gradient by a factor of k"""
    y = grads.deepcopy()
    for item in y:
        item[:] = item * k
    return y # return the parameters


def grad_add(grads_batch):
    """add up the list of gradients in lists"""
    y = grads_batch[0].deepcopy()
    for xx in grads_batch:
        for item1,item2 in zip(xx,y):
            item2 += item1
    return y # return the parameters with a different gradient
