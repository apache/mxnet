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

"""
Reference:

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
"""
import mxnet as mx
import numpy as np

def get_symbol(num_classes, dtype='float32', **kwargs):
    input_data = mx.sym.Variable(name="data")
    if dtype == 'float16':
        input_data = mx.sym.Cast(data=input_data, dtype=np.float16)
    # stage 1
    conv1 = mx.sym.Convolution(name='conv1',
        data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu")
    lrn1 = mx.sym.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool1 = mx.sym.Pooling(
        data=lrn1, pool_type="max", kernel=(3, 3), stride=(2,2))
    # stage 2
    conv2 = mx.sym.Convolution(name='conv2',
        data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.sym.Activation(data=conv2, act_type="relu")
    lrn2 = mx.sym.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool2 = mx.sym.Pooling(data=lrn2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 3
    conv3 = mx.sym.Convolution(name='conv3',
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.sym.Activation(data=conv3, act_type="relu")
    conv4 = mx.sym.Convolution(name='conv4',
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.sym.Activation(data=conv4, act_type="relu")
    conv5 = mx.sym.Convolution(name='conv5',
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.sym.Activation(data=conv5, act_type="relu")
    pool3 = mx.sym.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.sym.Flatten(data=pool3)
    fc1 = mx.sym.FullyConnected(name='fc1', data=flatten, num_hidden=4096)
    relu6 = mx.sym.Activation(data=fc1, act_type="relu")
    dropout1 = mx.sym.Dropout(data=relu6, p=0.5)
    # stage 5
    fc2 = mx.sym.FullyConnected(name='fc2', data=dropout1, num_hidden=4096)
    relu7 = mx.sym.Activation(data=fc2, act_type="relu")
    dropout2 = mx.sym.Dropout(data=relu7, p=0.5)
    # stage 6
    fc3 = mx.sym.FullyConnected(name='fc3', data=dropout2, num_hidden=num_classes)
    if dtype == 'float16':
        fc3 = mx.sym.Cast(data=fc3, dtype=np.float32)
    softmax = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return softmax
