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

"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""

import mxnet as mx
import numpy as np

def get_feature(internel_layer, layers, filters, batch_norm = False, **kwargs):
    for i, num in enumerate(layers):
        for j in range(num):
            internel_layer = mx.sym.Convolution(data = internel_layer, kernel=(3, 3), pad=(1, 1), num_filter=filters[i], name="conv%s_%s" %(i + 1, j + 1))
            if batch_norm:
                internel_layer = mx.symbol.BatchNorm(data=internel_layer, name="bn%s_%s" %(i + 1, j + 1))
            internel_layer = mx.sym.Activation(data=internel_layer, act_type="relu", name="relu%s_%s" %(i + 1, j + 1))
        internel_layer = mx.sym.Pooling(data=internel_layer, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool%s" %(i + 1))
    return internel_layer

def get_classifier(input_data, num_classes, **kwargs):
    flatten = mx.sym.Flatten(data=input_data, name="flatten")
    fc6 = mx.sym.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.sym.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.sym.Dropout(data=relu6, p=0.5, name="drop6")
    fc7 = mx.sym.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.sym.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.sym.Dropout(data=relu7, p=0.5, name="drop7")
    fc8 = mx.sym.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    return fc8

def get_symbol(num_classes, num_layers=11, batch_norm=False, dtype='float32', **kwargs):
    """
    Parameters
    ----------
    num_classes : int, default 1000
        Number of classification classes.
    num_layers : int
        Number of layers for the variant of densenet. Options are 11, 13, 16, 19.
    batch_norm : bool, default False
        Use batch normalization.
    dtype: str, float32 or float16
        Data precision.
    """
    vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
                13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
                16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
                19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}
    if num_layers not in vgg_spec:
        raise ValueError("Invalide num_layers {}. Possible choices are 11,13,16,19.".format(num_layers))
    layers, filters = vgg_spec[num_layers]
    data = mx.sym.Variable(name="data")
    if dtype == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)
    feature = get_feature(data, layers, filters, batch_norm)
    classifier = get_classifier(feature, num_classes)
    if dtype == 'float16':
        classifier = mx.sym.Cast(data=classifier, dtype=np.float32)
    symbol = mx.sym.SoftmaxOutput(data=classifier, name='softmax')
    return symbol
