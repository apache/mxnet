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
a simple multilayer perceptron
"""
import mxnet as mx

def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=1536)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 1536)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3 =  mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=1536)
    act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="relu")    
    fc4  = mx.symbol.FullyConnected(data = act3, name='fc4', num_hidden=1536)
    act4 = mx.symbol.Activation(data = fc4, name='relu4', act_type="relu")
    fc5  = mx.symbol.FullyConnected(data = act4, name='fc5', num_hidden=3101)
    mlp  = mx.symbol.SoftmaxOutput(data = fc5, name = 'softmax')
    return mlp
