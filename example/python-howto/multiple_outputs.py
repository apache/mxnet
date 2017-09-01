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

"""Create a Multiple output configuration.

This example shows how to create a multiple output configuration.
"""
import mxnet as mx

net = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
net = mx.symbol.FullyConnected(data=net, name='fc2', num_hidden=64)
out = mx.symbol.SoftmaxOutput(data=net, name='softmax')
# group fc1 and out together
group = mx.symbol.Group([fc1, out])
print group.list_outputs()

# You can go ahead and bind on the group
# executor = group.simple_bind(data=data_shape)
# executor.forward()
# executor.output[0] will be value of fc1
# executor.output[1] will be value of softmax
