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

require(mxnet)

data <- mx.symbol.Variable('data')
net1 <- mx.symbol.FullyConnected(data = data, name = 'fc1', num_hidden = 10)
net1 <- mx.symbol.FullyConnected(data = net1, name = 'fc2', num_hidden = 100)

all.equal(arguments(net1), c('data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias'))

net2 <- mx.symbol.FullyConnected(name = 'fc3', num_hidden = 10)
net2 <- mx.symbol.Activation(data = net2, act_type = 'relu')
net2 <- mx.symbol.FullyConnected(data = net2, name = 'fc4', num_hidden = 20)

composed <- mx.apply(net2, fc3_data = net1, name = 'composed')
