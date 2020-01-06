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

using MXNet

#--------------------------------------------------------------------------------
# define lenet

# input
data = mx.Variable(:data)

# first conv
conv1 = @mx.chain mx.Convolution(data, kernel=(5,5), num_filter=20)  =>
                  mx.Activation(act_type=:tanh) =>
                  mx.Pooling(pool_type=:max, kernel=(2,2), stride=(2,2))

# second conv
conv2 = @mx.chain mx.Convolution(conv1, kernel=(5,5), num_filter=50) =>
                  mx.Activation(act_type=:tanh) =>
                  mx.Pooling(pool_type=:max, kernel=(2,2), stride=(2,2))

# first fully-connected
fc1   = @mx.chain mx.Flatten(conv2) =>
                  mx.FullyConnected(num_hidden=500) =>
                  mx.Activation(act_type=:tanh)

# second fully-connected
fc2   = mx.FullyConnected(fc1, num_hidden=10)

# softmax loss
lenet = mx.SoftmaxOutput(fc2, name=:softmax)


#--------------------------------------------------------------------------------
# load data
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size; flat=false)

#--------------------------------------------------------------------------------
# fit model
model = mx.FeedForward(lenet, context=mx.gpu())

# optimizer
optimizer = mx.SGD(η=0.05, μ=0.9, λ=0.00001)

# fit parameters
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider)
