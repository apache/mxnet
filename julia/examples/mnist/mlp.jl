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
# define MLP
# the following two ways are equivalent

#-- Option 1: explicit composition
# data = mx.Variable(:data)
# fc1  = mx.FullyConnected(data, name=:fc1, num_hidden=128)
# act1 = mx.Activation(fc1, name=:relu1, act_type=:relu)
# fc2  = mx.FullyConnected(act1, name=:fc2, num_hidden=64)
# act2 = mx.Activation(fc2, name=:relu2, act_type=:relu)
# fc3  = mx.FullyConnected(act2, name=:fc3, num_hidden=10)
# mlp  = mx.SoftmaxOutput(fc3, name=:softmax)

#-- Option 2: using the mx.chain macro
# mlp = @mx.chain mx.Variable(:data)             =>
#   mx.FullyConnected(name=:fc1, num_hidden=128) =>
#   mx.Activation(name=:relu1, act_type=:relu)   =>
#   mx.FullyConnected(name=:fc2, num_hidden=64)  =>
#   mx.Activation(name=:relu2, act_type=:relu)   =>
#   mx.FullyConnected(name=:fc3, num_hidden=10)  =>
#   mx.SoftmaxOutput(name=:softmax)

#-- Option 3: using nn-factory
mlp = @mx.chain mx.Variable(:data) =>
  mx.MLP([128, 64, 10])            =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size)

# setup model
model = mx.FeedForward(mlp, context=mx.cpu())

# optimizer
optimizer = mx.SGD(η=0.1, μ=0.9, λ=0.00001)

# fit parameters
mx.fit(model, optimizer, train_provider, eval_data=eval_provider, n_epoch=20)

#--------------------------------------------------------------------------------
# Optional, demonstration of the predict API
probs = mx.predict(model, eval_provider)

# collect all labels from eval data
labels = reduce(
  vcat,
  copy(mx.get(eval_provider, batch, :softmax_label)) for batch ∈ eval_provider)
# labels are 0...9
labels .= labels .+ 1

# Now we use compute the accuracy
pred = map(i -> indmax(probs[1:10, i]), 1:size(probs, 2))
correct = sum(pred .== labels)
@printf "Accuracy on eval set: %.2f%%\n" 100correct/length(labels)
