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

#=
This script shows how a simple MLP net may be used
for regression. It shows how data in memory may be
used for training and evaluation, and how to obtain
the predictions from the trained net.
=#
using MXNet
using Distributions
#using Plots

# data generating process
generate_inputs(mean, var, size) = rand(MvNormal(mean, var), size)
output(data) = sin.(data[1:1,:]).*sin.(data[2:2,:])./(data[1:1,:].*data[2:2,:])

# create training and evaluation data sets
mean=[0.0; 0.0]
var=[1.0 0.0; 0.0 1.0]
samplesize  = 5000
TrainInput = generate_inputs(mean, var, samplesize)
TrainOutput = output(TrainInput)
ValidationInput = generate_inputs(mean, var, samplesize)
ValidationOutput = output(ValidationInput)

# how to set up data providers using data in memory
function data_source(batchsize = 100)
  train = mx.ArrayDataProvider(
    :data => TrainInput,
    :label => TrainOutput,
    batch_size = batchsize,
    shuffle = true,
    )
  valid = mx.ArrayDataProvider(
    :data => ValidationInput,
    :label => ValidationOutput,
    batch_size = batchsize,
    shuffle = true,
    )

  train, valid
end

# create a two hidden layer MPL: try varying num_hidden, and change tanh to relu,
# or add/remove a layer
data = mx.Variable(:data)
label = mx.Variable(:label)
net = @mx.chain     mx.Variable(:data) =>
                    mx.FullyConnected(num_hidden=10) =>
                    mx.Activation(act_type=:tanh) =>
                    mx.FullyConnected(num_hidden=3) =>
                    mx.Activation(act_type=:tanh) =>
                    mx.FullyConnected(num_hidden=1) =>
                    mx.LinearRegressionOutput(mx.Variable(:label))

# final model definition, don't change, except if using gpu
model = mx.FeedForward(net, context=mx.cpu())

# set up the optimizer: select one, explore parameters, if desired
#optimizer = mx.SGD(η=0.01, μ=0.9, λ=0.00001)
optimizer = mx.ADAM()

# train, reporting loss for training and evaluation sets
# initial training with small batch size, to get to a good neighborhood
trainprovider, evalprovider = data_source(#= batchsize =# 200)
mx.fit(model, optimizer, trainprovider,
       initializer = mx.NormalInitializer(0.0, 0.1),
       eval_metric = mx.MSE(),
       eval_data = evalprovider,
       n_epoch = 20,
       callbacks = [mx.speedometer()])
# more training with the full sample
trainprovider, evalprovider = data_source(#= batchsize =# samplesize)
mx.fit(model, optimizer, trainprovider,
       initializer = mx.NormalInitializer(0.0, 0.1),
       eval_metric = mx.MSE(),
       eval_data = evalprovider,
       n_epoch = 500,  # previous setting is batchsize = 200, epoch = 20
                       # implies we did (5000 / 200) * 20 times update in previous `fit`
       callbacks = [mx.speedometer()])

# obtain predictions
plotprovider = mx.ArrayDataProvider(:data => ValidationInput, :label => ValidationOutput)
fit = mx.predict(model, plotprovider)
println("correlation between fitted values and true regression line: ", cor(vec(fit), vec(ValidationOutput)))
#scatter(ValidationOutput',fit',w = 3, xlabel="true", ylabel="predicted", title="45º line is what we hope for", show=true)
