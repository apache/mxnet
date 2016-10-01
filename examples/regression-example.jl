#=
This script shows how a simple MLP net may be used
for regression. It shows how data in memory may be
used for training and evaluation, and how to obtain
the predictions from the trained net.

TO DO:
    * specify batch size, and allow different sizes
      for the training and evaluation sets
    * tanh activation does not seem to work properly,
      investigate
=#
using MXNet
using Distributions
using PyPlot

# data generating process for exogenous inputs
generate_inputs(media, var, tam) = rand(MvNormal(media, var), tam)

# function that maps inputs to outputs
f1(data) = sin(data[1,:]).*sin(data[2,:])./(data[1,:].*data[2,:])

# parameters for input d.g.p.
mean=[0.0;0.0]
var=[1.0 0.0;0.0 1.0]

# create training and evaluation data sets
TrainInput = generate_inputs(mean, var, 5000)
TrainOutput = f1(TrainInput)
ValidationInput = generate_inputs(mean, var, 5000)
ValidationOutput = f1(ValidationInput)

# how to set up data providers using data in memory
trainprovider = mx.ArrayDataProvider(:data => TrainInput, :label => TrainOutput)
evalprovider = mx.ArrayDataProvider(:data => ValidationInput, :label => ValidationOutput)

# create a single hidden layer MPL
data = mx.Variable(:data)
label = mx.Variable(:label)
fc1  = mx.FullyConnected(data = data, name=:fc1, num_hidden=20)
act1 = mx.Activation(data = fc1, name=:relu, act_type=:relu)
fc2  = mx.FullyConnected(data = act1, name=:fc2, num_hidden=1)

# cost is squared error loss
cost = mx.LinearRegressionOutput(data=fc2, label=label, name = :loss)

# final model definition
model = mx.FeedForward(cost, context=mx.cpu())

# set up the optimizer
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)

# train, reporting loss for training and evaluation sets
mx.fit(model, optimizer, eval_metric=mx.MSE(), trainprovider, eval_data=evalprovider, n_epoch = 1000)

# obtain predictions
fit = mx.predict(model, evalprovider)
plot(ValidationOutput,fit',".")
xlabel("true")
ylabel("predicted")
title("outputs: true versus predicted. 45º line is what we hope for")
 
