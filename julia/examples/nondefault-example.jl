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
    Contents: This file contains code for:
              - Setting the initial values of the biases and weights equal to the final values of a previous run.
	        This is helpful for re-estimating a model on updated training data, where the original and updated training data largely overlap.
	      - Changing the loss function (in our example from Accuracy to ACE)

    Notes:
    1. The model is a toy example with 4 outcomes (categories).
       The model is a poor fit to the data, but this is unimportant. The point of the example is to demonstrate the use of some non-default settings.
    2. For categorical outcomes, use 0-based categories! Some of the loss functions assume this, such as ACE.
    3. Incomplete batches are padded with repeated instances of an artificial observation.
       This is bad because the artificial data is over-represented and thus biases the results.
       The ideal solution is to distribute the observations from the incomplete batch among the complete batches.
       This would result in batches of variable but similar size, and thus the estimate of the gradient would not be significantly affected.
       But this doesn't happen.
       For simplicity we instead drop these extra observations, so that the number of observations in the data set is a multiple of the batch_size.
=#


using RDatasets
using MXNet


################################################################################
### Data: Exam scores discretised into 4 categories (use zero-based categories!).
df = dataset("mlmRev", "Gcsemv");    # 1905 x 5
complete_cases!(df)                  # 1523 x 5
n = nrow(df)
df[:written] = zeros(Int, n)
df[:course]  = zeros(Int, n)
for i = 1:n
    # Categorise :Written
    if df[i, :Written] <= 20.0
	df[i, :written] = 0
    elseif df[i, :Written] <= 40.0
	df[i, :written] = 1
    elseif df[i, :Written] <= 60.0
	df[i, :written] = 2
    else
	df[i, :written] = 3
    end

    # Categorise :Course
    if df[i, :Course] <= 25.0
	df[i, :course] = 0
    elseif df[i, :Course] <= 50.0
	df[i, :course] = 1
    elseif df[i, :Course] <= 75.0
	df[i, :course] = 2
    else
	df[i, :course] = 3
    end
end
df = df[1:1500, :]    # Ensure nrows is a multiple of batch_size (100 in our example, see below)

x = convert(Vector{Float64}, df[:course])
y = convert(Vector{Float64}, df[:written])


################################################################################
### Hyperparameters

# Architecture
mlp = @mx.chain mx.Variable(:data) =>
        mx.FullyConnected(name = :h1, num_hidden = 10) =>
	mx.Activation(name = :h1_out, act_type = :sigmoid) =>
        mx.FullyConnected(name = :out, num_hidden = 4) =>
	mx.SoftmaxOutput(name = :softmax)

# Hyperparameters
n_epoch    = 100
batch_size = 100
learn_rate = 0.1
mom        = 0.9
wt_decay   = 0.00001


# Connect data, network architecture and hyperparameters
train_prov = mx.ArrayDataProvider(x, y; batch_size = batch_size)
eval_prov  = mx.ArrayDataProvider(x, y; batch_size = batch_size)
opt        = mx.SGD(lr = learn_rate, momentum = mom, weight_decay = wt_decay)    # Optimizing algorithm

################################################################################
### Run 1: Basic run, storing initial and final state.

# Learn
mdl1 = mx.FeedForward(mlp, context = mx.cpu())                                               # Model targets the local CPU
cb = mx.do_checkpoint("first", frequency = n_epoch, save_epoch_0 = true)                     # Write initial and final states to disk
mx.fit(mdl1, opt, train_prov, n_epoch = n_epoch, eval_data = eval_prov, callbacks = [cb])    # Random initial biases and weights


################################################################################
### Run 2: Load the previously trained model and run it some more, starting where Run 1 finished.

# Load final state of 1st run from disk
arch, arg_params, aux_params = mx.load_checkpoint("first", 100)    # arch is the network structure, arg_params contains the weights and biases
mdl2 = mx.FeedForward(arch, context = mx.cpu())                    # Only populates the arch and ctx fields
mdl2.arg_params = arg_params                                       # Populate the arg_params fields
cb   = mx.do_checkpoint("second", frequency = n_epoch, save_epoch_0 = true)
mx.fit(mdl2, opt, train_prov, n_epoch = n_epoch, eval_data = eval_prov, callbacks = [cb])

# Test whether the final state of 1st run equals the initial state of 2nd run
run(`diff first-0100.params second-0000.params`)    # Throws error if not true, does nothing otherwise


#=
    # Other useful functions
    arch       = mx.load("first-symbol.json", mx.SymbolicNode)
    arg_params = mx.load("first-0100.params", mx.NDArray)
=#


################################################################################
### Run 3: Change the loss function from the default Accuracy to ACE

mdl3 = mx.FeedForward(mlp, context = mx.cpu())
mx.fit(mdl3, opt, train_prov, n_epoch = n_epoch, eval_data = eval_prov, eval_metric = mx.ACE())
#mx.fit(mdl3, opt, train_prov, n_epoch = n_epoch, eval_data = eval_prov, eval_metric = mx.Accuracy())    # Default eval_metric
#mx.fit(mdl3, opt, train_prov, n_epoch = n_epoch, eval_data = eval_prov, eval_metric = mx.MultiACE(4))

# Test manually
probs = mx.predict(mdl3, eval_prov)
LL    = 0.0
for i = 1:size(y, 1)
    LL += log(probs[Int(y[i]) + 1, i])
end
-LL / size(y, 1)    # Should equal the value of ACE from the final iteration of fit(mdl3, ...)


# EOF
