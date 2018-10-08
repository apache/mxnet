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
    AbstractCallback

Abstract type of callback functions used in training.
"""
abstract type AbstractCallback end

"""
    AbstractBatchCallback

Abstract type of callbacks to be called every mini-batch.
"""
abstract type AbstractBatchCallback <: AbstractCallback end

"""
    AbstractEpochCallback

Abstract type of callbacks to be called every epoch.
"""
abstract type AbstractEpochCallback <: AbstractCallback end

mutable struct BatchCallback <: AbstractBatchCallback
  frequency :: Int
  call_on_0 :: Bool
  callback  :: Function
end

"""
    every_n_batch(callback :: Function, n :: Int; call_on_0 = false)

A convenient function to construct a callback that runs every `n` mini-batches.

# Arguments
* `call_on_0::Bool`: keyword argument, default false. Unless set, the callback
          will *not* be run on batch 0.

For example, the [`speedometer`](@ref) callback is defined as

```julia
every_n_batch(frequency, call_on_0=true) do state :: OptimizationState
  if state.curr_batch == 0
    # reset timer
  else
    # compute and print speed
  end
end
```

See also [`every_n_epoch`](@ref) and [`speedometer`](@ref).
"""
function every_n_batch(callback::Function, n::Int; call_on_0::Bool = false)
  BatchCallback(n, call_on_0, callback)
end
function (cb :: BatchCallback)(state :: OptimizationState)
  if state.curr_batch == 0
    if cb.call_on_0
      cb.callback(state)
    end
  elseif state.curr_batch % cb.frequency == 0
    cb.callback(state)
  end
end

"""
    speedometer(;frequency=50)

Create an `AbstractBatchCallback` that measure the training speed
   (number of samples processed per second) every k mini-batches.

# Arguments
* `frequency::Int`: keyword argument, default 50. The frequency (number of
          min-batches) to measure and report the speed.
"""
function speedometer(;frequency::Int = 50)
  cl_tic = 0
  every_n_batch(frequency, call_on_0 = true) do state::OptimizationState
    if state.curr_batch == 0
      # reset timer
      cl_tic = time()
    else
      speed = frequency * state.batch_size / (time() - cl_tic)
      info(format("Speed: {1:>6.2f} samples/sec", speed))
      cl_tic = time()
    end
  end
end


mutable struct EpochCallback <: AbstractEpochCallback
  frequency :: Int
  call_on_0 :: Bool
  callback  :: Function
end

"""
    every_n_epoch(callback :: Function, n :: Int; call_on_0 = false)

A convenient function to construct a callback that runs every `n` full data-passes.

* `call_on_0::Bool`: keyword argument, default false. Unless set, the callback
          will *not* be run on epoch 0. Epoch 0 means no training has been performed
          yet. This is useful if you want to inspect the randomly initialized model
          that has not seen any data yet.

See also [`every_n_batch`](@ref).
"""
every_n_epoch(callback::Function, n::Int; call_on_0::Bool = false) =
  EpochCallback(n, call_on_0, callback)

function (cb::EpochCallback)(model::Any, state::OptimizationState,
                             metric::Vector{Tuple{Symbol, T}}) where T<:Real
  if state.curr_epoch == 0
    if cb.call_on_0
      cb.callback(model, state, metric)
    end
  elseif state.curr_epoch % cb.frequency == 0
    cb.callback(model, state, metric)
  end
end

"""
    do_checkpoint(prefix; frequency=1, save_epoch_0=false)

Create an `AbstractEpochCallback` that save checkpoints of the model to disk.
The checkpoints can be loaded back later on.

# Arguments
* `prefix::AbstractString`: the prefix of the filenames to save the model.
  The model architecture will be saved to prefix-symbol.json,
  while the weights will be saved to prefix-0012.params,
  for example, for the 12-th epoch.
* `frequency::Int`: keyword argument, default is 1.
  The frequency (measured in epochs) to save checkpoints.
* `save_epoch_0::Bool`: keyword argument, default false. Whether we should save a
  checkpoint for epoch 0 (model initialized but not seen any data yet).
"""
function do_checkpoint(prefix::AbstractString;
                       frequency::Int = 1, save_epoch_0::Bool = false)
  mkpath(dirname(prefix))
  every_n_epoch(frequency, call_on_0=save_epoch_0) do model, state, metric
    save_checkpoint(model, prefix, state)
  end
end
