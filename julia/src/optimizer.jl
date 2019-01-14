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

###############################################################################
#  Types
###############################################################################

"""
    AbstractOptimizer

Base type for all optimizers.
"""
abstract type AbstractOptimizer end

"""
    AbstractLearningRateScheduler

Base type for all learning rate scheduler.
"""
abstract type AbstractLearningRateScheduler end

"""
    AbstractMomentumScheduler

Base type for all momentum scheduler.
"""
abstract type AbstractMomentumScheduler end

"""
    OptimizationState

### Attributes
* `batch_size`: The size of the mini-batch used in stochastic training.
* `curr_epoch`:
  The current epoch count. Epoch 0 means no training yet, during the first
  pass through the data, the epoch will be 1; during the second pass, the
  epoch count will be 1, and so on.
* `curr_batch`:
  The current mini-batch count. The batch count is reset during every epoch.
  The batch count 0 means the beginning of each epoch, with no mini-batch
  seen yet. During the first mini-batch, the mini-batch count will be 1.
* `curr_iter`:
  The current iteration count. One iteration corresponds to one mini-batch,
  but unlike the mini-batch count, the iteration count does **not** reset
  in each epoch. So it track the *total* number of mini-batches seen so far.
"""
mutable struct OptimizationState
  batch_size :: Int
  curr_epoch :: Int
  curr_batch :: Int
  curr_iter  :: Int
end

OptimizationState(batch_size::Int) = OptimizationState(batch_size, 0, 0, 0)

###############################################################################
#  LearningRate module
###############################################################################

module LearningRate

using Markdown

import Base: get
import ..mx: AbstractLearningRateScheduler, OptimizationState, update!

export initlrsched

initlrsched(η::Real) = LearningRate.Fixed(η)

update!(a::AbstractLearningRateScheduler) = (isdefined(a, :t) && (a.t += 1))

"""
    get(sched::AbstractLearningRateScheduler)

Returns the current learning rate.
"""
get(::AbstractLearningRateScheduler) = nothing

"""
    LearningRate.Fixed(η)

Fixed learning rate scheduler always return the same learning rate.
"""
struct Fixed <: AbstractLearningRateScheduler
  η::Float64
end

get(f::Fixed) = f.η

@doc doc"""
    LearningRate.Exp(η₀; γ = 0.9)

```math
\eta_t = \eta_0\gamma^t
```

Where `t` is the epoch count, or the iteration count.
"""
mutable struct Exp <: AbstractLearningRateScheduler
  η₀::Float64
  γ ::Float64
  t ::Int
end

function Exp(η₀; γ = 0.9, t = 0)
  @assert 0 < γ < 1
  Exp(η₀, γ, t)
end

get(a::Exp) = a.η₀ * a.γ^a.t

@doc doc"""
    LearningRate.Inv(η₀; γ = 0.9, p = 0.5)

```math
\eta_t = \eta_0 (1 + \gamma t)^{-p}
```

Where `t` is the epoch count, or the iteration count.
"""
mutable struct Inv <: AbstractLearningRateScheduler
  η₀::Float64
  γ ::Float64
  p ::Float64
  t ::Int
end

function Inv(η₀; γ = 0.9, p = 0.5, t = 0)
  @assert 0 < γ < 1
  @assert 0 <= p
  Inv(η₀, γ, p, t)
end

get(i::Inv) = i.η₀ * (1 + i.γ*i.t)^(-i.p)

end  # module LearningRate

using .LearningRate

###############################################################################
#  Momentum module
###############################################################################

module Momentum

using Markdown

import Base: get
import ..mx: AbstractMomentumScheduler, OptimizationState

export initmomsched

"""
    get(sched)

* `sched::AbstractMomentumScheduler`: the momentum scheduler.

Returns the current momentum.
"""
get

initmomsched(μ::Real) = iszero(μ) ? Momentum.Null() : Momentum.Fixed(μ)

"""
    Momentum.Null

The null momentum scheduler always returns 0 for momentum. It is also used to
explicitly indicate momentum should not be used.
"""
struct Null <: AbstractMomentumScheduler
end

get(::Null) = 0.0

"""
    Momentum.Fixed

Fixed momentum scheduler always returns the same value.
"""
mutable struct Fixed <: AbstractMomentumScheduler
  μ::Float64
end

get(f::Fixed) = f.μ

@doc doc"""
    NadamScheduler(; μ = 0.99, δ = 0.004, γ = 0.5, α = 0.96)

Nesterov-accelerated adaptive momentum scheduler.

Description in [Incorporating Nesterov Momentum into Adam]
(http://cs229.stanford.edu/proj2015/054_report.pdf).

```math
\mu_t = \mu_0 * (1 - \gamma * \alpha^{t * \delta})
```

Where
* `t`: iteration count
* `μ`: default `0.99`, μ₀
* `δ`: default `0.004` is scheduler decay.
* `γ`: default `0.5`
* `α`: default `0.96`
"""
struct NadamScheduler <: AbstractMomentumScheduler
  μ::Float64
  δ::Float64
  γ::Float64
  α::Float64
end

function NadamScheduler(; μ = 0.99, δ = 0.004, γ = 0.5, α = 0.96)
  @assert 0.0 <= μ < 1.0
  @assert 0.0 <= δ
  @assert 0.0 <= γ <= 1.0
  @assert 0.0 <= α <= 1.0
  NadamScheduler(μ, δ, γ, α)
end

"""
    get(n::NadamScheduler, t)

Where `t` is the iteration count.
"""
get(n::NadamScheduler, t) =
  n.μ * (1.0 - n.γ * n.α^( t      * n.δ)),
  n.μ * (1.0 - n.γ * n.α^((t + 1) * n.δ))

end  # module Momentum

using .Momentum

###############################################################################
# Public APIs
###############################################################################

"""
    getupdater(optimizer)

A utility function to create an updater function of `KVStore`,
that uses its closure to store all the states needed for each weights.

Ther returned function has following signature:

```julia
decend!(index::Int, ∇::NDArray, x::NDArray)
```

If the optimizer is stateful and need access/store states during updating,
`index` will be the key to access/store states.
"""
function getupdater(optimizer::AbstractOptimizer)
  states = Dict{Int,Any}()
  function updater(index::Int, ∇::NDArray, x::NDArray)
    if !haskey(states, index)
      states[index] = create_state(optimizer, index, x)
    end
    update!(optimizer, index, x, ∇, states[index])
  end
  updater
end

"""
    normgrad(optimizer, W, ∇)

Get the properly normalized gradient (re-scaled and clipped if necessary).

* `optimizer`: the optimizer,
  should contain the field `scale`, `clip` and `λ`.
* `W::NDArray`: the trainable weights.
* `∇::NDArray`: the original gradient of the weights.
"""
function normgrad!(opt::AbstractOptimizer, W::NDArray, ∇::NDArray)
  # rescaling
  s = opt.scale
  !iszero(s) && @inplace ∇ .*= s
  # gradient clipping
  c = opt.clip
  c > 0 && clip!(∇, -c, c)
  # weight decay
  λ = opt.λ
  λ > 0 && @inplace ∇ += λ .* W

  ∇
end

###############################################################################
# Builtin Optimizers
###############################################################################

include("optimizers/sgd.jl")
include("optimizers/adam.jl")
include("optimizers/adagrad.jl")
include("optimizers/adadelta.jl")
include("optimizers/adamax.jl")
include("optimizers/rmsprop.jl")
include("optimizers/nadam.jl")
