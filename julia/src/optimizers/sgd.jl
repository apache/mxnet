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

@doc doc"""
    SGD(; kwargs...)

Stochastic gradient descent optimizer.

Vanilla SGD:

```math
\theta \leftarrow \theta - \eta \nabla
```

SGD with momentum::

```math
\begin{align*}
  \nu    & \leftarrow \mu \nu_{t-1} - \eta \nabla \\
  \theta & \leftarrow \theta + \nu_t
\end{align*}
```

### Arguments

* `η`: default `0.01`, learning rate.
* `μ`: default `0`, the momentum, usually set to `0.9` in this implementation.
* `λ`: default `0.0001`, weight decay is equivalent to
  adding a global l2 regularizer to the parameters.
* `clip`: default `0`, gradient clipping.
  If positive, will clip the gradient into the bounded range `[-clip, clip]`.
* `scale`: default `0`, gradient rescaling.
  If != 0, multiply the gradient with `scale` before updating.
  Often choose to be `1.0 / batch_size`.
  If leave it default, high-level API like `fit!` will set it to
  `1.0 / batch_size`, since `fit!` knows the `batch_size`.
* `μ_sched::AbstractMomentumScheduler`: default `Momentum.Null()`,
  a dynamic momentum scheduler. If set, will overwrite the `momentum`
  parameter.
* `η_sched::AbstractLearningRateScheduler`: default `LearningRate.Fixed(η)`, a
  dynamic learning rate scheduler. If set, will overwrite the `η` parameter.
"""
SGD

@defstruct SGD <: AbstractOptimizer (
  (η      :: Real = 0.01,   η > 0),
  (μ      :: Real = 0.0,    μ >= 0),
  (clip   :: Real = 0,      clip >= 0),
   scale  :: Real = 0,
  (λ      :: Real = 0.0001, λ >= 0),
  η_sched :: Any  = initlrsched(η),
  μ_sched :: Any  = initmomsched(μ)
)

create_state(sgd::SGD, ::Int, W::NDArray) =
  isa(sgd.μ_sched, Momentum.Null) ? nothing : zeros(size(W), context(W))

function update!(sgd::SGD, ::Int, W::NDArray, ∇::NDArray, ::Nothing)
  η = get(sgd.η_sched)
  normgrad!(sgd, W, ∇)
  @inplace W += -η * ∇
end

# update with momentum
function update!(sgd::SGD, ::Int, W::NDArray, ∇::NDArray, ν::NDArray)
  η = get(sgd.η_sched)
  μ = get(sgd.μ_sched)

  normgrad!(sgd, W, ∇)

  @inplace ν .*= μ
  @inplace ν .+= -η .* ∇
  @inplace W .+= ν
end
