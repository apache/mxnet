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
    Nadam(; kwargs...)

Nesterov Adam optimizer: Adam RMSprop with Nesterov momentum,
see [1] and notes for further description.


### Arguments
* `η`: default `0.001`, learning rate.
* `β1`: default `0.99`.
* `β2`: default `0.999`.
* `ϵ`: default `1e-8`, small value added for numerical stability.
* `clip`: default `0`, gradient clipping.
  If positive, will clip the gradient into the range `[-clip, clip]`.
* `scale`: default `0`, gradient rescaling.
  If != 0, multiply the gradient with `scale` before updating.
  Often choose to be `1.0 / batch_size`.
  If leave it default, high-level API like `fit!` will set it to
  `1.0 / batch_size`, since `fit!` knows the `batch_size`.
* `λ`: default `0.00001`, weight decay is equivalent
  to adding a global l2 regularizer for all the parameters.
* `η_sched::AbstractLearningRateScheduler`: default `nothing`, a
  dynamic learning rate scheduler. If set, will overwrite the `η`
  parameter.
* `μ_sched::NadamScheduler` default `NadamScheduler()` of the form.

  ```math
  \mu_t = β_1 (1 - 0.5 \times 0.96^{t \times 0.004})
  ```

### Notes
Default parameters follow those provided in the paper.
It is recommended to leave the parameters of this optimizer
at their default values.

### References
1. [Incorporating Nesterov Momentum into Adam]
   (http://cs229.stanford.edu/proj2015/054_report.pdf).

2. [On the importance of initialization and momentum in deep learning]
   (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf).
"""
Nadam

@defstruct Nadam <: AbstractOptimizer (
  (η      :: Real = 0.001, η > 0),
  (β1     :: Real = 0.99,  0 <= β1 < 1),
  (β2     :: Real = 0.999, 0 <= β2 < 1),
  (ϵ      :: Real = 1e-8,  ϵ > 0),
  (clip   :: Real = 0,     clip >= 0),
   scale  :: Real = 0,
  (λ      :: Real = 1e-5,  λ >= 0),
  η_sched :: Any = initlrsched(η),
  μ_sched :: Momentum.NadamScheduler = Momentum.NadamScheduler(μ = β1)
)

mutable struct NadamState
  m  :: NDArray
  n  :: NDArray
  Πμ  :: Float64
  β2ᵗ :: Float64
  t  :: Int  # use in NadamScheduler.
             # we store `t` in state because state is created for each `index`
end

create_state(n::Nadam, ::Int, W::NDArray) =
  NadamState(zeros(size(W), context(W)), zeros(size(W), context(W)),
             1.0, n.β2, 1)

function update!(na::Nadam, ::Int, W::NDArray, ∇::NDArray, s::NadamState)
  η = get(na.η_sched)
  μₜ, μₜ₁= get(na.μ_sched, s.t)
  β1, β2 = na.β1, na.β2
  ϵ = na.ϵ

  normgrad!(na, W, ∇)
  s.t += 1

  s.Πμ *= μₜ
  Πμ′ = s.Πμ * μₜ₁

  ∇′ = ∇ / (1.0 - s.Πμ)
  @inplace s.m .*= β1
  @inplace s.m .+= (1.0 - β1) * ∇
  m̂ = s.m / (1.0 - Πμ′)

  @inplace s.n .*= β2
  @inplace s.n .+= (1.0 - β2) .* ∇.^2
  n̂ = s.n / (1.0 - s.β2ᵗ)
  s.β2ᵗ *= β2

  m̄ = (1.0 - μₜ) * ∇′+ μₜ₁ * m̂
  @inplace W .+= -η * m̄ ./ (sqrt(n̂) + ϵ)
end
