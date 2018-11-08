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
     ADAM

The solver described in Diederik Kingma, Jimmy Ba: *Adam: A Method for
Stochastic Optimization*. arXiv:1412.6980 [cs.LG].

    ADAM(; kwargs...)

### Arguments
* `η`: default `0.001`, learning rate.
* `β1`: default `0.9`.
* `β2`: default `0.999`.
* `ϵ`: default `1e-8`.
* `clip`: default `0`, gradient clipping.
  If positive, will clip the gradient into the range `[-clip, clip]`.
* `scale`: default `0`, gradient rescaling.
  If != 0, multiply the gradient with `scale` before updating.
  Often choose to be `1.0 / batch_size`.
  If leave it default, high-level API like `fit!` will set it to
  `1.0 / batch_size`, since `fit!` knows the `batch_size`.
* `λ`: default `0.00001`, weight decay is equivalent
  to adding a global l2 regularizer for all the parameters.
* `η_sched::AbstractLearningRateScheduler`: default `LearningRate.Fixed(η)`, a
  dynamic learning rate scheduler. If set, will overwrite the `η` parameter.
"""
ADAM

@defstruct ADAM <: AbstractOptimizer (
  (η      :: Real = 0.001, η > 0),
  (β1     :: Real = 0.9,   0 <= β1 < 1),
  (β2     :: Real = 0.999, 0 <= β2 < 1),
  (ϵ      :: Real = 1e-8,  ϵ > 0),
  (clip   :: Real = 0,     clip >= 0),
   scale  :: Real = 0,
  (λ      :: Real = 1e-5,  λ >= 0),
  η_sched :: Any  = initlrsched(η)
)

mutable struct ADAMState
  η   :: Float64  # current learning rate
  mₜ  :: NDArray
  vₜ  :: NDArray
  β1ᵗ :: Float64
  β2ᵗ :: Float64
end

create_state(adam::ADAM, ::Int, W::NDArray) =
  ADAMState(get(adam.η_sched),
            zeros(size(W), context(W)),
            zeros(size(W), context(W)),
            adam.β1, adam.β2)

function update!(adam::ADAM, ::Int, W::NDArray, ∇:: NDArray, s::ADAMState)
  η = s.η
  β1 = adam.β1
  β2 = adam.β2
  ϵ = adam.ϵ

  normgrad!(adam, W, ∇)

  s.mₜ = β1 * s.mₜ + (1 - β1) .* ∇
  s.vₜ = β2 * s.vₜ + (1 - β2) .* ∇.^2

  aₜ= sqrt(1.0 - s.β2ᵗ)/(1.0 - s.β1ᵗ)

  # update βᵗ to βᵗ⁺¹
  s.β1ᵗ *= β1
  s.β2ᵗ *= β2

  @inplace W .+= -η * aₜ * s.mₜ ./ (sqrt(s.vₜ) .+ ϵ)
end
