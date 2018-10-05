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
    AdaMax(; kwargs...)

This is a variant of of the Adam algorithm based on the infinity norm.
See [1] for further description.

### Arguments
* `η`: default `0.002`, learning rate.
* `β1`: default `0.9`, exponential decay rate for the first moment estimates.
* `β2`: default `0.999`, exponential decay rate for the weighted
  infinity norm estimates.
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

### References
1. Kingma, Diederik, and Jimmy Ba (2014):
   Adam: A Method for Stochastic Optimization. Section 7.
   [http://arxiv.org/abs/1412.6980]
   (http://arxiv.org/abs/1412.6980).
"""
AdaMax

@defstruct AdaMax <: AbstractOptimizer (
  (η      :: Real = 0.002, η > 0),
  (β1     :: Real = 0.9,   0 <= β1 < 1),
  (β2     :: Real = 0.999, 0 <= β2 < 1),
  (ϵ      :: Real = 1e-8,  ϵ > 0),
  (clip   :: Real = 0,     clip >= 0),
   scale  :: Real = 0,
  (λ      :: Real = 1e-5,  λ >= 0),
  η_sched :: Any  = initlrsched(η)
)

mutable struct AdaMaxState
  mₜ  :: NDArray
  uₜ  :: NDArray
  β1ᵗ :: Float64
end

create_state(ada::AdaMax, ::Int, W::NDArray) =
  AdaMaxState(zeros(size(W), context(W)),
              zeros(size(W), context(W)),
              ada.β1)

function update!(ada::AdaMax, ::Int, W::NDArray, ∇::NDArray, s::AdaMaxState)
  η = get(ada.η_sched)
  β1 = ada.β1
  β2 = ada.β2
  ϵ = ada.ϵ

  normgrad!(ada, W, ∇)

  s.mₜ = β1 * s.mₜ .+ (1 - β1) .* ∇
  s.uₜ = _maximum(β2 * s.uₜ, abs(∇))  # FIXME abs dot-call

  @inplace W .+= -η / (1 - s.β1ᵗ) * s.mₜ ./ (s.uₜ + ϵ)

  s.β1ᵗ *= ada.β1
end
