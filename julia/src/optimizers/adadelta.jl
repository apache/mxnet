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
    AdaDelta(; kwargs...)

Scale learning rates by the ratio of accumulated gradients to accumulated
updates, see [1] and notes for further description.

### Attributes
* `η`: default `1.0`, learning rate.
* `ρ`: default `0.95`, squared gradient moving average decay factor.
* `ϵ`: default `1e-6`, small value added for numerical stability.
* `clip`: default `0`, gradient clipping.
  If positive, will clip the gradient into the range `[-clip, clip]`.
* `scale`: default `0`, gradient rescaling.
  If != 0, multiply the gradient with `scale` before updating.
  Often choose to be `1.0 / batch_size`.
  If leave it default, high-level API like `fit!` will set it to
  `1.0 / batch_size`, since `fit!` knows the `batch_size`.
* `λ`: default `0.00001`, weight decay is equivalent
  to adding a global l2 regularizer for all the parameters.

### Notes
`ρ` should be between 0 and 1. A value of `ρ` close to 1 will decay the
moving average slowly and a value close to 0 will decay the moving average
fast.

`ρ = 0.95` and `ϵ = 1e-6` are suggested in the paper and reported to
work for multiple datasets (MNIST, speech). In the paper, no learning rate is
considered (so `η = 1.0`). Probably best to keep it at this value.

`ϵ` is important for the very first update (so the numerator does not become 0).

Using the step size `η` and a decay factor `ρ` the learning rate is
calculated as:

```math
\begin{align*}
  r_t &= ρ r_{t-1} + (1 - ρ) g^2 \\
  η_t &= η \frac{\sqrt{s_{t-1} + ϵ}} {\sqrt{r_t + ϵ}} \\
  s_t &= ρ s_{t-1} + (1 - ρ) (η_t \times g)^2
\end{align*}
```

### References
1. Zeiler, M. D. (2012):
   ADADELTA: An Adaptive Learning Rate Method. arXiv Preprint arXiv:1212.5701.
"""
AdaDelta

@defstruct AdaDelta <: AbstractOptimizer (
  (η      :: Real = 1.0,  η > 0),
  (ρ      :: Real = 0.95, 0 < ρ < 1 ),
  (ϵ      :: Real = 1e-6, ϵ > 0),
  (clip   :: Real = 0,    clip >= 0),
   scale  :: Real = 0,
  (λ      :: Real = 1e-5, λ >= 0),
  η_sched :: Any  = initlrsched(η)
)

mutable struct AdaDeltaState
  x  :: NDArray
  Δx :: NDArray
end

create_state(::AdaDelta, ::Int, W::NDArray) =
  AdaDeltaState(zeros(size(W), context(W)), zeros(size(W), context(W)))

function update!(ada::AdaDelta, ::Int, W::NDArray, ∇::NDArray, s::AdaDeltaState)
  η  = get(ada.η_sched)
  x  = s.x
  Δx = s.Δx
  ρ  = ada.ρ
  ϵ  = ada.ϵ

  normgrad!(ada, W, ∇)

  # Update s.acc as in RMSProp
  @inplace x .*= ρ
  @inplace x .+= (1 - ρ) .* ∇.^2

  # Compute update using the "old" Δx
  Δxₜ = ∇ .* sqrt(Δx .+ ϵ) ./ sqrt(x .+ ϵ)  # FIXME: sqrt dot-call
  @inplace W .+= -η .* Δxₜ

  # update Δx using update
  @inplace Δx .*= ρ
  @inplace Δx .+= (1 - ρ) .* Δxₜ.^2
end
