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
    RMSProp(; kwargs...)

Scale learning rates by dividing with the moving average of the root mean
squared (RMS) gradients. See [1] for further description.

### Arguments

* `η`: default `0.1`, learning rate.
* `ρ`: default `0.9`, gradient moving average decay factor.
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

### Notes
`ρ` should be between 0 and 1. A value of `ρ` close to 1 will decay the
moving average slowly and a value close to 0 will decay the moving average
fast.

Using the step size `η` and a decay factor `ρ the
learning rate `ηₜ` is calculated as:

```math
\begin{align*}
  r_t &= ρ r_{t-1} + (1 - ρ)g^2 \\
  η_t &= \frac{η}{\sqrt{r_t + ϵ}}
\end{align*}
```

### References
1. Tieleman, T. and Hinton, G. (2012):
   Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
   Coursera. [http://www.youtube.com/watch?v=O3sxAc4hxZU]
   (http://www.youtube.com/watch?v=O3sxAc4hxZU) (formula @5:20)
"""
RMSProp

@defstruct RMSProp <: AbstractOptimizer (
  (η      :: Real = 0.001, η > 0),
  (ρ      :: Real = 0.9,   0 < ρ < 1),
  (ϵ      :: Real = 1e-8,  ϵ > 0),
  (clip   :: Real = 0,     clip >= 0),
   scale  :: Real = 0,
  (λ      :: Real = 1e-5,  λ >= 0),
  η_sched :: Any  = initlrsched(η)
)

create_state(::RMSProp, ::Int, W::NDArray) = zeros(size(W), context(W))

function update!(rms::RMSProp, ::Int, W::NDArray, ∇::NDArray, s::NDArray)
  η = get(rms.η_sched)
  ρ = rms.ρ
  ϵ = rms.ϵ

  normgrad!(rms, W, ∇)

  @inplace s .*= ρ
  @inplace s .+= (1 - ρ) .* (∇.^2)
  @inplace W .+= -η .* ∇ ./ sqrt(s .+ ϵ)  # FIXME: sqrt should be dot-call
end
