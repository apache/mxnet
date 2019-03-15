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
    AdaGrad(; kwargs...)

Scale learning rates by dividing with the square root of accumulated
squared gradients. See [1] for further description.

### Arguments
* `η`: default `0.1`, learning rate.
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
Using step size `η` AdaGrad calculates the learning rate for feature `i` at
time step t as:

```math
η_{t,i} = \frac{lr}{\sqrt{\sum^t_{t^\prime} g^2_{t^\prime,i} + ϵ}} g_{t,i}
```

as such the learning rate is monotonically decreasing.
Epsilon is not included in the typical formula, see [2].

### References
1. Duchi, J., Hazan, E., & Singer, Y. (2011):
   Adaptive subgradient methods for online learning and
   stochastic optimization. JMLR, 12:2121-2159.
2. Chris Dyer: Notes on AdaGrad.
   [http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf]
   (http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf)
"""
AdaGrad

@defstruct AdaGrad <: AbstractOptimizer (
  (η      :: Real = 0.1,  η > 0),
  (ϵ      :: Real = 1e-6, ϵ > 0),
  (clip   :: Real = 0,    clip >= 0),
   scale  :: Real = 0,
  (λ      :: Real = 1e-5, λ >= 0),
  η_sched :: Any  = initlrsched(η)
)

create_state(::AdaGrad, ::Int, W::NDArray) = zeros(size(W), context(W))

function update!(ada::AdaGrad, ::Int, W::NDArray, ∇::NDArray, x::NDArray)
  η = get(ada.η_sched)
  ϵ = ada.ϵ

  normgrad!(ada, W, ∇)

  @inplace x .+= ∇.^2  # update state
  @inplace W .+= -η .* ∇ ./ sqrt(x .+ ϵ)  # FIXME: sqrt dot-call
end
