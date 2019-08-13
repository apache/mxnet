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
    rand!(x::NDArray; low = 0, high = 1)

Draw random samples from a uniform distribution.
Samples are uniformly distributed over the half-open interval [low, high)
(includes low, but excludes high).

```julia
julia> mx.rand!(NDArray(undef, 2, 3))
2×3 mx.NDArray{Float32,2} @ CPU0:
 0.385748   0.839275  0.444536
 0.0879585  0.215928  0.104636

julia> mx.rand!(NDArray(undef, 2, 3), low = 1, high = 10)
2×3 mx.NDArray{Float32,2} @ CPU0:
 6.6385   4.18888  2.07505
 8.97283  2.5636   1.95586
```
"""
rand!(x::NDArray; low = 0, high = 1) =
  _random_uniform(NDArray, low = low, high = high, shape = size(x), out = x)

"""
    rand(dims...; low = 0, high = 1, context = cpu())

Draw random samples from a uniform distribution.
Samples are uniformly distributed over the half-open interval [low, high)
(includes low, but excludes high).

```julia
julia> mx.rand(2, 2)
2×2 mx.NDArray{Float32,2} @ CPU0:
 0.487866   0.825691
 0.0234245  0.794797

julia> mx.rand(2, 2; low = 1, high = 10)
2×2 mx.NDArray{Float32,2} @ CPU0:
 5.5944   5.74281
 9.81258  3.58068
```
"""
rand(dims::Integer...; low = 0, high = 1, context = cpu()) =
  rand!(NDArray(undef, dims, ctx = context), low = low, high = high)

"""
    randn!(x::NDArray; μ = 0, σ = 1)

Draw random samples from a normal (Gaussian) distribution.
"""
randn!(x::NDArray; μ = 0, σ = 1) =
  _random_normal(NDArray, loc = μ, scale = σ, shape = size(x), out = x)

"""
    randn(dims...; μ = 0, σ = 1, context = cpu())

Draw random samples from a normal (Gaussian) distribution.
"""
randn(dims::Int...; μ = 0, σ = 1, context = cpu()) =
  randn!(NDArray(undef, dims, ctx = context), μ = μ, σ = σ)

"""
    seed!(seed::Int)

Set the random seed of libmxnet
"""
seed!(s::Int) = @mxcall :MXRandomSeed (Cint,) s

function srand(s::Int)
  @warn "`mx.srand` is deprecated, use `mx.seed!` instead."
  seed!(s)
end
