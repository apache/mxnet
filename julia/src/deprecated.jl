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

# NDArray reshape (#272)
@deprecate reshape(arr::NDArray; shape=()) reshape(arr, shape)
@deprecate Reshape(arr::NDArray; shape=()) reshape(arr, shape)

# SymbolicNode reshape (#279)
@deprecate reshape(sym::SymbolicNode; shape=()) reshape(sym, shape)
@deprecate Reshape(sym::SymbolicNode; shape=()) reshape(sym, shape)

# srand (#282)
@deprecate srand!(seed_state::Int) srand(seed_state)

# v0.4
@deprecate sin(x::NDArray)    sin.(x)
@deprecate cos(x::NDArray)    cos.(x)
@deprecate tan(x::NDArray)    tan.(x)
@deprecate arcsin(x::NDArray) asin.(x)
@deprecate arccos(x::NDArray) acos.(x)
@deprecate arctan(x::NDArray) atan.(x)

@deprecate sinh(x::NDArray)    sinh.(x)
@deprecate cosh(x::NDArray)    cosh.(x)
@deprecate tanh(x::NDArray)    tanh.(x)
@deprecate arcsinh(x::NDArray) asinh.(x)
@deprecate arccosh(x::NDArray) acosh.(x)
@deprecate arctanh(x::NDArray) atanh.(x)

# @deprecate make `randn` exported accidentially
# so we make the depwarn manually
function randn(μ, σ, dims::NTuple{N,Int}, ctx::Context = cpu()) where N
  @warn("mx.randn(μ, σ, dims, ctx = cpu()) is deprecated, use " *
        "mx.randn(dims...; μ = μ, σ = σ, context = ctx) instead.")
  mx.randn(dims...; μ = μ, σ = σ, context = ctx)
end

function randn!(μ, σ, x::NDArray)
  @warn("mx.randn!(μ, σ, x::NDArray) is deprecated, use " *
        "mx.randn!(x; μ = μ, σ = σ) instead.")
  randn!(x; μ = μ, σ = σ)
end

function rand!(low::Real, high::Real, x::NDArray)
  @warn("rand!(low, high, x::NDArray) is deprecated, use " *
        "rand!(x, low = low, high = high) instead.")
  rand!(x, low = low, high = high)
end

function rand(low::Real, high::Real, dims::NTuple{N,Int}, context::Context = cpu()) where N
  @warn("rand!(low, high, dims, x::NDArray, context = cpu()) is deprecated, use " *
        "rand!(dims..., x; low = low, high = high, context = cpu()) instead.")
  rand(dims...; low = low, high = high, context = context)
end

@deprecate sigmoid(x::NDArray)                      sigmoid.(x)
@deprecate relu(x::NDArray)                         relu.(x)
@deprecate softmax(x::NDArray; axis = ndims(x))     softmax.(x, axis)
@deprecate log_softmax(x::NDArray; axis = ndims(x)) log_softmax.(x, axis)

function broadcast_plus(x::NDArray, y::NDArray)
  @warn("broadcast_plus(x, y) is deprecated, use x .+ y instead.")
  x .+ y
end

function broadcast_add(x::NDArray, y::NDArray)
  @warn("broadcast_add(x, y) is deprecated, use x .+ y instead.")
  x .+ y
end

function broadcast_sub(x::NDArray, y::NDArray)
  @warn("broadcast_sub(x, y) is deprecated, use x .- y instead.")
  x .- y
end

function broadcast_minus(x::NDArray, y::NDArray)
  @warn("broadcast_minus(x, y) is deprecated, use x .- y instead.")
  x .- y
end

function broadcast_mul(x::NDArray, y::NDArray)
  @warn("broadcast_mul(x, y) is deprecated, use x .* y instead.")
  x .* y
end

function broadcast_div(x::NDArray, y::NDArray)
  @warn("broadcast_div(x, y) is deprecated, use x ./ y instead.")
  x ./ y
end

function broadcast_mod(x::NDArray, y::NDArray)
  @warn("broadcast_mod(x, y) is deprecated, use x .% y instead.")
  x .% y
end

function broadcast_power(x::NDArray, y::NDArray)
  @warn("broadcast_power(x, y) is deprecated, use x.^y instead.")
  x.^y
end

function broadcast_equal(x::NDArray, y::NDArray)
  @warn("broadcast_equal(x, y) is deprecated, use x .== y instead.")
  x .== y
end

function broadcast_not_equal(x::NDArray, y::NDArray)
  @warn("broadcast_not_equal(x, y) is deprecated, use x .== y instead.")
  x .!= y
end

function broadcast_greater(x::NDArray, y::NDArray)
  @warn("broadcast_greater(x, y) is deprecated, use x .== y instead.")
  x .> y
end

function broadcast_greater_equal(x::NDArray, y::NDArray)
  @warn("broadcast_greater_equal(x, y) is deprecated, use x .== y instead.")
  x .>= y
end

function broadcast_lesser(x::NDArray, y::NDArray)
  @warn("broadcast_lesser(x, y) is deprecated, use x .== y instead.")
  x .< y
end

function broadcast_lesser_equal(x::NDArray, y::NDArray)
  @warn("broadcast_lesser_equal(x, y) is deprecated, use x .== y instead.")
  x .<= y
end

function broadcast_maximum(x::NDArray, y::NDArray)
  @warn("broadcast_maximum(x, y) is deprecated, use max.(x, y) instead.")
  max.(x, y)
end

function broadcast_minimum(x::NDArray, y::NDArray)
  @warn("broadcast_minimum(x, y) is deprecated, use min.(x, y) instead.")
  min.(x, y)
end

function broadcast_hypot(x::NDArray, y::NDArray)
  @warn("broadcast_hypot(x, y) is deprecated, use hypot.(x, y) instead.")
  hypot.(x, y)
end

# Introduced by https://github.com/apache/incubator-mxnet/pull/12845
import Base: sum, maximum, minimum, prod, cat
@deprecate sum(x::NDArray, dims) sum(x, dims = dims)
@deprecate maximum(x::NDArray, dims) maximum(x, dims = dims)
@deprecate minimum(x::NDArray, dims) minimum(x, dims = dims)
@deprecate prod(x::NDArray, dims) prod(x, dims = dims)
@deprecate cat(dims, As::NDArray{T}...) where T cat(As..., dims = dims)

import Statistics: mean
@deprecate mean(x::NDArray, dims) mean(x, dims = dims)

# replaced by UndefInitializer
function empty(::Type{T}, dims::NTuple{N,Int}, ctx::Context = cpu()) where {N,T<:DType}
  @warn("`mx.empty(T, dims, ctx)` is deprecated, " *
        "use `NDArray{T,N}(undef, dims; ctx = ctx)` instead.")
  NDArray{T,N}(undef, dims; ctx = ctx)
end

function empty(::Type{T}, dims::Int...) where {T<:DType}
  @warn("`mx.empty(T, dims...)` is deprecated, " *
        "use `NDArray{T,N}(undef, dims...)` instead.")
  NDArray{T,N}(undef, dims...)
end

function empty(dims::NTuple{N,Int}, ctx::Context = cpu()) where N
  @warn("`mx.empty(dims, ctx)` is deprecated, " *
        "use `NDArray(undef, dims; ctx = ctx)` instead.")
  NDArray(undef, dims; ctx = ctx)
end

function empty(dims::Int...)
  @warn("`mx.empty(dims...)` is deprecated, " *
        "use `NDArray(undef, dims...)` instead.")
  NDArray(undef, dims...)
end

# replaced by Base.clamp
@deprecate clip(x::NDArray, lo::Real, hi::Real)  clamp(x, lo, hi)
@deprecate clip!(x::NDArray, lo::Real, hi::Real) clamp!(x, lo, hi)
@deprecate clip(x; a_min = 0, a_max = 0) clamp(x, a_min, a_max)

