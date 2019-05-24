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

import Base: +

"""
    +(args...)
    .+(args...)

Summation. Multiple arguments of either scalar or `NDArray` could be
added together. Note at least the first or second argument needs to be an
`NDArray` to avoid ambiguity of built-in summation.
"""
+(x::NDArray)             = x
+(x::NDArray, y::NDArray) = _plus(x, y)
+(x::NDArray, y::Real)    = _plus_scalar(x, scalar = y)
+(y::Real,    x::NDArray) = _plus_scalar(x, scalar = y)

broadcasted(::typeof(+), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_add(x, y)

"""
    sub_from!(dst::NDArray, args::NDArrayOrReal...)

Subtract a bunch of arguments from `dst`. Inplace updating.
"""
function sub_from!(dst::NDArray, arg::NDArrayOrReal)
  @assert dst.writable
  if isa(arg, Real)
    _minus_scalar(dst, scalar = arg, out = dst)
  else
    _minus!(dst, arg)
  end
  dst
end

import Base: -

"""
    -(x::NDArray)
    -(x, y)
    .-(x, y)

Subtraction `x - y`, of scalar types or `NDArray`.
Or create the negative of `x`.
"""
-(x::NDArray)             = _mul_scalar(x, scalar = -one(eltype(x)))
-(x::NDArray, y::NDArray) = _minus(x, y)
-(x::NDArray, y::Real)    = _minus_scalar(x, scalar = y)
-(y::Real, x::NDArray)    = _rminus_scalar(x, scalar = y)

broadcasted(::typeof(-), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_minus(x, y)

"""
    mul_to!(dst::NDArray, arg::NDArrayOrReal)

Elementwise multiplication into `dst` of either a scalar or an `NDArray` of the same shape.
Inplace updating.
"""
function mul_to!(dst::NDArray, arg::NDArrayOrReal)
  @assert dst.writable
  if isa(arg, Real)
    _mul_scalar(dst, scalar = arg, out = dst)
  else
    _mul(dst, arg, out = dst)
  end
  dst
end

import Base: *

"""
    .*(x, y)

Elementwise multiplication for `NDArray`.
"""
*(x::NDArray, y::Real)  = _mul_scalar(x, scalar = y)
*(y::Real, x::NDArray)  = _mul_scalar(x, scalar = y)

broadcasted(::typeof(*), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N} =
  _mul(x, y)
broadcasted(::typeof(*), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_mul(x, y)

"""
    *(A::NDArray, B::NDArray)

Matrix/tensor multiplication.
"""
*(x::NDArray{T}, y::NDArray{T}) where T = x ⋅ y

LinearAlgebra.adjoint(x::NDArray{T,1}) where T = transpose(x)
LinearAlgebra.adjoint(x::NDArray{T,2}) where T = transpose(x)

"""
    div_from!(dst::NDArray, arg::NDArrayOrReal)

Elementwise divide a scalar or an `NDArray` of the same shape from `dst`. Inplace updating.
"""
function div_from!(dst::NDArray, arg::NDArrayOrReal)
  @assert dst.writable
  if isa(arg, Real)
    _div_scalar(dst, scalar = arg, out = dst)
  else
    _div(dst, arg, out = dst)
  end
  dst
end

function div_from!(dst::NDArray{T}, arg::Real) where {T<:Integer}
  @assert dst.writable
  @assert(round(T, arg) != zero(T), "Integer divided by zero")
  _div_scalar(dst, scalar = arg, out = dst)
  dst
end

"""
    rdiv_from!(x:: Real, y::NDArray)

Elementwise divide a scalar by an `NDArray`. Inplace updating.
"""
function rdiv_from!(x::Real, y::NDArray)
  @assert y.writable
  _rdiv_scalar(y, scalar = x, out = y)
  y
end

import Base: /

"""
    ./(x::NDArray, y::NDArray)
    ./(x::NDArray, y::Real)
    ./(x::Real, y::NDArray)

* Elementwise dividing an `NDArray` by a scalar or another `NDArray`
of the same shape.

* Elementwise divide a scalar by an `NDArray`.

* Matrix division (solving linear systems) is not implemented yet.
"""
/(x::NDArray, y::Real) = _div_scalar(x, scalar = y)

broadcasted(::typeof(/), y::Real, x::NDArray) = _rdiv_scalar(x, scalar = y)
broadcasted(::typeof(/), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N} =
  _div(x, y)
broadcasted(::typeof(/), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_div(x, y)

function broadcasted(::typeof(/), x::NDArray{T}, y::Real) where {T<:Integer}
  @assert(round(T, y) != zero(T), "Integer divided by zero")
  _div_scalar(x, scalar = y)
end

"""
    mod_from!(x::NDArray, y::NDArray)
    mod_from!(x::NDArray, y::Real)

Elementwise modulo for `NDArray`.
Inplace updating.
"""
mod_from!(x::NDArray, y::NDArray) = _mod!(x, y)
mod_from!(x::NDArray, y::Real)    = _mod_scalar!(x, y)

"""
    rmod_from!(y::Real, x::NDArray)

Elementwise modulo for `NDArray`.
Inplace updating.
"""
rmod_from!(y::Real, x::NDArray) = _rmod_scalar!(x, y)

import Base: %

"""
    .%(x::NDArray, y::NDArray)
    .%(x::NDArray, y::Real)
    .%(x::Real, y::NDArray)

Elementwise modulo for `NDArray`.
"""
%(x::NDArray, y::Real) = _mod_scalar(x, y)

broadcasted(::typeof(%), y::Real, x::NDArray) = _rmod_scalar(x, y)
broadcasted(::typeof(%), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N} =
  _mod(x, y)
broadcasted(::typeof(%), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_mod(x, y)

# document of `.^` is merged into SymbolicNode's

broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::NDArray, ::Val{s}) where {s} =
  _power_scalar(x, scalar = s)
broadcasted(::typeof(^), x::NDArray, s::Real) = _power_scalar(x,  scalar = s)
broadcasted(::typeof(^), s::Real, x::NDArray) = _rpower_scalar(x, scalar = s)

broadcasted(::typeof(^), ::Irrational{:ℯ}, x::NDArray) = exp(x)
broadcasted(::typeof(^), x::NDArray, s::Irrational)    = _power_scalar(x, scalar = s)
broadcasted(::typeof(^), s::Irrational, x::NDArray)    = _rpower_scalar(x, scalar = s)

broadcasted(::typeof(^), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N} =
  _power(x, y)
broadcasted(::typeof(^), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_power(x, y)

"""
    clamp(x::NDArray, lo, hi)

Clamps (limits) the values in `NDArray`.
Given an interval, values outside the interval are clipped to the interval edges.
Clamping `x` between low `lo` and high `hi` would be:

```julia
clamp(x, lo, hi) = max(min(x, lo), hi))
```

The storage type of clip output depends on storage types of inputs and the
`lo`, `hi` parameter values:

- clamp(default) -> default
- clamp(row_sparse, lo <= 0, hi >= 0) -> row_sparse
- clamp(csr, lo <= 0, hi >= 0) -> csr
- clamp(row_sparse, lo < 0, hi < 0) -> default
- clamp(row_sparse, lo > 0, hi > 0) -> default
- clamp(csr, lo < 0, hi < 0) -> csr
- clamp(csr, lo > 0, hi > 0) -> csr

## Examples

```jldoctest
julia> x = NDArray(1:9);

julia> clamp(x, 2, 8)'
1×9 mx.NDArray{Int64,2} @ CPU0:
 2  2  3  4  5  6  7  8  8

julia> clamp(x, 8, 2)'
1×9 NDArray{Int64,2} @ CPU0:
 8  8  2  2  2  2  2  2  2
 ```
"""
Base.clamp(x::NDArray, lo::Real, hi::Real) = _clamp(x, lo, hi)
@_remap _clamp(x::NDArray, lo::Real, hi::Real) clip(x; a_min = lo, a_max = hi)

"""
    clamp!(x::NDArray, lo, hi)

See also [`clamp`](@ref).
"""
Base.clamp!(x::NDArray, lo::Real, hi::Real) = _clamp!(x, lo, hi)
@_remap _clamp!(x::NDArray, lo::Real, hi::Real) clip(x; a_min = lo, a_max = hi)

################################################################################
# remapping to solving type unstablility
################################################################################

@_remap _plus(x::NDArray, y::NDArray)  _plus(x, y)
@_remap _plus!(x::NDArray, y::NDArray) _plus(x, y)

@_remap _minus(x::NDArray, y::NDArray)  _minus(x, y)
@_remap _minus!(x::NDArray, y::NDArray) _minus(x, y)

@_remap _mod(x::NDArray, y::NDArray)  _mod(x, y)
@_remap _mod!(x::NDArray, y::NDArray) _mod(x, y)

@_remap _mod_scalar(x::NDArray, y::Real)  _mod_scalar(x; scalar = y)
@_remap _mod_scalar!(x::NDArray, y::Real) _mod_scalar(x; scalar = y)

@_remap _rmod_scalar(x::NDArray, y::Real)  _rmod_scalar(x; scalar = y)
@_remap _rmod_scalar!(x::NDArray, y::Real) _rmod_scalar(x; scalar = y)

@_remap _broadcast_add(x::NDArray, y::NDArray)  broadcast_add(x, y)
@_remap _broadcast_add!(x::NDArray, y::NDArray) broadcast_add(x, y)

@_remap _broadcast_minus(x::NDArray, y::NDArray)  broadcast_minus(x, y)
@_remap _broadcast_minus!(x::NDArray, y::NDArray) broadcast_minus(x, y)

@_remap _broadcast_mul(x::NDArray, y::NDArray)  broadcast_mul(x, y)
@_remap _broadcast_mul!(x::NDArray, y::NDArray) broadcast_mul(x, y)

@_remap _broadcast_div(x::NDArray, y::NDArray)  broadcast_div(x, y)
@_remap _broadcast_div!(x::NDArray, y::NDArray) broadcast_div(x, y)

@_remap _broadcast_mod(x::NDArray, y::NDArray)  broadcast_mod(x, y)
@_remap _broadcast_mod!(x::NDArray, y::NDArray) broadcast_mod(x, y)

@_remap _broadcast_power(x::NDArray, y::NDArray)  broadcast_power(x, y)
@_remap _broadcast_power!(x::NDArray, y::NDArray) broadcast_power(x, y)
