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

Elementwise summation of `SymbolicNode`.
"""
function +(x::SymbolicNode, ys::SymbolicNodeOrReal...)
  ret = x
  for y ∈ ys
    if y isa SymbolicNode
      ret = _plus(ret, y)
    else
      ret = _plus_scalar(ret, scalar=MX_float(y))
    end
  end
  ret
end

+(s::Real, x::SymbolicNode, ys::SymbolicNodeOrReal...) = +(x + s, ys...)

broadcasted(::typeof(+), x::SymbolicNode, ys::SymbolicNodeOrReal...) = +(x, ys...)
broadcasted(::typeof(+), s::Real, x::SymbolicNode, ys::SymbolicNodeOrReal...) = +(x + s, ys...)

import Base: -

"""
    -(x, y)
    .-(x, y)

Elementwise substraction of `SymbolicNode`.
Operating with `Real` is available.
"""
x::SymbolicNode - y::SymbolicNode = _minus(x, y)
x::SymbolicNode - s::Real         = _minus_scalar(x,  scalar=MX_float(s))
s::Real         - x::SymbolicNode = _rminus_scalar(x, scalar=MX_float(s))

-(x::SymbolicNode) = 0 - x

broadcasted(::typeof(-), x::SymbolicNode, y::SymbolicNodeOrReal) = x - y
broadcasted(::typeof(-), s::Real, x::SymbolicNode) = s - x

import Base: *

"""
    .*(x, y)

Elementwise multiplication of `SymbolicNode`.
"""
x::SymbolicNode * s::Real = _mul_scalar(x, scalar=MX_float(s))
s::Real * x::SymbolicNode = _mul_scalar(x, scalar=MX_float(s))

function broadcasted(::typeof(*), x::SymbolicNode, ys::SymbolicNodeOrReal...)
  ret = x
  for y in ys
    if y isa SymbolicNode
      ret = _mul(ret, y)
    else
      ret = _mul_scalar(ret, scalar=MX_float(y))
    end
  end
  ret
end

broadcasted(::typeof(*), s::Real, x::SymbolicNode, ys::SymbolicNodeOrReal...) =
  broadcasted(*, x * s, ys...)

import Base: /

"""
    ./(x, y)

* Elementwise dividing a `SymbolicNode` by a scalar or another `SymbolicNode`
of the same shape.

* Elementwise divide a scalar by an `SymbolicNode`.

* Matrix division (solving linear systems) is not implemented yet.
"""
x::SymbolicNode / s::Real = _DivScalar(x, scalar=MX_float(s))

broadcasted(::typeof(/), x::SymbolicNode, y::SymbolicNode) = _div(x, y)
broadcasted(::typeof(/), x::SymbolicNode, s::Real) = _div_scalar(x,  scalar=MX_float(s))
broadcasted(::typeof(/), s::Real, x::SymbolicNode) = _rdiv_scalar(x, scalar=MX_float(s))


import Base: ^

"""
    .^(x, y)

Elementwise power of `SymbolicNode` and `NDArray`.
Operating with `Real` is available.
"""
^

broadcasted(::typeof(^), x::SymbolicNode, y::SymbolicNode) = _power(x, y)
broadcasted(::typeof(^), x::SymbolicNode, s::Real) = _power_scalar(x,  scalar = s)
broadcasted(::typeof(^), s::Real, x::SymbolicNode) = _rpower_scalar(x, scalar = s)
broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::SymbolicNode, ::Val{s}) where {s} =
  _power_scalar(x, scalar = s)

broadcasted(::typeof(^), ::Irrational{:ℯ}, x::SymbolicNode) = exp(x)
broadcasted(::typeof(^), x::SymbolicNode, s::Irrational) =
  _power_scalar(x, scalar=MX_float(s))
broadcasted(::typeof(^), s::Irrational, x::SymbolicNode) =
  _rpower_scalar(x, scalar=MX_float(s))


