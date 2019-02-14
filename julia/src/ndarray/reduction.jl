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

Base.prod(x::NDArray; dims = :) = _prod(x, dims)
@_remap _prod(x::NDArray, ::Colon) prod(x)
@_remap _prod(x::NDArray, dims)    prod(x; axis = 0 .- dims, keepdims = true)

Base.maximum(x::NDArray; dims = :) = _nd_maximum(x, dims)
@_remap _nd_maximum(x::NDArray, ::Colon) max(x)
@_remap _nd_maximum(x::NDArray, dims)    max(x; axis = 0 .- dims, keepdims = true)

Base.minimum(x::NDArray; dims = :) = _nd_minimum(x, dims)
@_remap _nd_minimum(x::NDArray, ::Colon) min(x)
@_remap _nd_minimum(x::NDArray, dims)    min(x; axis = 0 .- dims, keepdims = true)

###############################################################################
# min/max
###############################################################################

import Base: min, max

broadcasted(::typeof(max), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_maximum(x, y)

broadcasted(::typeof(min), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_minimum(x, y)

###############################################################################
# argmin/argmax
###############################################################################

# TODO: support CartesianIndex ?
"""
    argmax(x::NDArray; dims) -> indices

Note that `NaN` is skipped during comparison.
This is different from Julia `Base.argmax`.

## Examples

```julia-repl
julia> x = NDArray([0. 1 2; 3 4 5])
2×3 NDArray{Float64,2} @ CPU0:
 0.0  1.0  2.0
 3.0  4.0  5.0

julia> argmax(x, dims = 1)
1×3 NDArray{Float64,2} @ CPU0:
 2.0  2.0  2.0

julia> argmax(x, dims = 2)
2×1 NDArray{Float64,2} @ CPU0:
 3.0
 3.0
```

See also [`argmin`](@ref mx.argmin).
"""
Base.argmax(x::NDArray; dims = :) = _argmax(x, dims) .+ 1
@_remap _argmax(x::NDArray, ::Colon) argmax(x)
@_remap _argmax(x::NDArray, dims)    argmax(x; axis = 0 .- dims, keepdims = true)

"""
    argmin(x::NDArray; dims) -> indices

Note that `NaN` is skipped during comparison.
This is different from Julia `Base.argmin`.

## Examples

```julia-repl
julia> x = NDArray([0. 1 2; 3 4 5])
2×3 NDArray{Float64,2} @ CPU0:
 0.0  1.0  2.0
 3.0  4.0  5.0

julia> argmax(x, dims = 1)
1×3 NDArray{Float64,2} @ CPU0:
 2.0  2.0  2.0

julia> argmax(x, dims = 2)
2×1 NDArray{Float64,2} @ CPU0:
 3.0
 3.0
```

See also [`argmax`](@ref mx.argmax).
"""
Base.argmin(x::NDArray; dims = :) = _argmin(x, dims) .+ 1
@_remap _argmin(x::NDArray, ::Colon) argmin(x)
@_remap _argmin(x::NDArray, dims)    argmin(x; axis = 0 .- dims, keepdims = true)

################################################################################
# remapping to solving type unstablility
################################################################################

@_remap _broadcast_maximum(x::NDArray, y::NDArray)  broadcast_maximum(x, y)
@_remap _broadcast_maximum!(x::NDArray, y::NDArray) broadcast_maximum(x, y)

@_remap _broadcast_minimum(x::NDArray, y::NDArray)  broadcast_minimum(x, y)
@_remap _broadcast_minimum!(x::NDArray, y::NDArray) broadcast_minimum(x, y)
