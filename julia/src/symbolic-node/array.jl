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

# Base.Array related interface

import Base: reshape

"""
    reshape(sym::SymbolicNode, dim; reverse=false, name)
    reshape(sym::SymbolicNode, dim...; reverse=false, name)

Reshape SymbolicNode operator

Some dimensions of the shape can take special values from the set
{0, -1, -2, -3, -4}.
The significance of each is explained below:

- `0`  copy this dimension from the input to the output shape.

  Example:

  - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
  - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)

- `-1` infers the dimension of the output shape by using the remainder of the
  input dimensions keeping the size of the new array same as that of the input
  array. At most one dimension of shape can be -1.

  Example:

  - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
  - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
  - input shape = (2,3,4), shape=(-1,), output shape = (24,)

- `-2` copy all/remainder of the input dimensions to the output shape.

  Example:

  - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
  - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
  - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)

- `-3` use the product of two consecutive dimensions of the input shape as the
  output dimension.

  Example:

  - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
  - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
  - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
  - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)

- `-4` split one dimension of the input into two dimensions passed subsequent
  to -4 in shape (can contain -1).

  Example:

  - input shape = (2,3,4), shape = (-4,1,2,-2), output shape = (1,2,3,4)
  - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)

If the argument `reverse` is set to `1`, then the special values are inferred
from right to left.

  Example:

  - with `reverse=false`, for input shape = (10,5,4), shape = (-1,0),
    output shape would be (40,5)
  - with `reverse=true`, output shape will be (50,4).
"""
reshape(sym::SymbolicNode, dim::NTuple{N, Integer}; kwargs...) where {N} =
  _reshape(sym, dim; kwargs...)
reshape(sym::SymbolicNode, dim::Integer...; kwargs...) =
  _reshape(sym, dim; kwargs...)

@inline function _reshape(sym::SymbolicNode, dim::NTuple{N,Integer};
                          reverse::Bool=false, name::String="") where N
  op = _get_cached_libmx_op_handle("reshape")
  node = _create_atomic_symbol(op.value, ["shape", "reverse"],
                               [dump_mx_param(dim), dump_mx_param(!reverse)])
  name = get!(DEFAULT_NAME_MANAGER, name, "reshape")
  _compose!(node, name=name, data=sym)
end

################################################################################
# Base.getindex
################################################################################

"""
    getindex(self :: SymbolicNode, idx :: Union{Int, Base.Symbol, AbstractString})

Get a node representing the specified output of this node. The index could be
a symbol or string indicating the name of the output, or a 1-based integer
indicating the index, as in the list of [`list_outputs`](@ref).
"""
function Base.getindex(self :: SymbolicNode, idx :: Union{Base.Symbol, AbstractString})
  idx   = Symbol(idx)
  i_idx = findall(idx .== list_outputs(self))
  @assert(length(i_idx) > 0, "Cannot find output with name '$idx'")
  @assert(length(i_idx) < 2, "Found duplicated output with name '$idx'")
  Base.getindex(self, i_idx[1])
end
function Base.getindex(self :: SymbolicNode, idx :: Int)
  ref_hdr = Ref{MX_handle}(0)
  # note Julia is 1-based, while MXNet is 0-based
  @mxcall(:MXSymbolGetOutput, (MX_handle, MX_uint, Ref{MX_handle}), self, idx-1, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

