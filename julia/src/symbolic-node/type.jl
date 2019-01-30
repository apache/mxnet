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
    SymbolicNode

SymbolicNode is the basic building block of the symbolic graph in MXNet.jl.
It's a callable object and supports following calls:

    (s::SymbolicNode)(args::SymbolicNode...)
    (s::SymbolicNode)(; kwargs...)

Make a new node by composing `s` with `args`. Or the arguments
can be specified using keyword arguments.
"""
mutable struct SymbolicNode
  handle::MX_SymbolHandle
end

const SymbolicNodeOrReal = Union{SymbolicNode,Real}

Base.unsafe_convert(::Type{MX_handle}, s::SymbolicNode) =
  Base.unsafe_convert(MX_handle, s.handle)
Base.convert(T::Type{MX_handle}, s::SymbolicNode) = Base.unsafe_convert(T, s)
Base.cconvert(T::Type{MX_handle}, s::SymbolicNode) = Base.unsafe_convert(T, s)

"""
    deepcopy(s::SymbolicNode)

Make a deep copy of a SymbolicNode.
"""
function Base.deepcopy(s::SymbolicNode)
  ref_hdr = Ref{MX_handle}(C_NULL)
  @mxcall(:MXSymbolCopy, (MX_handle, Ref{MX_handle}), s, ref_hdr)
  SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

"""
    copy(s::SymbolicNode)

Make a copy of a SymbolicNode. The same as making a deep copy.
"""
Base.copy(s::SymbolicNode) = Base.deepcopy(s)


function (s::SymbolicNode)(args::SymbolicNode...)
  s = deepcopy(s)
  _compose!(s, args...)
end

function (s::SymbolicNode)(; kwargs...)
  s = deepcopy(s)
  _compose!(s; kwargs...)
end

"""
    Variable(name::Union{Symbol,AbstractString}; attrs)

Create a symbolic variable with the given name. This is typically used as a placeholder.
For example, the data node, acting as the starting point of a network architecture.

## Arguments

* `attrs::Dict{Symbol,<:AbstractString}`: The attributes associated with this `Variable`.
"""
function Variable(name::Union{Symbol,AbstractString}; attrs = Dict())
  attrs = convert(Dict{Symbol, AbstractString}, attrs)
  hdr_ref = Ref{MX_handle}(C_NULL)
  @mxcall(:MXSymbolCreateVariable, (char_p, Ref{MX_handle}), name, hdr_ref)
  node = SymbolicNode(MX_SymbolHandle(hdr_ref[]))
  for (k, v) in attrs
    set_attr(node, k, v)
  end
  node
end

"""
    @var <symbols>...

A handy macro for creating `mx.Variable`.

```julia
julia> x = @mx.var x
MXNet.mx.SymbolicNode x

julia> x, y, z = @mx.var x y z
(MXNet.mx.SymbolicNode x, MXNet.mx.SymbolicNode y, MXNet.mx.SymbolicNode z)
```
"""
macro var(n::Symbol)
  Expr(:call, :Variable, QuoteNode(n))
end

macro var(names::Symbol...)
  Expr(:tuple, map(n -> Expr(:call, :Variable, QuoteNode(n)), names)...)
end

"""
    Group(nodes::SymbolicNode...)

Create a `SymbolicNode` by grouping nodes together.
"""
function Group(nodes::SymbolicNode...)
  handles = MX_handle[nodes...]
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateGroup, (MX_uint, Ptr{MX_handle}, Ref{MX_handle}),
          length(handles), handles, ref_hdr)
  SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end
