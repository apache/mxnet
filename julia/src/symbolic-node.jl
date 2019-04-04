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

    (self :: SymbolicNode)(args :: SymbolicNode...)
    (self :: SymbolicNode)(; kwargs...)

Make a new node by composing `self` with `args`. Or the arguments
can be specified using keyword arguments.
"""
mutable struct SymbolicNode
  handle::MX_SymbolHandle
end

const SymbolicNodeOrReal = Union{SymbolicNode, Real}

@unfuse SymbolicNode  # for broadcasting

Base.unsafe_convert(::Type{MX_handle}, obj::SymbolicNode) =
  Base.unsafe_convert(MX_handle, obj.handle)
Base.convert(t::Type{MX_handle}, obj::SymbolicNode) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::SymbolicNode) = Base.unsafe_convert(t, obj)

"""
    deepcopy(self :: SymbolicNode)

Make a deep copy of a SymbolicNode.
"""
function Base.deepcopy(self :: SymbolicNode)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCopy, (MX_handle, Ref{MX_handle}), self, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

"""
    copy(self :: SymbolicNode)

Make a copy of a SymbolicNode. The same as making a deep copy.
"""
function Base.copy(self :: SymbolicNode)
  Base.deepcopy(self)
end

function (self::SymbolicNode)(args :: SymbolicNode...)
  s = deepcopy(self)
  _compose!(s, args...)
end
function (self::SymbolicNode)(;kwargs...)
  s = deepcopy(self)
  _compose!(s; kwargs...)
end

macro _list_symbol_info(self, func_name)
  quote
    ref_sz    = Ref{MX_uint}(0)
    ref_names = Ref{char_pp}(0)
    @mxcall($func_name, (MX_handle, Ref{MX_uint}, Ref{char_pp}),
            $(esc(self)), ref_sz, ref_names)
    narg = ref_sz[]
    names = unsafe_wrap(Array, ref_names[], narg)
    names = [Symbol(unsafe_string(x)) for x in names]
    return names
  end
end

"""
    list_arguments(self :: SymbolicNode)

List all the arguments of this node. The argument for a node contains both
the inputs and parameters. For example, a `FullyConnected` node will
have both data and weights in its arguments. A composed node (e.g. a MLP) will
list all the arguments for intermediate nodes.

Returns a list of symbols indicating the names of the arguments.
"""
function list_arguments(self :: SymbolicNode)
  @_list_symbol_info(self, :MXSymbolListArguments)
end

"""
    list_outputs(self :: SymbolicNode)

List all the outputs of this node.

Returns a list of symbols indicating the names of the outputs.
"""
function list_outputs(self :: SymbolicNode)
  @_list_symbol_info(self, :MXSymbolListOutputs)
end


"""
    list_auxiliary_states(self :: SymbolicNode)


List all auxiliary states in the symbool.

Auxiliary states are special states of symbols that do not corresponds to an argument,
and do not have gradient. But still be useful for the specific operations.
A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
Most operators do not have Auxiliary states.

Returns a list of symbols indicating the names of the auxiliary states.
"""
function list_auxiliary_states(self :: SymbolicNode)
  @_list_symbol_info(self, :MXSymbolListAuxiliaryStates)
end

"""
    get_internals(self :: SymbolicNode)

Get a new grouped `SymbolicNode` whose output contains all the internal outputs of
this `SymbolicNode`.
"""
function get_internals(self :: SymbolicNode)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolGetInternals, (MX_handle, Ref{MX_handle}), self, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

"""
    get_children(x::SymbolicNode)

Gets a new grouped `SymbolicNode` whose output contains inputs to output
nodes of the original symbol.

```julia
julia> x = mx.Variable(:x)
MXNet.mx.SymbolicNode x

julia> y = mx.Variable(:y)
MXNet.mx.SymbolicNode y

julia> z = x + y
MXNet.mx.SymbolicNode _plus1

julia> a |> mx.get_children |> mx.list_outputs
2-element Array{Symbol,1}:
 :x
 :y
```
"""
function get_children(x::SymbolicNode)
  hdl = Ref{MX_handle}(C_NULL)
  @mxcall(:MXSymbolGetChildren, (MX_handle, Ref{MX_handle}), x, hdl)
  sym = hdl[] |> MX_SymbolHandle |> SymbolicNode
  isempty(list_outputs(sym)) ? nothing : sym
end

"""
    get_attr(self :: SymbolicNode, key :: Symbol)

Get attribute attached to this `SymbolicNode` belonging to key.

Returns the value belonging to key as a `Nullable`.
"""
function get_attr(self :: SymbolicNode, key :: Symbol)
  key_s = string(key)
  ref_out = Ref{Cstring}()
  ref_success = Ref{Cint}(-1)
  @mxcall(:MXSymbolGetAttr, (MX_handle, Cstring, Ref{Cstring}, Ref{Cint}),
          self, key_s, ref_out, ref_success)
  if ref_success[] == 1
    return Nullable{String}(unsafe_string(ref_out[]))
  else
    return Nullable{String}()
  end
end

"""
    list_attr(self :: SymbolicNode)

Get all attributes from a symbol.

Returns a dictionary of attributes.
"""
function list_attr(self :: SymbolicNode)
  ref_sz    = Ref{MX_uint}(0)
  ref_strings = Ref{char_pp}(0)
  @mxcall(:MXSymbolListAttrShallow, (MX_handle, Ref{MX_uint}, Ref{char_pp}),
            self, ref_sz, ref_strings)
  narg = 2*ref_sz[]
  strings = unsafe_wrap(Array, ref_strings[], narg)
  out = Dict{Symbol, String}()
  for i in 1:2:narg
    key = Symbol(unsafe_string(strings[i]))
    value = unsafe_string(strings[i+1]) # Creates a copy of string
    out[key] = value
  end
  return out
end

"""
    list_all_attr(self :: SymbolicNode)

Get all attributes from the symbol graph.

Returns a dictionary of attributes.
"""
function list_all_attr(self :: SymbolicNode)
  ref_sz    = Ref{MX_uint}(0)
  ref_strings = Ref{char_pp}(0)
  @mxcall(:MXSymbolListAttr, (MX_handle, Ref{MX_uint}, Ref{char_pp}),
            self, ref_sz, ref_strings)
  narg = 2*ref_sz[]
  strings = unsafe_wrap(Array, ref_strings[], narg)
  out = Dict{Symbol, String}()
  for i in 1:2:narg
    key = Symbol(unsafe_string(strings[i]))
    value = unsafe_string(strings[i+1])
    out[key] = value
  end
  return out
end

"""
    set_attr(self:: SymbolicNode, key :: Symbol, value :: AbstractString)

Set the attribute key to value for this `SymbolicNode`.

!!! note
    It is encouraged not to call this function directly, unless you know exactly what you are doing. The
    recommended way of setting attributes is when creating the `SymbolicNode`. Changing
    the attributes of a `SymbolicNode` that is already been used somewhere else might
    cause unexpected behavior and inconsistency.
"""
function set_attr(self :: SymbolicNode, key :: Symbol, value :: AbstractString)
  key_s = string(key)
  value_s = String(value)

  @mxcall(:MXSymbolSetAttr, (MX_handle, Cstring, Cstring), self, key_s, value_s)
end

"""
    get_name(self :: SymbolicNode)

Get the name of the symbol.

    julia> x = mx.Variable(:data)
    julia> mx.get_name(x)
    :data

    julia> y = mx.FullyConnected(x, num_hidden = 128)
    julia> mx.get_name(y)
    :fullyconnected0
"""
function get_name(self :: mx.SymbolicNode)
    name = Ref{mx.char_p}(0)
    success = Ref(0)
    @mxcall(:MXSymbolGetName, (MX_handle, Ref{char_p}, Ref{Int}), self.handle.value, name, success)
    @assert success[] != -1

    str = name[]
    if str == C_NULL  # e.g. the symbol returned via get_internals
        string(self.handle.value)
    else
        Symbol(unsafe_string(str))
    end
end

Base.show(io::IO, sym::SymbolicNode) =
  print(io, "$(typeof(sym)) $(get_name(sym))")

import Base: print

function print(io::IO, sym::SymbolicNode)
  out = Ref{mx.char_p}(C_NULL)
  @mx.mxcall(:MXSymbolPrint, (mx.MX_SymbolHandle, Ref{mx.char_p}), sym.handle, out)
  print(io, unsafe_string(out[]))
end

print(sym::SymbolicNode) = print(STDOUT, sym)

"""
    print([io::IO], sym::SymbolicNode)

Print the content of symbol, used for debug.

```julia
julia> layer = @mx.chain mx.Variable(:data)           =>
         mx.FullyConnected(name=:fc1, num_hidden=128) =>
         mx.Activation(name=:relu1, act_type=:relu)
MXNet.mx.SymbolicNode(MXNet.mx.MX_SymbolHandle(Ptr{Void} @0x000055b29b9c3520))

julia> print(layer)
Symbol Outputs:
        output[0]=relu1(0)
Variable:data
Variable:fc1_weight
Variable:fc1_bias
--------------------
Op:FullyConnected, Name=fc1
Inputs:
        arg[0]=data(0) version=0
        arg[1]=fc1_weight(0) version=0
        arg[2]=fc1_bias(0) version=0
Attrs:
        num_hidden=128
--------------------
Op:Activation, Name=relu1
Inputs:
        arg[0]=fc1(0)
Attrs:
        act_type=relu
```
"""
print

"""
    grad(self :: SymbolicNode, wrt :: Vector{SymbolicNode})

Get the autodiff gradient of the current `SymbolicNode`. This function can
only be used if the current symbol is a loss function.

# Arguments:
* `self::SymbolicNode`: current node.
* `wrt::Vector{Symbol}`: the names of the arguments to the gradient.

Returns a gradient symbol of the corresponding gradient.
"""
function grad(self :: SymbolicNode, wrt :: Vector{Symbol})
  hdr_ref = Ref{MX_handle}(0)
  keys = String[string(key) for key in wrt]

  @mxcall(:MXSymbolGrad, (MX_handle, MX_uint, char_pp, Ptr{MX_handle}), self, length(keys), keys, hdr_ref)
  return SymbolicNode(MX_SymbolHandle(hdr_ref[]))
end

"""
    Variable(name :: Union{Symbol, AbstractString})

Create a symbolic variable with the given name. This is typically used as a placeholder.
For example, the data node, acting as the starting point of a network architecture.

# Arguments
* Dict{Symbol, AbstractString} attrs: The attributes associated with this `Variable`.
"""
function Variable(name :: Union{Symbol, AbstractString}; attrs = Dict())
  attrs = convert(Dict{Symbol, AbstractString}, attrs)
  hdr_ref = Ref{MX_handle}(0)
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
    Group(nodes :: SymbolicNode...)

Create a `SymbolicNode` by grouping nodes together.
"""
function Group(nodes :: SymbolicNode...)
  handles = MX_handle[nodes...]
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateGroup, (MX_uint, Ptr{MX_handle}, Ref{MX_handle}),
          length(handles), handles, ref_hdr)
  SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

function _build_shapes(shape_size::MX_uint, shape_ndim::Ptr{MX_uint}, shape_data::Ptr{Ptr{MX_uint}})
  shape_ndim = unsafe_wrap(Array, shape_ndim, shape_size)
  shape_data = unsafe_wrap(Array, shape_data, shape_size)
  shapes = map(1:shape_size) do i
    my_shape = unsafe_wrap(Array, shape_data[i], shape_ndim[i])
    tuple(flipdim(Int[my_shape...],1)...)
  end
  convert(Vector{Tuple}, shapes)
end

function _infer_shape(self, keys, indptr, sdata)
  ref_arg_shape_size = Ref{MX_uint}(0)
  ref_arg_shape_ndim = Ref{Ptr{MX_uint}}(0)
  ref_arg_shape_data = Ref{Ptr{Ptr{MX_uint}}}(0)
  ref_out_shape_size = Ref{MX_uint}(0)
  ref_out_shape_ndim = Ref{Ptr{MX_uint}}(0)
  ref_out_shape_data = Ref{Ptr{Ptr{MX_uint}}}(0)
  ref_aux_shape_size = Ref{MX_uint}(0)
  ref_aux_shape_ndim = Ref{Ptr{MX_uint}}(0)
  ref_aux_shape_data = Ref{Ptr{Ptr{MX_uint}}}(0)
  ref_complete       = Ref{Cint}(0)
  @mxcall(:MXSymbolInferShape,
          (MX_handle, MX_uint, char_pp, Ptr{MX_uint}, Ptr{MX_uint},
           Ref{MX_uint}, Ref{Ptr{MX_uint}}, Ref{Ptr{Ptr{MX_uint}}},
           Ref{MX_uint}, Ref{Ptr{MX_uint}}, Ref{Ptr{Ptr{MX_uint}}},
           Ref{MX_uint}, Ref{Ptr{MX_uint}}, Ref{Ptr{Ptr{MX_uint}}},
           Ref{Cint}),
          self, length(indptr)-1, keys, indptr, sdata,
          ref_arg_shape_size, ref_arg_shape_ndim, ref_arg_shape_data,
          ref_out_shape_size, ref_out_shape_ndim, ref_out_shape_data,
          ref_aux_shape_size, ref_aux_shape_ndim, ref_aux_shape_data,
          ref_complete)
  if ref_complete[] == 0
    return (nothing, nothing, nothing)
  else
    return (
      _build_shapes(ref_arg_shape_size[], ref_arg_shape_ndim[], ref_arg_shape_data[]),
      _build_shapes(ref_out_shape_size[], ref_out_shape_ndim[], ref_out_shape_data[]),
      _build_shapes(ref_aux_shape_size[], ref_aux_shape_ndim[], ref_aux_shape_data[])
    )
  end
end

"""
    infer_shape(self :: SymbolicNode, args...)
    infer_shape(self :: SymbolicNode; kwargs...)

Do shape inference according to the input shapes. The input shapes could be provided
as a list of shapes, which should specify the shapes of inputs in the same order as
the arguments returned by [`list_arguments`](@ref). Alternatively, the shape information
could be specified via keyword arguments.

Returns a 3-tuple containing shapes of all the arguments, shapes of all the outputs and
shapes of all the auxiliary variables. If shape inference failed due to incomplete
or incompatible inputs, the return value will be `(nothing, nothing, nothing)`.
"""
function infer_shape(self :: SymbolicNode; kwargs...)
  sdata  = MX_uint[]
  indptr = MX_uint[0]
  for (k,v) in kwargs
    append!(sdata, flipdim([v...],1))
    push!(indptr, length(sdata))
  end
  keys = AbstractString[string(x[1]) for x in kwargs]
  _infer_shape(self, keys, indptr, sdata)
end
function infer_shape(self :: SymbolicNode, args :: Union{Tuple, Void}...)
  sdata  = MX_uint[]
  indptr = MX_uint[0]
  for arg in args
    if isa(arg, Void); continue; end
    append!(sdata, flipdim([arg...],1))
    push!(indptr, length(sdata))
  end
  keys = Ptr{char_p}(0)
  _infer_shape(self, keys, indptr, sdata)
end

function _infer_type(self, keys, arg_type_data)
  ref_in_type_size  = Ref{MX_uint}()
  ref_in_type_data  = Ref{Ptr{Cint}}()
  ref_out_type_size = Ref{MX_uint}()
  ref_out_type_data = Ref{Ptr{Cint}}()
  ref_aux_type_size = Ref{MX_uint}()
  ref_aux_type_data = Ref{Ptr{Cint}}()
  ref_complete      = Ref{Cint}()

  @mxcall(:MXSymbolInferType,
          (MX_handle, MX_uint, char_pp, Ptr{Cint},
           Ref{MX_uint}, Ref{Ptr{Cint}},
           Ref{MX_uint}, Ref{Ptr{Cint}},
           Ref{MX_uint}, Ref{Ptr{Cint}},
           Ref{Cint}),
          self, length(arg_type_data)-1, keys, arg_type_data,
          ref_in_type_size, ref_in_type_data,
          ref_out_type_size, ref_out_type_data,
          ref_aux_type_size, ref_aux_type_data,
          ref_complete)

  if ref_complete[] == 0
    return (nothing, nothing, nothing)
  else
    in_type = unsafe_wrap(Array, ref_in_type_data[], ref_in_type_size[])
    out_type = unsafe_wrap(Array, ref_out_type_data[], ref_out_type_size[])
    aux_type = unsafe_wrap(Array, ref_aux_type_data[], ref_aux_type_size[])
    return ([fromTypeFlag(TypeFlag(t)) for t in in_type],
            [fromTypeFlag(TypeFlag(t)) for t in out_type],
            [fromTypeFlag(TypeFlag(t)) for t in aux_type])
  end
end

"""
    infer_type(self :: SymbolicNode; kwargs...)
    infer_type(self :: SymbolicNode, args...)

Do type inference according to the input types. The input types could be provided
as a list of types, which should specify the types of inputs in the same order as
the arguments returned by [`list_arguments`](@ref). Alternatively, the type information
could be specified via keyword arguments.

Returns a 3-tuple containing types of all the arguments, types of all the outputs and
types of all the auxiliary variables. If type inference failed due to incomplete
or incompatible inputs, the return value will be `(nothing, nothing, nothing)`.
"""
function infer_type(self :: SymbolicNode; kwargs...)
  types = Cint[toTypeFlag(x[2]) for x in kwargs]
  keys = AbstractString[string(x[1]) for x in kwargs]
  _infer_type(self, keys, types)
end

function infer_type(self :: SymbolicNode, args :: Union{Tuple, Void}...)
  types = Cint[]
  keys = Ptr{char_p}(0)

  for arg in args
    if isa(arg, Void); continue; end
    push!(types, toTypeFlag(arg))
  end
  _infer_type(self, keys, types)
end

"""
    getindex(self :: SymbolicNode, idx :: Union{Int, Base.Symbol, AbstractString})

Get a node representing the specified output of this node. The index could be
a symbol or string indicating the name of the output, or a 1-based integer
indicating the index, as in the list of [`list_outputs`](@ref).
"""
function Base.getindex(self :: SymbolicNode, idx :: Union{Base.Symbol, AbstractString})
  idx   = Symbol(idx)
  i_idx = find(idx .== list_outputs(self))
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

broadcast_(::typeof(+), x::SymbolicNode, ys::SymbolicNodeOrReal...) = +(x, ys...)
broadcast_(::typeof(+), s::Real, x::SymbolicNode, ys::SymbolicNodeOrReal...) = +(x + s, ys...)

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

broadcast_(::typeof(-), x::SymbolicNode, y::SymbolicNodeOrReal) = x - y
broadcast_(::typeof(-), s::Real, x::SymbolicNode) = s - x

import Base: *

"""
    .*(x, y)

Elementwise multiplication of `SymbolicNode`.
"""
x::SymbolicNode * s::Real = _mul_scalar(x, scalar=MX_float(s))
s::Real * x::SymbolicNode = _mul_scalar(x, scalar=MX_float(s))

function broadcast_(::typeof(*), x::SymbolicNode, ys::SymbolicNodeOrReal...)
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

broadcast_(::typeof(*), s::Real, x::SymbolicNode, ys::SymbolicNodeOrReal...) =
  broadcast_(*, x * s, ys...)

import Base: /

"""
    ./(x, y)

* Elementwise dividing a `SymbolicNode` by a scalar or another `SymbolicNode`
of the same shape.

* Elementwise divide a scalar by an `SymbolicNode`.

* Matrix division (solving linear systems) is not implemented yet.
"""
x::SymbolicNode / s::Real = _DivScalar(x, scalar=MX_float(s))

broadcast_(::typeof(/), x::SymbolicNode, y::SymbolicNode) = _div(x, y)
broadcast_(::typeof(/), x::SymbolicNode, s::Real) = _div_scalar(x,  scalar=MX_float(s))
broadcast_(::typeof(/), s::Real, x::SymbolicNode) = _rdiv_scalar(x, scalar=MX_float(s))


import Base: ^

"""
    .^(x, y)

Elementwise power of `SymbolicNode` and `NDArray`.
Operating with `Real` is available.
"""
^

broadcast_(::typeof(^), x::SymbolicNode, y::SymbolicNode) = _power(x, y)
broadcast_(::typeof(^), x::SymbolicNode, s::Real) = _power_scalar(x,  scalar=MX_float(s))
broadcast_(::typeof(^), s::Real, x::SymbolicNode) = _rpower_scalar(x, scalar=MX_float(s))

broadcast_(::typeof(^), ::Irrational{:e}, x::SymbolicNode) = exp(x)
broadcast_(::typeof(^), x::SymbolicNode, s::Irrational) =
  _power_scalar(x, scalar=MX_float(s))
broadcast_(::typeof(^), s::Irrational, x::SymbolicNode) =
  _rpower_scalar(x, scalar=MX_float(s))

function _compose!(node :: SymbolicNode; kwargs...)
  name     = char_p(0)
  arg_keys = AbstractString[]
  arg_vals = MX_handle[]

  for (k,v) in kwargs
    if k == :name
      name = string(v)
    else
      @assert(isa(v, SymbolicNode), "Compose expect `SymbolicNode` as arguments")
      push!(arg_keys, string(k))
      push!(arg_vals, v)
    end
  end

  @mxcall(:MXSymbolCompose,
          (MX_handle, char_p, MX_uint, Ptr{char_p}, Ptr{MX_handle}),
          node, name, length(arg_keys), arg_keys, arg_vals)
  return node
end
function _compose!(node :: SymbolicNode, args::SymbolicNode...)
  _compose!(node, char_p(0), args...)
end
function _compose!(node :: SymbolicNode, name :: Union{Base.Symbol, char_p}, args::SymbolicNode...)
  if isa(name, Base.Symbol); name = string(name); end
  arg_keys = Ptr{char_p}(0)
  arg_vals = MX_handle[args...]

  @mxcall(:MXSymbolCompose,
          (MX_handle, char_p, MX_uint, Ptr{char_p}, Ptr{MX_handle}),
          node, name, length(arg_vals), arg_keys, arg_vals)
  return node
end

"""
    to_json(self :: SymbolicNode)

Convert a `SymbolicNode` into a JSON string.
"""
function to_json(self :: SymbolicNode)
  ref_json = Ref{char_p}(0)
  @mxcall(:MXSymbolSaveToJSON, (MX_handle, Ref{char_p}), self, ref_json)
  return unsafe_string(ref_json[])
end

"""
    from_json(repr :: AbstractString, ::Type{SymbolicNode})

Load a `SymbolicNode` from a JSON string representation.
"""
function from_json(repr :: AbstractString, ::Type{SymbolicNode})
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateFromJSON, (char_p, Ref{MX_handle}), repr, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

"""
    load(filename :: AbstractString, ::Type{SymbolicNode})

Load a `SymbolicNode` from a JSON file.
"""
function load(filename :: AbstractString, ::Type{SymbolicNode})
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateFromFile, (char_p, Ref{MX_handle}), filename, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

"""
    save(filename :: AbstractString, node :: SymbolicNode)

Save a `SymbolicNode` to a JSON file.
"""
function save(filename :: AbstractString, node :: SymbolicNode)
  @mxcall(:MXSymbolSaveToFile, (MX_handle, char_p), node, filename)
end

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

@inline function _reshape(sym::SymbolicNode, dim::NTuple{N, Integer};
                          reverse::Bool=false, name::String="") where N
  op = _get_cached_libmx_op_handle("reshape")
  node = _create_atomic_symbol(op.value, ["shape", "reverse"],
                               [dump_mx_param(dim), dump_mx_param(!reverse)])
  name = get!(DEFAULT_NAME_MANAGER, name, "reshape")
  _compose!(node, name=name, data=sym)
end

################################################################################
# Atomic SymbolicNode functions dynamically imported from libmxnet
################################################################################
@inline function _create_atomic_symbol(creator::MX_handle, keys::Vector{String},
                                       vals::Vector{String})
  ref_sym_hdr = Ref{MX_handle}(C_NULL)
  @mxcall(:MXSymbolCreateAtomicSymbol,
          (MX_handle, MX_uint, Ptr{char_p}, Ptr{char_p}, Ref{MX_handle}),
          creator, length(keys), keys, vals, ref_sym_hdr)
  SymbolicNode(MX_SymbolHandle(ref_sym_hdr[]))
end

@inline function _create_atomic_symbol(creator::MX_handle, keys::Vector{String},
                                       vals::Vector{String},
                                       attrs::Dict{Symbol, String})
  node = _create_atomic_symbol(creator, keys, vals)
  # set attrs
  for (k, v) in attrs
    set_attr(node, k, v)
  end
  node
end

function _define_atomic_symbol_creator(name :: String)
  handle = _get_libmx_op_handle(name)
  f_desc, key_narg = _get_libmx_op_description(name, handle)

  f_desc *= "* `name::Symbol`: The name of the `SymbolicNode`. (e.g. `:my_symbol`), optional.\n"
  f_desc *= "* `attrs::Dict{Symbol, AbstractString}`: The attributes associated with this `SymbolicNode`.\n\n"

  func_name = Symbol(name)
  func_def = quote
  function $func_name(::Type{SymbolicNode}, args::SymbolicNode...; kwargs...)
    idx = findfirst(x -> x[1] == :name, kwargs)
    if idx > 0
      name = kwargs[idx][2]
    else
      name = ""
    end

    # XXX: hacky way of solving the problem that the arguments of `dot` should be swapped
    # See https://github.com/dmlc/MXNet.jl/issues/55
    if $name == "dot"
      args = reverse(args)
    end

    # XXX: hacky way of solving the semantic difference of the axes parameter in Julia
    # and in libmxnet.
    # See https://github.com/dmlc/MXNet.jl/pull/123
    if $name == "transpose"
      kwargs = Any[key != :axes ? (key, arg) : (key, reverse(map(i->length(arg)-i, arg))) for (key, arg) in kwargs]
    end

    param_keys = String[]
    param_vals = String[]
    symbol_kws = Dict{Symbol, SymbolicNode}()
    attrs = Dict{Symbol, String}()

    $(if key_narg != ""
      quote
        if !in($key_narg, param_keys)
          push!(param_keys, $key_narg)
          push!(param_vals, string(length(args)))
        end
      end
    end)

    for (k,v) in kwargs
      if k == :name; continue; end
      if isa(v, SymbolicNode)
        symbol_kws[k] = v
      elseif k == :attrs
        if isa(v, Dict)
          attrs = convert(Dict{Symbol, String}, v)
        else
          throw(ArgumentError("attrs needs to be a Dictionary"))
        end
      else
        push!(param_keys, string(k))
        push!(param_vals, dump_mx_param(v))
      end
    end

    if length(args) > 1 && length(symbol_kws) != 0
      @assert(false, $name * " only accepts SymbolicNode either as positional or keyword arguments with optional positional `data` argument, not both.")
    end
    $(if key_narg != ""
      quote
        if length(symbol_kws) > 0
          @assert(false, $name * " takes variable number of SymbolicNode arguments, " *
                         "please pass input Symbols via positional arguments, instead of keyword arguments.")
        end
      end
    end)

    local op = _get_cached_libmx_op_handle($name)
    node = _create_atomic_symbol(op.value, param_keys, param_vals, attrs)

    # generate a new name for the new symbol if user not provided in kwargs
    hint = lowercase($name)
    name = get!(DEFAULT_NAME_MANAGER, name, hint)

    if length(symbol_kws) == 0
      _compose!(node, name, args...)
    elseif length(args) == 1
      _compose!(node; name=name, data=args[1], symbol_kws...)
    else
      _compose!(node; name=name, symbol_kws...)
    end

    return node
  end # function
  end # quote

  func_def2 = quote
  @doc $f_desc ->
  function $func_name(args::SymbolicNode...; kwargs...)
    $func_name(SymbolicNode, args...; kwargs...)
  end # function
  end # quote

  return quote
    $func_def
    $func_def2
  end
end

macro _import_atomic_symbol_creators()
  # XXX: those are operators defined for NDArray, we exclude them here
  # because the calling convention for the type signature is not strong
  # enough to disambiguate the method for NDArray and SymbolicNode
  const ignored_ops = ["_set_value", "reshape"]  # in lowercase

  op_names = _get_libmx_op_names()
  func_exprs = map(op_names) do name
    if lowercase(name) ∉ ignored_ops
      expr = _define_atomic_symbol_creator(name)
    end
  end

  esc(quote
    $(func_exprs...)
  end)
end

@_import_atomic_symbol_creators()

################################################################################
# Utility macros to chain up symbols
################################################################################
macro chain(layers)
    exprs = []
    last_layer = nothing

    function _chain_layer(layer, last_layer)
        if isa(last_layer, Void)
            return esc(layer)
        else
            if @capture(layer, f_(x__))
                x′ = esc.(x)
                return :($f($last_layer, $(x′...)))
            else
                throw(AssertionError("$layer is not a valid function call and cannot be chained."))
            end
        end
    end

    while true
        if @capture(layers, l1_=>l2_)
            new_layer = gensym()
            push!(exprs, :($new_layer = $(_chain_layer(l1, last_layer))))
            last_layer = new_layer
            layers = l2
        else
            push!(exprs, _chain_layer(layers, last_layer))
            break
        end
    end
    Expr(:block, exprs...)
end
