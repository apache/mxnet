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

# compute graph related operators

################################################################################
# SymbolicNode attribute getter and setter
################################################################################

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
    list_arguments(s::SymbolicNode)

List all the arguments of this node. The argument for a node contains both
the inputs and parameters. For example, a `FullyConnected` node will
have both data and weights in its arguments. A composed node (e.g. a MLP) will
list all the arguments for intermediate nodes.

Returns a list of symbols indicating the names of the arguments.
"""
list_arguments(s::SymbolicNode) = @_list_symbol_info(s, :MXSymbolListArguments)

"""
    list_outputs(s::SymbolicNode)

List all the outputs of this node.

Returns a list of symbols indicating the names of the outputs.
"""
list_outputs(s::SymbolicNode) = @_list_symbol_info(s, :MXSymbolListOutputs)


"""
    list_auxiliary_states(s::SymbolicNode)


List all auxiliary states in the symbool.

Auxiliary states are special states of symbols that do not corresponds to an argument,
and do not have gradient. But still be useful for the specific operations.
A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
Most operators do not have Auxiliary states.

Returns a list of symbols indicating the names of the auxiliary states.
"""
list_auxiliary_states(s::SymbolicNode) =
  @_list_symbol_info(s, :MXSymbolListAuxiliaryStates)

"""
    get_internals(s::SymbolicNode)

Get a new grouped `SymbolicNode` whose output contains all the internal outputs of
this `SymbolicNode`.
"""
function get_internals(s::SymbolicNode)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolGetInternals, (MX_handle, Ref{MX_handle}), s, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

"""
    get_children(x::SymbolicNode)

Gets a new grouped `SymbolicNode` whose output contains inputs to output
nodes of the original symbol.

```julia
julia> x, y = @mx.var x y
(SymbolicNode x, SymbolicNode y)

julia> z = x + y
SymbolicNode _plus0

julia> z |> mx.get_children |> mx.list_outputs
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
    get_attr(s::SymbolicNode, key::Symbol)

Get attribute attached to this `SymbolicNode` belonging to key.

Returns the value belonging to key as a `String`.
If not available, returns `missing`.
"""
function get_attr(s::SymbolicNode, key::Symbol)
  key_s = string(key)
  ref_out = Ref{Cstring}()
  ref_success = Ref{Cint}(-1)
  @mxcall(:MXSymbolGetAttr, (MX_handle, Cstring, Ref{Cstring}, Ref{Cint}),
          s, key_s, ref_out, ref_success)
  if ref_success[] == 1
    unsafe_string(ref_out[])
  else
    missing
  end
end

"""
    list_attr(s::SymbolicNode)

Get all attributes from a symbol.

Returns a dictionary of attributes.
"""
function list_attr(s::SymbolicNode)
  ref_sz    = Ref{MX_uint}(0)
  ref_strings = Ref{char_pp}(0)
  @mxcall(:MXSymbolListAttrShallow, (MX_handle, Ref{MX_uint}, Ref{char_pp}),
            s, ref_sz, ref_strings)
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
    list_all_attr(s::SymbolicNode)

Get all attributes from the symbol graph.

Returns a dictionary of attributes.
"""
function list_all_attr(s::SymbolicNode)
  ref_sz    = Ref{MX_uint}(0)
  ref_strings = Ref{char_pp}(0)
  @mxcall(:MXSymbolListAttr, (MX_handle, Ref{MX_uint}, Ref{char_pp}),
            s, ref_sz, ref_strings)
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
    set_attr(s::SymbolicNode, key::Symbol, value::AbstractString)

Set the attribute key to value for this `SymbolicNode`.

!!! note
    It is encouraged not to call this function directly, unless you know exactly what you are doing. The
    recommended way of setting attributes is when creating the `SymbolicNode`. Changing
    the attributes of a `SymbolicNode` that is already been used somewhere else might
    cause unexpected behavior and inconsistency.
"""
function set_attr(s::SymbolicNode, key::Symbol, value::AbstractString)
  key_s = string(key)
  value_s = String(value)

  @mxcall(:MXSymbolSetAttr, (MX_handle, Cstring, Cstring), s, key_s, value_s)
end

"""
    get_name(s::SymbolicNode)

Get the name of the symbol.

    julia> x = mx.Variable(:data)
    julia> mx.get_name(x)
    :data

    julia> y = mx.FullyConnected(x, num_hidden = 128)
    julia> mx.get_name(y)
    :fullyconnected0
"""
function get_name(s::mx.SymbolicNode)
    name = Ref{mx.char_p}(C_NULL)
    success = Ref(0)
    @mxcall(:MXSymbolGetName, (MX_handle, Ref{char_p}, Ref{Int}), s.handle.value, name, success)
    @assert success[] != -1

    str = name[]
    if str == C_NULL  # e.g. the symbol returned via get_internals
        string(s.handle.value)
    else
        Symbol(unsafe_string(str))
    end
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

function _define_atomic_symbol_creator(name::String)
  handle = _get_libmx_op_handle(name)
  f_desc, key_narg = _get_libmx_op_description(name, handle)

  f_desc *= "* `name::Symbol`: The name of the `SymbolicNode`. (e.g. `:my_symbol`), optional.\n"
  f_desc *= "* `attrs::Dict{Symbol,String}`: The attributes associated with this `SymbolicNode`.\n\n"

  func_name = Symbol(name)
  import_expr = _import_expr(func_name)

  func_def = quote
  function $func_name(::Type{SymbolicNode}, args::SymbolicNode...; name = "", kwargs...)

    # NOTE: hacky way of solving the problem that the arguments of `dot` should be swapped
    # See https://github.com/dmlc/MXNet.jl/issues/55
    if $name == "dot"
      args = reverse(args)
    end

    # NOTE: hacky way of solving the semantic difference of the axes parameter in Julia
    # and in libmxnet.
    # See https://github.com/dmlc/MXNet.jl/pull/123
    if $name == "transpose"
      kwargs = Any[key != :axes ? (key, arg) : (key, reverse(map(i->length(arg)-i, arg))) for (key, arg) in kwargs]
    end

    param_keys = String[]
    param_vals = String[]
    symbol_kws = Dict{Symbol,SymbolicNode}()
    attrs = Dict{Symbol,String}()

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
  @doc $f_desc
  function $func_name(args::SymbolicNode...; kwargs...)
    $func_name(SymbolicNode, args...; kwargs...)
  end # function
  end # quote

  return quote
    $import_expr
    $func_def
    $func_def2
  end
end

macro _import_atomic_symbol_creators()
  # NOTE: those are operators defined for NDArray, we exclude them here
  # because the calling convention for the type signature is not strong
  # enough to disambiguate the method for NDArray and SymbolicNode
  ignored_ops = ("_set_value", "reshape")  # in lowercase

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

@_import_atomic_symbol_creators

################################################################################
# Utility macros to chain up symbols
################################################################################

macro chain(layers)
    exprs = []
    last_layer = nothing

    function _chain_layer(layer, last_layer)
        if last_layer ≡ nothing
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

################################################################################
# compose
################################################################################

function _compose!(node::SymbolicNode; kwargs...)
  name     = char_p(C_NULL)
  arg_keys = AbstractString[]  # FIXME: can it be String[] ?
  arg_vals = MX_handle[]

  for (k, v) in kwargs
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
  node
end

_compose!(node::SymbolicNode, args::SymbolicNode...) =
  _compose!(node, char_p(0), args...)

function _compose!(node::SymbolicNode, name::Union{Symbol, char_p}, args::SymbolicNode...)
  if name isa Symbol
    name = string(name)
  end
  arg_keys = Ptr{char_p}(C_NULL)
  arg_vals = MX_handle[args...]

  @mxcall(:MXSymbolCompose,
          (MX_handle, char_p, MX_uint, Ptr{char_p}, Ptr{MX_handle}),
          node, name, length(arg_vals), arg_keys, arg_vals)
  node
end
