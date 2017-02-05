"""
    SymbolicNode

SymbolicNode is the basic building block of the symbolic graph in MXNet.jl.

    (self :: SymbolicNode)(args :: SymbolicNode...)
    (self :: SymbolicNode)(; kwargs...)

Make a new node by composing `self` with `args`. Or the arguments
can be specified using keyword arguments.
"""
type SymbolicNode
  handle :: MX_SymbolHandle
end
function Base.unsafe_convert(::Type{MX_handle}, obj::SymbolicNode)
  Base.unsafe_convert(MX_handle, obj.handle)
end
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

@compat function (self::SymbolicNode)(args :: SymbolicNode...)
  s = deepcopy(self)
  _compose!(s, args...)
end
@compat function (self::SymbolicNode)(;kwargs...)
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

import Base: +, .+
function +(self :: SymbolicNode, args :: Union{SymbolicNode,Real}...)
  ret = self
  for arg in args
    if isa(arg, SymbolicNode)
      ret = _Plus(ret, arg)
    else
      ret = _PlusScalar(ret, scalar=MX_float(arg))
    end
  end
  ret
end
function .+(self :: SymbolicNode, args :: Union{SymbolicNode,Real}...)
  +(self, args...)
end
function +(s1 :: Real, self :: SymbolicNode, args :: Union{SymbolicNode,Real}...)
  +(self, s1, args...)
end
function .+(s1 :: Real, self :: SymbolicNode, args :: Union{SymbolicNode,Real}...)
  +(self, s1, args...)
end

import Base: -, .-
function -(self :: SymbolicNode, arg :: SymbolicNode)
  _Minus(self, arg)
end
function .-(self :: SymbolicNode, arg :: SymbolicNode)
  -(self, arg)
end
function -(self :: SymbolicNode, arg :: Real)
  _MinusScalar(self, scalar=MX_float(arg))
end
function .-(self :: SymbolicNode, arg :: Real)
  -(self, arg)
end

function -(arg :: Real, self :: SymbolicNode)
  _RMinusScalar(self, scalar=arg)
end
function .-(arg :: Real, self :: SymbolicNode)
  -(arg, self)
end

function -(self :: SymbolicNode)
  -(0, self)
end

import Base: .*, *
function .*(self :: SymbolicNode, args :: Union{SymbolicNode,Real}...)
  ret = self
  for arg in args
    if isa(arg, SymbolicNode)
      ret = _Mul(ret, arg)
    else
      ret = _MulScalar(ret, scalar=MX_float(arg))
    end
  end
  ret
end
function .*(arg :: Real, self :: SymbolicNode, args :: Union{SymbolicNode,Real}...)
  .*(self, arg, args...)
end
function *(arg :: Real, self :: SymbolicNode)
  _MulScalar(self, scalar=arg)
end
function *(self :: SymbolicNode, arg :: Real)
  *(arg, self)
end

import Base: ./, /
function ./(self :: SymbolicNode, arg :: SymbolicNode)
  _Div(self, arg)
end
function ./(self :: SymbolicNode, arg :: Real)
  _DivScalar(self, scalar=MX_float(arg))
end
function /(self :: SymbolicNode, arg :: Real)
  ./(self, arg)
end
function /(arg :: Real, self :: SymbolicNode)
  _RDivScalar(self, scalar=arg)
end
function ./(arg :: Real, self :: SymbolicNode)
  _RDivScalar(self, scalar=arg)
end

import Base: .^, ^
function .^(self :: SymbolicNode, pow :: SymbolicNode)
  _Power(self, pow)
end
function .^(self :: SymbolicNode, pow :: AbstractFloat)
  _PowerScalar(self, scalar=pow)
end
function ^(self :: SymbolicNode, pow :: AbstractFloat)
  .^(self, pow)
end

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

################################################################################
# Atomic SymbolicNode functions dynamically imported from libmxnet
################################################################################
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
          attrs = convert(Dict{Symbol, AbstractString}, v)
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

    local hdr = _get_cached_libmx_op_handle($name)

    # create the SymbolicNode
    ref_sym_hdr = Ref{MX_handle}()
    @mxcall(:MXSymbolCreateAtomicSymbol,
            (MX_handle, MX_uint, Ptr{char_p}, Ptr{char_p}, Ref{MX_handle}),
            hdr, length(param_keys), param_keys, param_vals, ref_sym_hdr)
    sym_hdr = ref_sym_hdr[]

    node = SymbolicNode(MX_SymbolHandle(sym_hdr))
    hint = lowercase($name)
    name = get!(DEFAULT_NAME_MANAGER, name, hint)

    # set attrs
    for (k, v) in attrs
      set_attr(node, k, v)
    end

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
  const ignored_ops = ["_set_value"]

  op_names = _get_libmx_op_names()
  func_exprs = map(op_names) do name
    if name ∉ ignored_ops
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
      esc(layer)
    else
      @assert(isa(layer, Expr) && layer.head == :call, "Do not know how to chain up $layer")
      return Expr(:call, esc(layer.args[1]), last_layer, map(esc, layer.args[2:end])...)
    end
  end
  while true
    if layers.head == :(=>)
      new_layer = gensym()
      push!(exprs, :($new_layer = $(_chain_layer(layers.args[1], last_layer))))
      last_layer = new_layer
      layers = layers.args[2]
    else
      push!(exprs, _chain_layer(layers, last_layer))
      break
    end
  end
  return Expr(:block, exprs...)
end
