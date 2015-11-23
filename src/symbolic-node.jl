#=doc
Symbolic API
============
=#

#=doc
.. class:: SymbolicNode

   SymbolicNode is the basic building block of the symbolic graph in MXNet.jl.
=#
type SymbolicNode
  handle :: MX_SymbolHandle
end
function Base.unsafe_convert(::Type{MX_handle}, obj::SymbolicNode)
  Base.unsafe_convert(MX_handle, obj.handle)
end
Base.convert(t::Type{MX_handle}, obj::SymbolicNode) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::SymbolicNode) = Base.unsafe_convert(t, obj)

#=doc
.. function:: deepcopy(self :: SymbolicNode)

   Make a deep copy of a SymbolicNode.
=#
function Base.deepcopy(self :: SymbolicNode)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCopy, (MX_handle, Ref{MX_handle}), self, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

#=doc
.. function:: copy(self :: SymbolicNode)

   Make a copy of a SymbolicNode. The same as making a deep copy.
=#
function Base.copy(self :: SymbolicNode)
  Base.deepcopy(self)
end

#=doc
.. function::
   call(self :: SymbolicNode, args :: SymbolicNode...)
   call(self :: SymbolicNode; kwargs...)

   Make a new node by composing ``self`` with ``args``. Or the arguments
   can be specified using keyword arguments.
=#
function Base.call(self :: SymbolicNode, args :: SymbolicNode...)
  s = deepcopy(self)
  _compose!(s, args...)
end
function Base.call(self :: SymbolicNode; kwargs...)
  s = deepcopy(self)
  _compose!(s; kwargs...)
end

macro _list_symbol_info(self, func_name)
  quote
    ref_sz    = Ref{MX_uint}(0)
    ref_names = Ref{char_pp}(0)
    @mxcall($func_name, (MX_handle, Ref{MX_uint}, Ref{char_pp}),
            $self, ref_sz, ref_names)
    narg = ref_sz[]
    names = pointer_to_array(ref_names[], narg)
    names = [symbol(bytestring(x)) for x in names]
    return names
  end
end

#=doc
.. function:: list_arguments(self :: SymbolicNode)

   List all the arguments of this node. The argument for a node contains both
   the inputs and parameters. For example, a :class:`FullyConnected` node will
   have both data and weights in its arguments. A composed node (e.g. a MLP) will
   list all the arguments for intermediate nodes.

   :return: A list of symbols indicating the names of the arguments.
=#
function list_arguments(self :: SymbolicNode)
  @_list_symbol_info(self, :MXSymbolListArguments)
end

#=doc
.. function:: list_outputs(self :: SymbolicNode)

   List all the outputs of this node.

   :return: A list of symbols indicating the names of the outputs.
=#
function list_outputs(self :: SymbolicNode)
  @_list_symbol_info(self, :MXSymbolListOutputs)
end


#=doc
.. function:: list_auxiliary_states(self :: SymbolicNode)


   List all auxiliary states in the symbool.

   Auxiliary states are special states of symbols that do not corresponds to an argument,
   and do not have gradient. But still be useful for the specific operations.
   A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
   Most operators do not have Auxiliary states.

   :return: A list of symbols indicating the names of the auxiliary states.
=#
function list_auxiliary_states(self :: SymbolicNode)
  @_list_symbol_info(self, :MXSymbolListAuxiliaryStates)
end

#=doc
.. function:: get_internals(self :: SymbolicNode)

   Get a new grouped :class:`SymbolicNode` whose output contains all the internal outputs of
   this :class:`SymbolicNode`.
=#
function get_internals(self :: SymbolicNode)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolGetInternals, (MX_handle, Ref{MX_handle}), self, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

#=doc
.. function:: get_attr(self :: SymbolicNode, key :: Symbol)

   Get attribute attached to this :class:`SymbolicNode` belonging to key.
=#
function get_attr(self :: SymbolicNode, key :: Symbol)
  key_s = bytestring(string(key))
  ref_out = Ref{Cstring}()
  ref_success = Ref{Cint}(-1)
  @mxcall(:MXSymbolGetAttr, (MX_handle, Cstring, Ref{Cstring}, Ref{Cint}), self, key_s, ref_out, ref_success)
  if ref_success[] == 1
    return bytestring(ref_out[])
  else
    throw(KeyError(key))
  end
end

#=doc
.. function:: set_attr(self:: SymbolicNode, key :: Symbol, value :: AbstractString)

   Set the attribute key to value for this :class:`SymbolicNode`.
=#
function set_attr(self :: SymbolicNode, key :: Symbol, value :: AbstractString)
  key_s = bytestring(string(key))
  value_s = bytestring(value)

  @mxcall(:MXSymbolSetAttr, (MX_handle, Cstring, Cstring), self, key_s, value_s)
end

#=doc
.. function:: Variable(name :: Union{Base.Symbol, AbstractString})

   Create a symbolic variable with the given name. This is typically used as a placeholder.
   For example, the data node, acting as the starting point of a network architecture.
=#
function Variable(name :: Union{Base.Symbol, AbstractString})
  hdr_ref = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateVariable, (char_p, Ref{MX_handle}), name, hdr_ref)
  SymbolicNode(MX_SymbolHandle(hdr_ref[]))
end

#=doc
.. function:: Group(nodes :: SymbolicNode...)

   Create a :class:`SymbolicNode` by grouping nodes together.
=#
function Group(nodes :: SymbolicNode...)
  handles = MX_handle[nodes...]
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateGroup, (MX_uint, Ptr{MX_handle}, Ref{MX_handle}),
          length(handles), handles, ref_hdr)
  SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

macro _infer_shape(self, keys, indptr, sdata)
  quote
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
      function build_shapes(shape_size::MX_uint, shape_ndim::Ptr{MX_uint}, shape_data::Ptr{Ptr{MX_uint}})
        shape_ndim = pointer_to_array(shape_ndim, shape_size)
        shape_data = pointer_to_array(shape_data, shape_size)
        shapes = map(1:shape_size) do i
          my_shape = pointer_to_array(shape_data[i], shape_ndim[i])
          tuple(flipdim(Int[my_shape...],1)...)
        end
        convert(Vector{Tuple}, shapes)
      end
      return (
        build_shapes(ref_arg_shape_size[], ref_arg_shape_ndim[], ref_arg_shape_data[]),
        build_shapes(ref_out_shape_size[], ref_out_shape_ndim[], ref_out_shape_data[]),
        build_shapes(ref_aux_shape_size[], ref_aux_shape_ndim[], ref_aux_shape_data[])
      )
    end
  end
end

#=doc
.. function::
   infer_shape(self :: SymbolicNode; args...)
   infer_shape(self :: SymbolicNode; kwargs...)

   Do shape inference according to the input shapes. The input shapes could be provided
   as a list of shapes, which should specify the shapes of inputs in the same order as
   the arguments returned by :func:`list_arguments`. Alternatively, the shape information
   could be specified via keyword arguments.

   :return: A 3-tuple containing shapes of all the arguments, shapes of all the outputs and
            shapes of all the auxiliary variables. If shape inference failed due to incomplete
            or incompatible inputs, the return value will be ``(nothing, nothing, nothing)``.
=#
function infer_shape(self :: SymbolicNode; kwargs...)
  sdata  = MX_uint[]
  indptr = MX_uint[0]
  for (k,v) in kwargs
    append!(sdata, flipdim([v...],1))
    push!(indptr, length(sdata))
  end
  keys = AbstractString[string(x[1]) for x in kwargs]
  @_infer_shape(self, keys, indptr, sdata)
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
  @_infer_shape(self, keys, indptr, sdata)
end

#=doc
.. function::
   getindex(self :: SymbolicNode, idx :: Union{Int, Base.Symbol, AbstractString})

   Get a node representing the specified output of this node. The index could be
   a symbol or string indicating the name of the output, or a 1-based integer
   indicating the index, as in the list of :func:`list_outputs`.
=#
function Base.getindex(self :: SymbolicNode, idx :: Union{Base.Symbol, AbstractString})
  idx   = symbol(idx)
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
  _MinusScalar(self, scalar=arg, scalar_on_left=true)
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
function ./(arg :: Real, self :: SymbolicNode)
  _DivScalar(self, scalar=arg, scalar_on_left=true)
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

#=doc
.. function:: to_json(self :: SymbolicNode)

   Convert a :class:`SymbolicNode` into a JSON string.
=#
function to_json(self :: SymbolicNode)
  ref_json = Ref{char_p}(0)
  @mxcall(:MXSymbolSaveToJSON, (MX_handle, Ref{char_p}), self, ref_json)
  return bytestring(ref_json[])
end

#=doc
.. function:: from_json(repr :: AbstractString, ::Type{SymbolicNode})

   Load a :class:`SymbolicNode` from a JSON string representation.
=#
function from_json(repr :: AbstractString, ::Type{SymbolicNode})
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateFromJSON, (char_p, Ref{MX_handle}), repr, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

#=doc
.. function:: load(filename :: AbstractString, ::Type{SymbolicNode})

   Load a :class:`SymbolicNode` from a JSON file.
=#
function load(filename :: AbstractString, ::Type{SymbolicNode})
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateFromFile, (char_p, Ref{MX_handle}), filename, ref_hdr)
  return SymbolicNode(MX_SymbolHandle(ref_hdr[]))
end

#=doc
.. function:: save(filename :: AbstractString, node :: SymbolicNode)

   Save a :class:`SymbolicNode` to a JSON file.
=#
function save(filename :: AbstractString, node :: SymbolicNode)
  @mxcall(:MXSymbolSaveToFile, (MX_handle, char_p), node, filename)
end

#=doc
libmxnet APIs
-------------

**autogen:EMBED:symbolic-node:EMBED:autogen**
=#
################################################################################
# Atomic SymbolicNode functions dynamically imported from libmxnet
################################################################################
function _define_atomic_symbol_creator(hdr :: MX_handle; gen_docs=false)
  ref_name      = Ref{char_p}(0)
  ref_desc      = Ref{char_p}(0)
  ref_kv_nargs  = Ref{char_p}(0)
  ref_nargs     = Ref{MX_uint}(0)
  ref_arg_names = Ref{char_pp}(0)
  ref_arg_types = Ref{char_pp}(0)
  ref_arg_descs = Ref{char_pp}(0)

  @mxcall(:MXSymbolGetAtomicSymbolInfo,
          (MX_handle, Ref{char_p}, Ref{char_p}, Ref{MX_uint}, Ref{char_pp}, Ref{char_pp},
           Ref{char_pp}, Ref{char_p}),
          hdr, ref_name, ref_desc, ref_nargs, ref_arg_names, ref_arg_types, ref_arg_descs, ref_kv_nargs)

  func_name_s= bytestring(ref_name[])
  func_name  = symbol(func_name_s)
  kv_nargs_s = bytestring(ref_kv_nargs[])
  kv_nargs   = symbol(kv_nargs_s)

  if gen_docs
    f_desc = bytestring(ref_desc[]) * "\n\n"
    if !isempty(kv_nargs_s)
      f_desc *= "This function support variable length positional :class:`SymbolicNode` inputs.\n\n"
    end
    f_desc *= _format_docstring(Int(ref_nargs[]), ref_arg_names, ref_arg_types, ref_arg_descs)
    f_desc *= ":param Base.Symbol name: The name of the :class:`SymbolicNode`. (e.g. `:my_symbol`), optional.\n\n"
    f_desc *= ":return: the constructed :class:`SymbolicNode`.\n\n"
    return (func_name, f_desc)
  end

  # function $func_name(args...; kwargs...)
  func_head = Expr(:call, func_name, Expr(:parameters, Expr(:..., :kwargs)), Expr(:..., :args))
  func_body = quote
    idx = findfirst(x -> x[1] == :name, kwargs)
    if idx > 0
      name = kwargs[idx][2]
    else
      name = ""
    end

    param_keys = AbstractString[]
    param_vals = AbstractString[]
    symbol_kws = Dict{Base.Symbol, SymbolicNode}()

    $(if kv_nargs != symbol("")
      quote
        if !in($kv_nargs_s, param_keys)
          push!(param_keys, $kv_nargs_s)
          push!(param_vals, string(length(args)))
        end
      end
    end)

    for (k,v) in kwargs
      if k == :name; continue; end
      if isa(v, SymbolicNode)
        symbol_kws[k] = v
      else
        push!(param_keys, string(k))
        push!(param_vals, dump_mx_param(v))
      end
    end

    if length(args) != 0 && length(symbol_kws) != 0
      @assert(false, $func_name_s * " only accepts Symbols either as positional or keyword arguments, not both.")
    end
    $(if kv_nargs != symbol("")
      quote
        if length(symbol_kws) > 0
          @assert(false, $func_name * " takes variable number of SymbolicNode arguments, please pass input Symbols " *
                         "via positional arguments, instead of keyword arguments.")
        end
      end
    end)

    # create the SymbolicNode
    ref_sym_hdr = Ref{MX_handle}()
    @mxcall(:MXSymbolCreateAtomicSymbol,
            (MX_handle, MX_uint, Ptr{char_p}, Ptr{char_p}, Ref{MX_handle}),
            $hdr, length(param_keys), param_keys, param_vals, ref_sym_hdr)
    sym_hdr = ref_sym_hdr[]

    node = SymbolicNode(MX_SymbolHandle(sym_hdr))
    hint = lowercase($func_name_s)
    name = get!(DEFAULT_NAME_MANAGER, name, hint)

    if length(args) != 0
      _compose!(node, name, args...)
    else
      _compose!(node; name=name, symbol_kws...)
    end

    return node
  end

  func_def = Expr(:function, func_head, Expr(:block, func_body))
  eval(func_def)
end

function _import_atomic_symbol_creators(;gen_docs=false)
  n_ref = Ref{MX_uint}(0)
  h_ref = Ref{Ptr{MX_handle}}(0)
  @mxcall(:MXSymbolListAtomicSymbolCreators, (Ref{MX_uint}, Ref{Ptr{MX_handle}}), n_ref, h_ref)

  n_creators = n_ref[]
  h_creators = pointer_to_array(h_ref[], n_creators)

  if gen_docs
    docs = Dict{Base.Symbol, AbstractString}()
  end

  for i = 1:n_creators
    creator_hdr = h_creators[i]
    ret = _define_atomic_symbol_creator(creator_hdr, gen_docs=gen_docs)
    if gen_docs
      docs[ret[1]] = ret[2]
    end
  end

  if gen_docs
    return docs
  end
end

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
