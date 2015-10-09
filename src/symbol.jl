export Symbol

################################################################################
# Symbol Type
################################################################################
type Symbol
  handle :: MX_SymbolHandle
end
function Base.unsafe_convert(::Type{MX_handle}, obj::Symbol)
  Base.unsafe_convert(MX_handle, obj.handle)
end
Base.convert(t::Type{MX_handle}, obj::Symbol) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::Symbol) = Base.unsafe_convert(t, obj)

function Base.deepcopy(self :: Symbol)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCopy, (MX_handle, Ref{MX_handle}), self, ref_hdr)
  return Symbol(MX_SymbolHandle(ref_hdr[]))
end
function Base.copy(self :: Symbol)
  Base.deepcopy(self)
end

function Base.call(self :: Symbol, args :: Symbol...)
  s = deepcopy(self)
  _compose!(s, args...)
end
function Base.call(self :: Symbol; kwargs...)
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
function list_arguments(self :: Symbol)
  @_list_symbol_info(self, :MXSymbolListArguments)
end
function list_outputs(self :: Symbol)
  @_list_symbol_info(self, :MXSymbolListOutputs)
end
"""List all auxiliary states in the symbool.

Auxiliary states are special states of symbols that do not corresponds to an argument,
and do not have gradient. But still be useful for the specific operations.
A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
Most operators do not have Auxiliary states.
"""
function list_auxiliary_states(self :: Symbol)
  @_list_symbol_info(self, :MXSymbolListAuxiliaryStates)
end

"Get a new grouped symbol whose output contains all the internal outputs of this symbol."
function get_internals(self :: Symbol)
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolGetInternals, (MX_handle, Ref{MX_handle}), self, ref_hdr)
  return Symbol(MX_SymbolHandle(ref_hdr[]))
end

"Create a symbolic variable with the given name"
function variable(name :: Union{Base.Symbol, AbstractString})
  hdr_ref = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateVariable, (char_p, Ref{MX_handle}), name, hdr_ref)
  Symbol(MX_SymbolHandle(hdr_ref[]))
end

"Create a symbol that groups symbols together"
function group(symbols :: Symbol...)
  handles = MX_handle[symbols...]
  ref_hdr = Ref{MX_handle}(0)
  @mxcall(:MXSymbolCreateGroup, (MX_uint, Ptr{MX_handle}, Ref{MX_handle}),
          length(handles), handles, ref_hdr)
  Symbol(MX_SymbolHandle(ref_hdr[]))
end

function Base.getindex(self :: Symbol, idx :: Union{Base.Symbol, AbstractString})
  idx   = symbol(idx)
  i_idx = find(idx .== list_outputs(self))
  @assert(length(i_idx) > 0, "Cannot find output with name '$idx'")
  @assert(length(i_idx) < 2, "Found duplicated output with name '$idx'")
  Base.getindex(self, i_idx[1])
end
function Base.getindex(self :: Symbol, idx :: Int)
  ref_hdr = Ref{MX_handle}(0)
  # note Julia is 1-based, while MXNet is 0-based
  @mxcall(:MXSymbolGetOutput, (MX_handle, MX_uint, Ref{MX_handle}), self, idx-1, ref_hdr)
  return Symbol(MX_SymbolHandle(ref_hdr[]))
end

"Compose symbol on inputs"
function _compose!(sym :: Symbol; kwargs...)
  name     = char_p(0)
  arg_keys = AbstractString[]
  arg_vals = MX_handle[]

  for (k,v) in kwargs
    if k == :name
      name = string(v)
    else
      @assert(isa(v, Symbol), "Compose expect `Symbol` as arguments")
      push!(arg_keys, string(k))
      push!(arg_vals, v)
    end
  end

  @mxcall(:MXSymbolCompose,
          (MX_handle, char_p, MX_uint, Ptr{char_p}, Ptr{MX_handle}),
          sym, name, length(arg_keys), arg_keys, arg_vals)
  return sym
end
function _compose!(sym :: Symbol, args::Symbol...)
  name     = char_p(0)
  arg_keys = Ptr{char_p}(0)
  arg_vals = MX_handle[args...]

  @mxcall(:MXSymbolCompose,
          (MX_handle, char_p, MX_uint, Ptr{char_p}, Ptr{MX_handle}),
          sym, name, length(arg_keys), arg_keys, arg_vals)
  return sym
end

################################################################################
# Atomic Symbol functions dynamically imported from libmxnet
################################################################################
function _define_atomic_symbol_creator(hdr :: MX_handle)
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

  func_name = symbol(bytestring(ref_name[]))
  kv_nargs  = symbol(bytestring(ref_kv_nargs[]))

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
    symbol_kws = Dict{Base.Symbol, Symbol}()

    $(if kv_nargs != symbol("")
      quote
        if !in("$kv_narg", param_keys)
          push!(param_keys, string("$kv_nargs"))
          push!(param_vals, string(length(args)))
        end
      end
    end)

    for (k,v) in kwargs
      if k == :name; continue; end
      if isa(v, Symbol)
        symbol_kws[k] = v
      else
        push!(param_keys, string(k))
        push!(param_vals, string(v))
      end
    end

    if length(args) != 0 && length(symbol_kws) != 0
      @assert(false, "$func_name only accepts Symbols either as positional or keyword arguments, not both.")
    end
    $(if kv_nargs != symbol("")
      quote
        if length(symbol_kws) > 0
          @assert(false, "$func_name takes variable number of Symbol arguments, please pass input Symbols " *
                         "via positional arguments, instead of keyword arguments.")
        end
      end
    end)

    # create the symbol
    ref_sym_hdr = Ref{MX_handle}()
    @mxcall(:MXSymbolCreateAtomicSymbol,
            (MX_handle, MX_uint, Ptr{char_p}, Ptr{char_p}, Ref{MX_handle}),
            $hdr, length(param_keys), param_keys, param_vals, ref_sym_hdr)
    sym_hdr = ref_sym_hdr[]

    sym = Symbol(MX_SymbolHandle(sym_hdr))
    hint = lowercase(string($func_name))
    name = get!(DEFAULT_NAME_MANAGER, name, hint)

    _compose!(sym; name=name, symbol_kws...)

    return sym
  end

  func_def = Expr(:function, func_head, Expr(:block, func_body))
  eval(func_def)

  # TODO: add doc string
  # eval(:(@doc($doc_str, $func_name)))
end

function _import_atomic_symbol_creators()
  n_ref = Ref{MX_uint}(0)
  h_ref = Ref{Ptr{MX_handle}}(0)
  @mxcall(:MXSymbolListAtomicSymbolCreators, (Ref{MX_uint}, Ref{Ptr{MX_handle}}), n_ref, h_ref)

  n_creators = n_ref[]
  h_creators = pointer_to_array(h_ref[], n_creators)

  for i = 1:n_creators
    creator_hdr = h_creators[i]
    _define_atomic_symbol_creator(creator_hdr)
  end
end
