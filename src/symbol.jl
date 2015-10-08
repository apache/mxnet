export Symbol

################################################################################
# Symbol Type
################################################################################
type Symbol
  handle :: MX_SymbolHandle
end

function variable(name :: Union{Base.Symbol, AbstractString})
  hdr_ref = Ref{MX_handle}
  @mxcall(:MXSymbolCreateVariable, (char_p, Ref{MX_handle}), name, hdr_ref)
  Symbol(MX_SymbolHandle(hdr_ref[]))
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
  info("defining $func_name, kv_nargs = ($kv_nargs)")

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

    if $kv_nargs != symbol("") && !in($kv_nargs, param_keys)
      push!(param_keys, string($kv_nargs))
      push!(param_vals, string(length(args)))
    end

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
    if $kv_nargs != symbol("") && length(symbol_kws)
      @assert(false, "$func_name takes variable number of Symbol arguments, please pass input Symbols " *
                     "via positional arguments, instead of keyword arguments.")
    end

    # create the symbol
    ref_sym_hdr = Ref{MX_handle}()
    @mxcall(:MXSymbolCreateAtomicSymbol,
            (MX_handle, MX_unit, Ptr{char_p}, Ptr{char_p}, Ref{MX_handle}),
            hdr, length(param_keys), param_keys, param_vals, ref_sym_hdr)
    sym_hdr = ref_sym_hdr[]

    sym = Symbol(MX_SymbolHandle(sym_hdr))
    hint = lowercase(string($func_name))
    name = get!(DEFAULT_NAME_MANAGER, name, hint)

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
