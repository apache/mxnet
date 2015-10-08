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
