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
# Atomic Symbol functions dynamically exported from libmxnet
################################################################################
