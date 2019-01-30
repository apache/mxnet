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

# Mapping NDArray functions to Base-like API

const _ndsig = Dict{Symbol,Expr}()
const _nddoc = Dict{Symbol,Any}()

_isinplace(name::Symbol) = endswith(string(name), "!")

_writable(name::Symbol, x) =
  _isinplace(name) ? :(@assert $x.writable "this NDArray isn't writable") : :()

function _outexpr(name::Symbol, x #= the first arg of `sig` =#)
  if _isinplace(name)  # `func!`
    Ptr, 1, :([[MX_handle(x.handle)]]), :($x)
  else
    retexpr = :(NDArray(MX_NDArrayHandle(unsafe_load(hdls_ref[], 1))))
    Ref, 0, :(Ref{Ptr{MX_handle}}(C_NULL)), retexpr
  end
end

_broadcast_target(sig::Expr) = sig.args[2].args[].args[end]

"""
Generate docstring from function signature
"""
function _docsig(fname::Symbol, sig::Expr, opname::String)
  if fname !== :broadcasted
    get(_nddoc, fname, "    $sig") * "\n" * _getdocdefine(opname)
  else
    name = _broadcast_target(sig)
    str = get(_nddoc, name, "")
    _nddoc[name] = false  # change to false, denote docstring has been set up
    if isempty(str)
      sig_ = Expr(:call, Symbol(name, "."), sig.args[3:end]...)
      str = "    $sig_"
    end
    if str ≠ false
      # append "Defined in ..."
      def = _getdocdefine(opname)
      str = if str isa Markdown.MD
        str = Markdown.MD(copy(str.content), copy(str.meta))
        push!(str, Markdown.Paragraph(def))
        str
      else
        str * def
      end

      @eval @doc $str $name
    end
    ""
  end
end

"""
    @_remap(sig::Expr, imp::Expr)

Creating a function in signature `sig` with the function implementation `imp`.

## Arguments
- `sig` is the function signature.
  If the function name ends with `!`, it will invoke the corresponding inplace
  call.
- `imp` is the underlying libmxnet API call

"""
macro _remap(sig::Expr, imp::Expr)
  d = splitdef(:($sig = $imp))
  @capture d[:name] (M_.fname_|fname_)

  opname = string(imp.args[1])

  if isa(imp.args[2], Expr) && imp.args[2].head == :parameters
    ndin = imp.args[3:end]
    mxargs = imp.args[2].args
  else  # no keyword arguments
    ndin = imp.args[2:end]
    mxargs = []
  end

  mxkeys = map(x -> string(x.args[1]), mxargs)
  mxvals = Expr(:vect, map(x -> :(dump_mx_param($(x.args[2]))), mxargs)...)
  ndhlds = Expr(:vect, map(x -> :($(x).handle), ndin)...)

  # handler for `func!` which has side effect on first argument.
  T, n_output, hdls_ref, retexpr = _outexpr(fname, _firstarg(sig))

  assert_expr = _writable(fname, _firstarg(sig))

  func_body = quote
    $assert_expr
    op_handle = _get_cached_libmx_op_handle($opname)
    n_output = Ref(Cint($n_output))
    hdls_ref = $hdls_ref
    @mxcall(:MXImperativeInvoke,
            (MX_handle,
             Cint,
             Ptr{MX_handle},
             Ref{Cint},
             $T{Ptr{MX_handle}},
             Cint,
             char_pp,
             char_pp),
            op_handle,
            $(length(ndin)),
            $(ndhlds),
            n_output,
            hdls_ref,
            $(length(mxargs)),
            $mxkeys,
            $mxvals)
    $retexpr
  end

  docstr = _docsig(fname, sig, opname)
  func_def = Expr(:function, sig, func_body)

  esc(quote
    @doc $docstr
    $func_def
  end)
end

macro _remap(sig::Expr, imp::Symbol)
  imp = _ndsig[imp]

  esc(quote
    @_remap($sig, $imp)
  end)
end
