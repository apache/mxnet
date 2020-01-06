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

# NDArray functions dynamically imported from libmxnet

function _invoke_mxfunction(func_handle::MX_handle, use_vars, scalars, mut_vars; kwargs...)
  names = String[string(entry[1]) for entry in kwargs]
  args = String[string(entry[2]) for entry in kwargs]
  @mxcall(:MXFuncInvokeEx,
          (MX_handle, Ptr{MX_handle}, Ptr{MX_float}, Ptr{MX_handle}, Cint, char_pp, char_pp),
          func_handle, use_vars, scalars, mut_vars, length(names), names, args)
end

@enum(LIBMX_FUNC_TYPE_MASK,
  NDARRAY_ARG_BEFORE_SCALAR = 1,
  ACCEPT_EMPTY_MUTATE_TARGET = (1 << 2)
)

# Import corresponding math functions from base so the automatically defined libmxnet
# functions can overload them
import Base: sqrt

"""
The libxmnet APIs are automatically imported from `libmxnet.so`. The functions listed
here operate on `NDArray` objects. The arguments to the functions are typically ordered
as

```julia
  func_name(arg_in1, arg_in2, ..., scalar1, scalar2, ..., arg_out1, arg_out2, ...)
```

unless `NDARRAY_ARG_BEFORE_SCALAR` is not set. In this case, the scalars are put before the input arguments:

```julia
  func_name(scalar1, scalar2, ..., arg_in1, arg_in2, ..., arg_out1, arg_out2, ...)
```

If `ACCEPT_EMPTY_MUTATE_TARGET` is set. An overloaded function without the output arguments will also be defined:

```julia
  func_name(arg_in1, arg_in2, ..., scalar1, scalar2, ...)
```

Upon calling, the output arguments will be automatically initialized with empty NDArrays.

Those functions always return the output arguments. If there is only one output (the typical situation), that
object (`NDArray`) is returned. Otherwise, a tuple containing all the outputs will be returned.
"""
function _get_ndarray_function_def(name::String)
  func_name = Symbol(name)

  func_def = quote
    function $func_name(::Type{<:NDArray}, args::NDArray...; out=nothing, kwargs...)
      if out != nothing
        output_vars = out
        if isa(output_vars, NDArray)
          output_vars = NDArray[output_vars]
        end
        num_outputs = length(output_vars)
      else
        output_vars = NDArray[]
        num_outputs = 0
      end

      args = collect(args)  # tuple to list
      if length(args) == 0
        args = MX_handle[]
      end

      output_handles_pp = if length(output_vars) > 0
        [map(x -> x.handle, output_vars)]
      else
        [Ptr{MX_handle}(C_NULL)]
      end
      num_outputs_p = [convert(Cint, num_outputs)]

      kw_keys_str = String[string(x[1]) for x in kwargs]
      kw_vals_str = String[dump_mx_param(x[2]) for x in kwargs]

      op_handle = _get_cached_libmx_op_handle($(name))
      @mxcall(:MXImperativeInvoke,
              (MX_handle, Cint, Ptr{MX_handle},
               Ptr{Cint}, Ptr{Ptr{MX_handle}},
               Cint, char_pp, char_pp),
              op_handle, length(args), args,
              num_outputs_p, output_handles_pp,
              length(kwargs), kw_keys_str, kw_vals_str)

      if out == nothing
        n = num_outputs_p[]
        hdls = unsafe_wrap(Array{MX_handle}, output_handles_pp[], n)
        xs = NDArray[NDArray(MX_NDArrayHandle(x)) for x in hdls]
        if n == 1
          return xs[]
        else
          return xs
        end
      else
        return out
      end
    end
  end

  func_def2 = quote
    function $func_name(args::NDArray...; out=nothing, kwargs...)
      $func_name(NDArray, args...; out=out, kwargs...)
    end
  end

  return func_def, func_def2
end

const _op_import_bl = [  # import black list; do not import these funcs
    "_full",   # we already have `mx.fill`
    "_ones",   # we already have `mx.ones`
    "_zeros",  # we already have `mx.zeros`
    "clip",
    "expand_dims",

    # arithmetic
    "_plus",
    "_minus",
    "_mod",
    "_mod_scalar",
    "_rmod_scalar",

    "dot",
    "max",
    "max_axis",
    "mean",
    "min",
    "min_axis",
    "prod",
    "reshape",
    "sum",
    "transpose",

    # trigonometric
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",

    # hyperbolic
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",

    # activation
    "sigmoid",
    "relu",
    "softmax",
    "log_softmax",

    # broadcast
    "broadcast_add",
    "broadcast_plus",
    "broadcast_minus",
    "broadcast_sub",
    "broadcast_mul",
    "broadcast_div",
    "broadcast_mod",
    "broadcast_power",
    "broadcast_equal",
    "broadcast_not_equal",
    "broadcast_greater",
    "broadcast_greater_equal",
    "broadcast_lesser",
    "broadcast_lesser_equal",
    "broadcast_maximum",
    "broadcast_minimum",
    "broadcast_to",
    "broadcast_axis",
    "broadcast_axes",
    "broadcast_hypot",

    # reduction
    "argmax",
    "argmin",
]

macro _import_ndarray_functions()
  names = filter(n -> ∉(lowercase(n), _op_import_bl), _get_libmx_op_names())

  func_exprs = map(names) do name
    op_handle = _get_libmx_op_handle(name)

    desc, key_narg = _get_libmx_op_description(name, op_handle)
    func_def, func_def2 = _get_ndarray_function_def(name)

    func_name = Symbol(name)

    import_expr = _import_expr(func_name)

    quote
      $import_expr
      $func_def
      @doc $desc
      $func_def2
    end
  end

  esc(quote
    $(func_exprs...)
  end)
end

@_import_ndarray_functions
