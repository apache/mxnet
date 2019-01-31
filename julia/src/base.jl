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

"Exception thrown when an error occurred calling MXNet API."
struct MXError <: Exception
  msg :: AbstractString
end

Base.show(io::IO, e::MXError) = print(io, e.msg)

################################################################################
# Common types used in MXNet API
################################################################################
const MX_uint = Cuint
const MX_float = Cfloat
const MX_handle = Ptr{Cvoid}

const char_p = Ptr{UInt8}
const char_pp = Ptr{char_p}

################################################################################
# Enumeration from MXNet headers
################################################################################
# OpReqType in include/mxnet/op_attr_types.h
@enum GRAD_REQ GRAD_NOP=0 GRAD_WRITE=1 GRAD_INPLACE=2 GRAD_ADD=3
const grad_req_map = Dict{Symbol,GRAD_REQ}(
    :nop     => GRAD_NOP,      # no operation, do not write anything
    :write   => GRAD_WRITE,    # write gradient to provided space
    :inplace => GRAD_INPLACE,  # perform an inplace write
    :add     => GRAD_ADD,      # add to the provided space
)

################################################################################
# Initialization and library API entrance
################################################################################
const MXNET_LIB = Libdl.find_library(["libmxnet.$(Libdl.dlext)", "libmxnet.so"],  # see build.jl
                                     [joinpath(get(ENV, "MXNET_HOME", ""), "lib"),
                                      get(ENV, "MXNET_HOME", ""),
                                      joinpath(@__DIR__, "..",
                                               "deps", "usr", "lib")])
const LIB_VERSION = Ref{Cint}(0)

if isempty(MXNET_LIB)
  # touch this file, so that after the user properly build libmxnet, the precompiled
  # MXNet.ji will be re-compiled to get MXNET_LIB properly.
  touch(@__FILE__)
  error("Cannot find or load libmxnet.$(Libdl.dlext). " *
        "Please see the document on how to build it.")
else
  include_dependency(MXNET_LIB)
end

function __init__()
  # TODO: bug in nnvm, if do not call this, call get handle "_copyto" will fail
  _get_libmx_op_names()
  _populate_iter_creator_cache!()
  _get_lib_version!()

  atexit() do
    # notify libmxnet we are shutting down
    ccall( ("MXNotifyShutdown", MXNET_LIB), Cint, () )
  end
end

function mx_get_last_error()
  msg = ccall( ("MXGetLastError", MXNET_LIB), char_p, () )
  if msg == C_NULL
    throw(MXError("Failed to get last error message"))
  end
  return unsafe_string(msg)
end

"Utility macro to call MXNet API functions"
macro mxcall(fv, argtypes, args...)
  f = eval(fv)
  args = map(esc, args)
  quote
    _mxret = ccall(($(QuoteNode(f)), $MXNET_LIB),
                   Cint, $argtypes, $(args...))
    if _mxret != 0
      err_msg = mx_get_last_error()
      throw(MXError(err_msg))
    end
  end
end

"""
Get libmxnet version

This function will changes the global variable `LIB_VERSION`.
"""
function _get_lib_version!()
  @mxcall :MXGetVersion (Ref{Cint},) LIB_VERSION
  LIB_VERSION[]
end

################################################################################
# Handle types
################################################################################
function mx_define_handle_t(name, destructor)
  @eval begin
    mutable struct $name
      value::MX_handle

      function $name(value = C_NULL)
        hdr = new(value)

        $(if destructor != nothing
          :(finalizer(delete!, hdr))
        end)

        return hdr
      end
    end

    $(if finalizer != nothing
      quote
        function delete!(h :: $name)
          if h.value != C_NULL
            @mxcall($(QuoteNode(destructor)), (MX_handle,), h.value)
            h.value = C_NULL
          end
        end
      end
    end)

    function Base.unsafe_convert(::Type{MX_handle}, obj::$name)
      obj.value
    end
    Base.convert(t::Type{MX_handle}, obj::$name) = Base.unsafe_convert(t, obj)
    Base.cconvert(t::Type{MX_handle}, obj::$name) = Base.unsafe_convert(t, obj)

    MX_handle(x::$name) = Base.convert(MX_handle, x)
  end
end

mx_define_handle_t(:MX_NDArrayHandle,  :MXNDArrayFree)
mx_define_handle_t(:MX_OpHandle,       nothing)
mx_define_handle_t(:MX_SymbolHandle,   :MXSymbolFree)
mx_define_handle_t(:MX_ExecutorHandle, :MXExecutorFree)
mx_define_handle_t(:MX_DataIterHandle, :MXDataIterFree)
mx_define_handle_t(:MX_KVStoreHandle,  :MXKVStoreFree)

################################################################################
# MXNet Params
#
# MXNet API use string to pass some common parameters like the configurations
# when defining layers. Typically, it is enough to use string(obj) to get a
# recognizable representation for libmxnet. However, there is currently a
# caveat:
#
# Because Julia use column-major ordering for tensors. In order to properly
# interact with Julia Arrays, the shape will look "reversed" from the Julia
# side. For example, a typical MNIST mini-batch tensor is of shape (28,28,1,100)
# from Julia side, while the shape information for the same piece of memory
# should be interpreted as (100,1,28,28) from C/C++/Python side.
#
# Therefore, when passing parameters to libmxnet, we should reverse the shape
# parameter. For example, when the user specify a non-square kernel size for
# a convolution or pooling layer. Unfortunately, those operators are automatically
# imported, and information about the type of each parameter is somehow limited.
# One hacky way is to match the type description for the string "Shape(tuple)"
# when importing operators. But currently we simply decided to reverse **all**
# NTuple{N, Int} passed to libmxnet.
#
# TODO: find a better solution in case this cause issues in the future.
# I made `@_remap` in `ndarray.jl`. (Iblis Lin)
################################################################################
dump_mx_param(val::Any)        = string(val)
dump_mx_param(val::Float64)    = @sprintf("%.16e", val)
dump_mx_param(val::Float32)    = @sprintf("%.8e", val)
dump_mx_param(val::Float16)    = @sprintf("%.4e", val)
dump_mx_param(val::Irrational) = @sprintf("%.16e", val)
dump_mx_param(shape::NTuple{N,<:Integer}) where N =
  string(reverse(shape))


"""
A convenient macro copied from Mocha.jl that could be used to define structs
with default values and type checks. For example
```julia
@defstruct MyStruct Any (
  field1 :: Int = 0,
  (field2 :: AbstractString = "", !isempty(field2))
)
```
where each field could be either
```julia
field_name :: field_type = default_value
```
or put within a tuple, with the second element
specifying a validation check on the field value.
In the example above, the default value for
field2 does not satisfy the assertion, this
could be used to force user to provide a
valid value when no meaningful default value
is available.

The macro will define a constructor that could accept
the keyword arguments.
"""
macro defstruct(name, fields)
  _defstruct_impl(false, name, fields)
end

"""A convenient macro to define immutable structs. The same as
`@defstruct` except that the defined type is immutable.
"""
macro defimmutable(name, fields)
  _defstruct_impl(true, name, fields)
end

"""Internal use only, this value is used to indicate a required value
is not specified.
"""
struct __Undefined
end

function _defstruct_impl(is_immutable, name, fields)
  if isa(fields, Expr) && fields.head == :tuple
    fields = fields.args
  else
    fields = [fields]
  end
  @assert length(fields) > 0

  if isa(name, Symbol)
    name       = esc(name)
    super_name = :Any
  else
    @assert(isa(name, Expr) && name.head == :(<:) && length(name.args) == 2 &&
            isa(name.args[1], Symbol) && isa(name.args[2], Symbol),
            "name must be of form 'Name <: SuperType'")

    super_name = esc(name.args[2])
    name       = esc(name.args[1])
  end

  field_defs     = Vector{Expr}(undef, length(fields))  # :(field2 :: Int)
  field_names    = Vector{Expr}(undef, length(fields))  # :field2
  field_defaults = Vector{Expr}(undef, length(fields))  # :(field2 = 0)
  field_types    = Vector{Expr}(undef, length(fields))  # Int
  field_asserts  = Vector{Expr}(undef, length(fields))  # :(field2 >= 0)
  required_field = Symbol[]

  for i = 1:length(fields)
    field = fields[i]
    if field.head == :tuple
      field_asserts[i] = esc(field.args[2])
      field = field.args[1]
    end
    if field.head == :(=)
      fname             = field.args[1].args[1]
      field_defs[i]     = esc(field.args[1])
      field_names[i]    = esc(fname)
      field_types[i]    = esc(field.args[1].args[2])
      field_defaults[i] = Expr(:kw, fname, esc(field.args[2]))
    else
      # no default value provided, required field
      fname             = field.args[1]
      field_defs[i]     = esc(field)
      field_names[i]    = esc(fname)
      field_types[i]    = esc(field.args[2])
      field_defaults[i] = Expr(:kw, fname, __Undefined())
      push!(required_field, fname)
    end
  end

  # body of layer type, defining fields
  type_body = Expr(:block, field_defs...)

  # constructor
  requires = map(required_field) do fname
    :(@assert(!isa($fname, __Undefined), "value for " * string($fname) * " is required"))
  end
  converts = map(zip(field_names, field_types)) do param
    f_name, f_type = param
    :($f_name = convert($f_type, $f_name))
  end
  asserts = map(filter(i -> isassigned(field_asserts,i), 1:length(fields))) do i
    :(@assert($(field_asserts[i])))
  end
  construct = Expr(:call, name, field_names...)
  ctor_body = Expr(:block, requires..., converts..., asserts..., construct)
  ctor_def = Expr(:call, name, Expr(:parameters, field_defaults...))
  ctor = Expr(:(=), ctor_def, ctor_body)

  if is_immutable
    quote
      struct $(name) <: $(super_name)
        $type_body
      end

      $ctor
    end
  else
    quote
      mutable struct $(name) <: $(super_name)
        $type_body
      end

      $ctor
    end
  end
end
