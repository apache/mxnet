"Exception thrown when an error occurred calling MXNet API."
immutable MXError <: Exception
  msg :: AbstractString
end

################################################################################
# Common types used in MXNet API
################################################################################
typealias MX_uint Cuint
typealias MX_float Cfloat
typealias MX_handle Ptr{Void}

typealias char_p Ptr{UInt8}
typealias char_pp Ptr{char_p}

################################################################################
# Initialization and library API entrance
################################################################################
const MXNET_LIB = Libdl.find_library(["libmxnet.so","libmxnet.dll"],
                                     [joinpath("$(get(ENV,"MXNET_HOME",""))","lib"),
                                      joinpath(Pkg.dir("MXNet"),"deps/usr/lib")])

function __init__()
  _import_ndarray_functions()
  _import_atomic_symbol_creators()
  _import_io_iterators()

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
  return bytestring(msg)
end

"Utility macro to call MXNet API functions"
macro mxcall(fv, argtypes, args...)
  f = eval(fv)
  args = map(esc, args)
  quote
    _mxret = ccall( ($(Meta.quot(f)), $MXNET_LIB),
                    Cint, $argtypes, $(args...) )
    if _mxret != 0
      err_msg = mx_get_last_error()
      throw(MXError(err_msg))
    end
  end
end

################################################################################
# Handle types
################################################################################
macro mx_define_handle_t(name, destructor)
  name = esc(name)
  quote
    type $name
      value :: MX_handle

      function $name(value = C_NULL)
        hdr = new(value)

        $(if destructor != :nop
          :(finalizer(hdr, delete!))
        end)

        return hdr
      end
    end

    $(if finalizer != :nop
      quote
        function delete!(h :: $name)
          if h.value != C_NULL
            @mxcall($(Meta.quot(destructor)), (MX_handle,), h.value)
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

    function Base.isnull(obj::$name) obj.value == C_NULL end
  end
end

@mx_define_handle_t(MX_NDArrayHandle, MXNDArrayFree)
@mx_define_handle_t(MX_FunctionHandle, nop)
@mx_define_handle_t(MX_SymbolHandle, MXSymbolFree)
@mx_define_handle_t(MX_ExecutorHandle, MXExecutorFree)
@mx_define_handle_t(MX_DataIterHandle, MXDataIterFree)
@mx_define_handle_t(MX_KVStoreHandle, MXKVStoreFree)

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
################################################################################
function dump_mx_param(val :: Any)
  string(val)
end
function dump_mx_param{N,T<:Integer}(shape :: NTuple{N, T})
  string(tuple(flipdim([shape...],1)...))
end

"""A convenient macro copied from Mocha.jl that could be used to define structs
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
macro defstruct(name, super_name, fields)
  @assert fields.head == :tuple
  fields = fields.args
  @assert length(fields) > 0
  name = esc(name)

  field_defs     = Array(Expr, length(fields))        # :(field2 :: Int)
  field_names    = Array(Base.Symbol, length(fields)) # :field2
  field_defaults = Array(Expr, length(fields))        # :(field2 :: Int = 0)
  field_asserts  = Array(Expr, length(fields))        # :(field2 >= 0)

  for i = 1:length(fields)
    field = fields[i]
    if field.head == :tuple
      field_asserts[i] = field.args[2]
      field = field.args[1]
    end
    field_defs[i] = esc(field.args[1])
    field_names[i] = field.args[1].args[1]
    field_defaults[i] = Expr(:kw, field.args...)
  end

  # body of layer type, defining fields
  type_body = Expr(:block, field_defs...)

  # constructor
  asserts = map(filter(i -> isdefined(field_asserts,i), 1:length(fields))) do i
    :(@assert($(field_asserts[i])))
  end
  construct = Expr(:call, name, field_names...)
  ctor_body = Expr(:block, asserts..., construct)
  ctor_def = Expr(:call, name, Expr(:parameters, field_defaults...))
  ctor = Expr(:(=), ctor_def, ctor_body)

  quote
    type $(name) <: $super_name
      $type_body
    end

    $ctor
  end
end
