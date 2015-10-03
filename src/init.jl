export MXError

"Exception thrown when an error occurred calling MXNet API."
immutable MXError <: Exception
  msg :: AbstractString
end

################################################################################
# Common types used in MXNet API
################################################################################
typealias MX_uint Cuint
typealias MX_float Cfloat

################################################################################
# Initialization and library API entrance
################################################################################
const MXNET_LIB = Libdl.find_library(["libmxnet.so"], ["/Users/chiyuan/work/mxnet/mxnet/lib"])

function __init__()
  atexit() do
    # notify libmxnet we are shutting down
    ccall( ("MXNotifyShutdown", MXNET_LIB), Cint, () )
  end
end

function mx_get_last_error()
  msg = ccall( ("MXGetLastError", MXNET_LIB), Ptr{UInt8}, () )
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
      value :: Ptr{Void}

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
            @mxcall($(Meta.quot(destructor)), (Ptr{Void},), h.value)
            h.value = C_NULL
          end
        end
      end
    end)

    function Base.cconvert(::Type{Ptr{Void}}, obj::$name)
      obj.value
    end
    function Base.isnull(obj::$name) obj.value == C_NULL end
  end
end

@mx_define_handle_t(MX_NDArrayHandle, MXNDArrayFree)
@mx_define_handle_t(MX_FunctionHandle, nop)

