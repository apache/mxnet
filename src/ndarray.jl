export NDArray
export empty

# create a NDArray handle of specific shape
function _ndarray_alloc{N}(shape :: NTuple{N, Int}, ctx :: Context, delay_alloc :: Bool)
  h_ref  = Ref{MX_handle}(0)
  shape  = MX_uint[shape...]
  @mxcall(:MXNDArrayCreate, (Ptr{MX_uint}, MX_uint, Cint, Cint, Cint, Ref{MX_handle}),
      shape, length(shape), ctx.device_type, ctx.device_id, delay_alloc, h_ref)
  handle = MX_NDArrayHandle(h_ref[])
  return handle
end

# create a handle to an empty NDArray, this handle can be used to hold
# results returned by libmx API calls
function _ndarray_alloc()
  h_ref = Ref{MX_handle}(0)
  @mxcall(:MXNDArrayCreateNone, (Ref{MX_handle},), h_ref)
  return MX_NDArrayHandle(h_ref[])
end

################################################################################
# NDArray Type
################################################################################
type NDArray
  handle   :: MX_NDArrayHandle
  writable :: Bool

  function NDArray(handle, writable=true)
    new(handle, writable)
  end
end

################################################################################
# NDArray functions exported to the users
################################################################################
function empty{N}(shape :: NTuple{N, Int}, ctx :: Context = DEFAULT_CONTEXT)
  NDArray(_ndarray_alloc(shape, ctx, false))
end
function empty(shape :: Int...)
  empty(shape)
end

function Base.size(arr :: NDArray)
  ref_ndim  = Ref{MX_uint}(0)
  ref_shape = Ref{Ptr{MX_uint}}(0)
  @mxcall(:MXNDArrayGetShape, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_uint}}),
          arr.handle, ref_ndim, ref_shape)
  tuple(map(Int, pointer_to_array(ref_shape[], ref_ndim[]))...)
end

function to_array(arr :: NDArray)
  out = Array(MX_float, size(arr))
  @mxcall(:MXNDArraySyncCopyToCPU, (MX_handle, Ptr{MX_float}, Csize_t),
          arr.handle, pointer(out), length(out))
  return out
end

################################################################################
# NDArray functions dynamically exported from libmx
################################################################################
module _lib
# this module is used to hold functions automatically imported
# from libmxnet
end
function _register_function(lib::Module, name::Symbol, func::Function)
  eval(lib, quote
    $name = $func
  end)
end

@enum(LIBMX_FUNC_TYPE_MASK,
  NDARRAY_ARG_BEFORE_SCALAR = 1,
  ACCEPT_EMPTY_MUTATE_TARGET = (1 << 2)
)

function _import_ndarray_functions()
  n_ref = Ref{MX_uint}(0)
  h_ref = Ref{Ptr{MX_handle}}(0)
  @mxcall(:MXListFunctions, (Ref{MX_uint}, Ref{Ptr{MX_handle}}), n_ref, h_ref)

  n_funcs = n_ref[]
  h_funcs = pointer_to_array(h_ref[], n_funcs)

  for i = 1:n_funcs
    func_handle = h_funcs[i]

    #----------------------------------------
    # get function information (human readable)
    ref_name = Ref{char_p}(0)
    ref_desc = Ref{char_p}(0)
    ref_narg = Ref{MX_uint}(0)

    ref_arg_names = Ref{char_pp}(0)
    ref_arg_types = Ref{char_pp}(0)
    ref_arg_descs = Ref{char_pp}(0)

    @mxcall(:MXFuncGetInfo,
            (MX_handle, Ref{char_p}, Ref{char_p}, Ref{MX_uint}, Ref{char_pp}, Ref{char_pp}, Ref{char_pp}),
            func_handle, ref_name, ref_desc, ref_narg, ref_arg_names, ref_arg_types, ref_arg_descs)
    func_name = symbol(bytestring(ref_name[]))

    #----------------------------------------
    # get function specification
    ref_n_use_vars = Ref{MX_uint}(0)
    ref_n_scalars  = Ref{MX_uint}(0)
    ref_n_mut_vars = Ref{MX_uint}(0)
    ref_type_mask  = Ref{Cint}(0)
    @mxcall(:MXFuncDescribe,
            (MX_handle, Ref{MX_uint}, Ref{MX_uint}, Ref{MX_uint}, Ref{Cint}),
            func_handle, ref_n_use_vars, ref_n_scalars, ref_n_mut_vars, ref_type_mask)

    #----------------------------------------
    # prepare function definition
    n_used_vars   = ref_n_use_vars[]
    n_scalars     = ref_n_scalars[]
    n_mutate_vars = ref_n_mut_vars[]
    type_mask     = ref_type_mask[]
    accept_empty_mutate = (type_mask & convert(Cint,ACCEPT_EMPTY_MUTATE_TARGET)) != 0
    if (type_mask & convert(Cint,NDARRAY_ARG_BEFORE_SCALAR)) != 0
      use_vars_range = 1:n_used_vars
      scalar_range   = n_used_vars+1:n_used_vars+n_scalars
    else
      scalar_range   = 1:n_scalars
      use_vars_range = n_scalars+1:n_scalars+n_used_vars
    end

    if n_mutate_vars == 1 && n_used_vars == 2 && n_scalars == 0
      println("defining $func_name")
      # binary ndarray function
      function binary_ndarray_function(lhs::NDArray, rhs::NDArray, out::NDArray)
        @assert(out.writable)
        use_vars = MX_handle[lhs.handle, rhs.handle]
        scalars  = MX_float[]
        mut_vars = MX_handle[out.handle]
        @mxcall(:MXFuncInvoke,
                (MX_handle, Ptr{MX_handle}, Ptr{MX_float}, Ptr{MX_handle}),
                func_handle, use_vars, scalars, mut_vars)
        return out
      end
      if accept_empty_mutate
        function binary_ndarray_function(lhs::NDArray, rhs::NDArray)
          out = NDArray(_ndarray_alloc())
          binary_ndarray_function(lhs, rhs, out)
        end
      end

      # add methods to the module
      eval(_lib, quote
        $func_name = $binary_ndarray_function
      end)
    end
  end
end

