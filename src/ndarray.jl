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

function Base.unsafe_convert(::Type{MX_handle}, obj::NDArray)
  Base.unsafe_convert(MX_handle, obj.handle)
end
Base.convert(t::Type{MX_handle}, obj::NDArray) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::NDArray) = Base.unsafe_convert(t, obj)

################################################################################
# NDArray functions exported to the users
################################################################################
function context(arr :: NDArray)
  ref_typeid = Ref{Cint}(0)
  ref_devid  = Ref{Cint}(0)
  @mxcall(:MXNDArrayGetContext, (MX_handle, Ref{Cint}, Ref{Cint}),
          arr, ref_typeid, ref_devid)
  return Context(ref_typeid[], ref_devid[])
end

function empty{N}(shape :: NTuple{N, Int}, ctx :: Context = DEFAULT_CONTEXT)
  NDArray(_ndarray_alloc(shape, ctx, false))
end
function empty(shape :: Int...)
  empty(shape)
end

#------------------------------------------------------------
# Interface functions similar to Julia Arrays
#------------------------------------------------------------
import Base: size, length, ndims, eltype
function size(arr :: NDArray)
  ref_ndim  = Ref{MX_uint}(0)
  ref_shape = Ref{Ptr{MX_uint}}(0)
  @mxcall(:MXNDArrayGetShape, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_uint}}),
          arr, ref_ndim, ref_shape)
  tuple(map(Int, pointer_to_array(ref_shape[], ref_ndim[]))...)
end
function size(arr :: NDArray, dim :: Int)
  size(arr)[dim]
end
function length(arr :: NDArray)
  prod(size(arr))
end
function ndims(arr :: NDArray)
  length(size(arr))
end
function eltype(arr :: NDArray)
  MX_float
end

"Create zero-ed NDArray of specific shape"
function zeros{N}(shape :: NTuple{N, Int}, ctx :: Context = DEFAULT_CONTEXT)
  arr = empty(shape, ctx)
  arr[:] = 0
  return arr
end
function zeros(shape :: Int...)
  zeros(shape)
end

import Base: setindex!
"Assign all elements of an NDArray to a scalar"
function setindex!(arr :: NDArray, val :: Real, ::Colon)
  _set_value(val, arr)
  return arr
end
function setindex!{T<:Real}(arr :: NDArray, val :: Array{T}, ::Colon)
  copy!(arr, val)
end
function setindex!(arr :: NDArray, val :: NDArray, ::Colon)
  copy!(arr, val)
end

#------------------------------------------------------------
# Copying functions
#------------------------------------------------------------
import Base: copy!, copy
"Copy data between NDArrays"
function copy!(dst :: NDArray, src :: NDArray)
  if dst.handle == src.handle
    warn("Copying an NDArray to itself")
    return
  end

  _copyto(src, dst)
  return dst
end

"Copy data from NDArray to Julia Array"
function copy!(dst :: Array{MX_float}, src :: NDArray)
  @assert size(dst) == size(src)
  @mxcall(:MXNDArraySyncCopyToCPU, (MX_handle, Ptr{MX_float}, Csize_t),
          src, pointer(dst), length(dst))
  return dst
end

"Copy data from Julia Array to NDArray"
function copy!{T<:Real}(dst :: NDArray, src :: Array{T})
  @assert size(dst) == size(src)
  src = convert(Array{MX_float}, src) # this might involve copying
  @mxcall(:MXNDArraySyncCopyFromCPU, (MX_handle, Ptr{MX_float}, Csize_t),
          dst.handle, pointer(src), length(src))
  return dst
end

"Create copy: NDArray -> Julia Array"
function copy(arr :: NDArray)
  j_arr = Array(MX_float, size(arr))
  copy!(j_arr, arr)
end

"Create copy: NDArray -> NDArray in a given context"
function copy(arr :: NDArray, ctx :: Context)
  dst = NDArray(_ndarray_alloc(size(arr), ctx, true))
  copy!(dst, arr)
end

"Create copy: Julia Array -> NDArray in a given context"
function copy{T<:Real}(arr :: Array{T}, ctx :: Context)
  dst = empty(size(arr), ctx)
  copy!(dst, arr)
end


#------------------------------------------------------------
# Basic arithmetics
#------------------------------------------------------------
"""
Julia does not support re-definiton of += operator (like __iadd__ in python),
When one write a += b, it gets translated to a = a+b. a+b will allocate new
memory for the results, and the newly allocated NDArray object is then assigned
back to a, while the original contents in a is discarded. This is very inefficient
when we want to do inplace update.

This macro is a simple utility to implement this behavior. Write

  @mx.inplace a += b

will translate into

  mx.add_to!(a, b)

which will do inplace adding of the contents of b into a.
"""
macro inplace(stmt)
  if stmt.head == :+= || stmt.head == :.+=
    Expr(:call, :add_to!, esc(stmt.args[1]), esc(stmt.args[2]))
  elseif stmt.head == :-= || stmt.head == :.-=
    Expr(:call, :sub_from!, esc(stmt.args[1]), esc(stmt.args[2]))
  else
    error("unsupported inplace translation for $stmt")
  end
end

function add_to!(dst :: NDArray, args :: Union{Real, NDArray}...)
  for arg in args
    if isa(arg, Real)
      _plus_scalar(dst, arg, dst)
    else
      _plus(dst, arg, dst)
    end
  end
  return dst
end

# We fix the first arg to be NDArray to avoid ambiguity
import Base: +, .+
function +(arg0 :: NDArray, args :: Union{Real, NDArray}...)
  ret = copy(arg0, context(arg0))
  add_to!(ret, args...)
end
function .+(arg0 :: NDArray, args :: Union{Real, NDArray}...)
  +(arg0, args...)
end

function sub_from!(dst :: NDArray, arg :: Union{Real, NDArray})
  if isa(arg, Real)
    _minus_scalar(dst, arg, dst)
  else
    _minus(dst, arg, dst)
  end
end
import Base: -, .-
function -(arg0 :: NDArray, arg1 :: Union{Real, NDArray})
  ret = copy(arg0, context(arg0))
  sub_from!(ret, arg1)
end
function .-(arg0 :: NDArray, arg1 :: Union{Real, NDArray})
  -(arg0, arg1)
end
function -(arg0 :: NDArray)
  _mul_scalar(arg0, -1.0)
end

################################################################################
# NDArray functions dynamically exported from libmx
################################################################################
function _invoke_mxfunction(func_handle::MX_handle, use_vars, scalars, mut_vars)
  @mxcall(:MXFuncInvoke,
          (MX_handle, Ptr{MX_handle}, Ptr{MX_float}, Ptr{MX_handle}),
          func_handle, use_vars, scalars, mut_vars)
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
    arg_before_scalar   = (type_mask & convert(Cint,NDARRAY_ARG_BEFORE_SCALAR)) != 0

    # general ndarray function
    if arg_before_scalar
      args = vcat([Expr(:(::), symbol("in$i"), NDArray) for i=1:n_used_vars],
                  [Expr(:(::), symbol("sca$i"), Real) for i=1:n_scalars],
                  [Expr(:(::), symbol("out$i"), NDArray) for i=1:n_mutate_vars])
    else
      args = vcat([Expr(:(::), symbol("sca$i"), Real) for i=1:n_scalars],
                  [Expr(:(::), symbol("in$i"), NDArray) for i=1:n_used_vars],
                  [Expr(:(::), symbol("out$i"), NDArray) for i=1:n_mutate_vars])
    end

    _use_vars = Expr(:ref, :MX_handle, [symbol("in$i") for i=1:n_used_vars]...)
    _scalars  = Expr(:ref, :MX_float, [symbol("sca$i") for i=1:n_scalars]...)
    _mut_vars = Expr(:ref, :MX_handle, [symbol("out$i") for i=1:n_mutate_vars]...)
    stmt_call = Expr(:call, :_invoke_mxfunction, func_handle, _use_vars, _scalars, _mut_vars)
    if n_mutate_vars == 1
      stmt_ret = :(return out1)
    else
      stmt_ret = Expr(:return, Expr(:tuple, [symbol("out$i") for i=1:n_mutate_vars]...))
    end

    func_body = Expr(:block, stmt_call, stmt_ret)
    func_head = Expr(:call, func_name, args...)

    func_def  = Expr(:function, func_head, func_body)
    eval(func_def)

    if accept_empty_mutate
      args0      = args[1:n_used_vars+n_scalars]
      func_head0 = Expr(:call, func_name, args0...)
      _mut_vars0 = [:(NDArray(_ndarray_alloc())) for i=1:n_mutate_vars]
      stmt_call0 = Expr(:call, func_name, args0..., _mut_vars0...)
      func_body0 = Expr(:block, stmt_call0)
      func_head0 = Expr(:call, func_name, args0...)

      func_def0  = Expr(:function, func_head0, func_body0)
      eval(func_def0)
    end
  end
end

