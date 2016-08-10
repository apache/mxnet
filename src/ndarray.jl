# All the types supported by mshadow.
typealias DType Union{Float32, Float64, Float16, UInt8, Int32}
@enum TypeFlag kFloat32 kFloat64 kFloat16 kUint8 kInt32
typealias DEFAULT_DTYPE Float32

function toTypeFlag{T <: DType}(:: Type{T})
  if T == Float32
    return kFloat32
  elseif T == Float64
    return kFloat64
  elseif T == Float16
    return kFloat16
  elseif T == UInt8
    return kUint8
  elseif T == Int32
    return kInt32
  else
    throw(ArgumentError("Can't convert $T to DType."))
  end
end

function fromTypeFlag(T :: TypeFlag)
  if T == kFloat32
    return Float32
  elseif T == kFloat64
    return Float64
  elseif T == kFloat16
    return Float16
  elseif T == kUint8
    return UInt8
  elseif T == kInt32
    return Int32
  else
    throw(ArgumentError("Can't convert DType $T."))
  end
end

# create a NDArray handle of specific shape
function _ndarray_alloc{N}(shape :: NTuple{N, Int}, ctx :: Context, delay_alloc :: Bool)
  h_ref  = Ref{MX_handle}(0)
  shape  = flipdim(MX_uint[shape...],1)
  @mxcall(:MXNDArrayCreate, (Ptr{MX_uint}, MX_uint, Cint, Cint, Cint, Ref{MX_handle}),
      shape, length(shape), ctx.device_type, ctx.device_id, delay_alloc, h_ref)
  handle = MX_NDArrayHandle(h_ref[])
  return handle
end

# create a NDArray handle of specific shape type
function _ndarray_alloc{T <: DType,N}(:: Type{T}, shape :: NTuple{N, Int}, ctx :: Context, delay_alloc :: Bool)
  h_ref  = Ref{MX_handle}(0)
  shape  = flipdim(MX_uint[shape...],1)
  dtype  = toTypeFlag(T)
  @mxcall(:MXNDArrayCreateEx, (Ptr{MX_uint}, MX_uint, Cint, Cint, Cint, Cint, Ref{MX_handle}),
      shape, length(shape), ctx.device_type, ctx.device_id, delay_alloc, dtype, h_ref)
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
"""
    NDArray

   Wrapper of the ``NDArray`` type in ``libmxnet``. This is the basic building block
   of tensor-based computation.

   .. _ndarray-shape-note:

   .. note::

      since C/C++ use row-major ordering for arrays while Julia follows a
      column-major ordering. To keep things consistent, we keep the underlying data
      in their original layout, but use *language-native* convention when we talk
      about shapes. For example, a mini-batch of 100 MNIST images is a tensor of
      C/C++/Python shape (100,1,28,28), while in Julia, the same piece of memory
      have shape (28,28,1,100).
"""
type NDArray
  handle   :: MX_NDArrayHandle
  writable :: Bool

  function NDArray(handle, writable=true)
    new(handle, writable)
  end
end

function Base.show(io :: IO, arr :: NDArray)
  print(io, "mx.NDArray{$(eltype(arr))}$(size(arr))")
end

function NDArray{T<:Real}(data :: Array{T})
  copy(data, cpu())
end

function Base.unsafe_convert(::Type{MX_handle}, obj::NDArray)
  Base.unsafe_convert(MX_handle, obj.handle)
end
Base.convert(t::Type{MX_handle}, obj::NDArray) = Base.unsafe_convert(t, obj)
Base.cconvert(t::Type{MX_handle}, obj::NDArray) = Base.unsafe_convert(t, obj)

################################################################################
# NDArray functions exported to the users
################################################################################
"""
    context(arr :: NDArray)

   Get the context that this :class:`NDArray` lives on.
"""
function context(arr :: NDArray)
  ref_typeid = Ref{Cint}(0)
  ref_devid  = Ref{Cint}(0)
  @mxcall(:MXNDArrayGetContext, (MX_handle, Ref{Cint}, Ref{Cint}),
          arr, ref_typeid, ref_devid)
  return Context(ref_typeid[], ref_devid[])
end


"""
.. function::
   empty(DType, shape :: Tuple, ctx :: Context)
   empty(DType, shape :: Tuple)
   empty(DType, dim1, dim2, ...)

   Allocate memory for an uninitialized :class:`NDArray` with a specified type.
"""
function empty{N,T<:DType}(::Type{T}, shape :: NTuple{N, Int})
  empty(T, shape, cpu())
end
function empty{N,T<:DType}(:: Type{T}, shape :: NTuple{N, Int}, ctx :: Context)
  NDArray(_ndarray_alloc(T, shape, ctx, false))
end
function empty{T<:DType}(:: Type{T}, shape :: Int...)
  empty(T, shape)
end

"""
.. function::
   empty(shape :: Tuple, ctx :: Context)
   empty(shape :: Tuple)
   empty(dim1, dim2, ...)

   Allocate memory for an uninitialized :class:`NDArray` with specific shape of type Float32.
"""
function empty{N}(shape :: NTuple{N, Int})
  empty(shape, cpu())
end
function empty{N}(shape :: NTuple{N, Int}, ctx :: Context)
  NDArray(_ndarray_alloc(shape, ctx, false))
end
function empty(shape :: Int...)
  empty(shape)
end

"""
    zeros(DType, shape :: Tuple, ctx :: Context)
    zeros(DType, shape :: Tuple)
    zeros(DType, dim1, dim2, ...)

Create zero-ed :class:`NDArray` with specific shape and type
"""
function zeros{N,T<:DType}(:: Type{T}, shape :: NTuple{N, Int})
  zeros(T, shape, cpu())
end
function zeros{N,T<:DType}(:: Type{T}, shape :: NTuple{N, Int}, ctx :: Context)
  arr = empty(T, shape, ctx)
  arr[:] = zero(T)
  return arr
end
function zeros{T<:DType}(:: Type{T}, shape :: Int...)
  zeros(T, shape)
end

"""
    zeros(shape :: Tuple, ctx :: Context)
    zeros(shape :: Tuple)
    zeros(dim1, dim2, ...)

Create zero-ed :class:`NDArray` with specific shape.
"""
function zeros{N}(shape :: NTuple{N, Int})
  zeros(shape, cpu())
end
function zeros{N}(shape :: NTuple{N, Int}, ctx :: Context)
  arr = empty(shape, ctx)
  arr[:] = 0
  return arr
end
function zeros(shape :: Int...)
  zeros(shape)
end

"""
    ones(DType, shape :: Tuple, ctx :: Context)
    ones(DType, shape :: Tuple)
    ones(DType, dim1, dim2, ...)

Create an :class:`NDArray` with specific shape & type, and initialize with 1.
"""
function ones{N,T<:DType}(:: Type{T}, shape :: NTuple{N, Int})
  ones(T, shape, cpu())
end
function ones{N,T<:DType}(:: Type{T}, shape :: NTuple{N, Int}, ctx :: Context)
  arr = empty(T, shape, ctx)
  arr[:] = one(T)
  return arr
end
function ones{T<:DType}(:: Type{T}, shape :: Int...)
  ones(T, shape)
end

"""
    ones(shape :: Tuple, ctx :: Context)
    ones(shape :: Tuple)
    ones(dim1, dim2, ...)

Create an :class:`NDArray` with specific shape and initialize with 1.
"""
function ones{N}(shape :: NTuple{N, Int})
  ones(shape, cpu())
end
function ones{N}(shape :: NTuple{N, Int}, ctx :: Context)
  arr = empty(shape, ctx)
  arr[:] = 1
  return arr
end
function ones(shape :: Int...)
  ones(shape)
end

import Base: size, length, ndims, eltype

"""
.. function::
   size(arr :: NDArray)
   size(arr :: NDArray, dim :: Int)

   Get the shape of an :class:`NDArray`. The shape is in Julia's column-major convention. See
   also the :ref:`notes on NDArray shapes <ndarray-shape-note>`.
"""
function size(arr :: NDArray)
  ref_ndim  = Ref{MX_uint}(0)
  ref_shape = Ref{Ptr{MX_uint}}(0)
  @mxcall(:MXNDArrayGetShape, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_uint}}),
          arr, ref_ndim, ref_shape)
  tuple(map(Int, flipdim(unsafe_wrap(Array, ref_shape[], ref_ndim[]),1))...)
end
function size(arr :: NDArray, dim :: Int)
  size(arr)[dim]
end

"""
    length(arr :: NDArray)

   Get the number of elements in an :class:`NDArray`.
"""
function length(arr :: NDArray)
  prod(size(arr))
end

"""
    ndims(arr :: NDArray)

   Get the number of dimensions of an :class:`NDArray`. Is equivalent to ``length(size(arr))``.
"""
function ndims(arr :: NDArray)
  length(size(arr))
end

"""
    eltype(arr :: NDArray)

   Get the element type of an :class:`NDArray`.
"""
function eltype{T <: Union{NDArray, MX_NDArrayHandle}}(arr :: T)
  dtype_ref = Ref{Cint}(0)
  @mxcall(:MXNDArrayGetDType, (MX_handle, Ptr{Cint}), arr, dtype_ref)

  if dtype_ref[] == -1 # arr->is_none()
    warn("Eltype of $arr is not defined")
    Base.show_backtrace(STDOUT,backtrace())
    println()
    return Float32
  else
    return fromTypeFlag(TypeFlag(dtype_ref[]))
  end
end


import Base: slice
"""
    slice(arr :: NDArray, start:stop)

   Create a view into a sub-slice of an :class:`NDArray`. Note only slicing at the slowest
   changing dimension is supported. In Julia's column-major perspective, this is the last
   dimension. For example, given an :class:`NDArray` of shape (2,3,4), ``slice(array, 2:3)`` will create
   a :class:`NDArray` of shape (2,3,2), sharing the data with the original array. This operation is
   used in data parallelization to split mini-batch into sub-batches for different devices.
"""
function slice(arr :: NDArray, ::Colon)
  arr
end
function slice(arr :: NDArray, slice::UnitRange{Int})
  dim1 = size(arr)[end]
  @assert(1 <= slice.start <= slice.stop <= dim1)
  if slice.start == 1 && slice.stop == dim1
    return arr
  end

  hdr_ref = Ref{MX_handle}(0)
  # note Julia is 1-based, inclusive-inclusive indexing, while C++ is
  # 0-based, inclusive-exclusive indexing. So 1:3 in Julia should
  # translates into 0:3 in C++.
  @mxcall(:MXNDArraySlice, (MX_handle, MX_uint, MX_uint, Ref{MX_handle}),
          arr, slice.start-1, slice.stop, hdr_ref)
  return NDArray(MX_NDArrayHandle(hdr_ref[]), arr.writable)
end

import Base: setindex!

"""
    setindex!(arr :: NDArray, val, idx)

   Assign values to an :class:`NDArray`. Elementwise assignment is not implemented, only the following
   scenarios are supported

   - ``arr[:] = val``: whole array assignment, ``val`` could be a scalar or an array (Julia ``Array``
     or :class:`NDArray`) of the same shape.
   - ``arr[start:stop] = val``: assignment to a *slice*, ``val`` could be a scalar or an array of
     the same shape to the slice. See also :func:`slice`.
"""
function setindex!(arr :: NDArray, val :: Real, ::Colon)
  @assert(arr.writable)
  _set_value(convert(eltype(arr), val), arr)
  return arr
end
function setindex!{T<:Real}(arr :: NDArray, val :: Array{T}, ::Colon)
  copy!(arr, val)
end
function setindex!(arr :: NDArray, val :: NDArray, ::Colon)
  copy!(arr, val)
end
function setindex!{T<:Real}(arr :: NDArray, val :: Union{T,Array{T},NDArray}, idx::UnitRange{Int})
  setindex!(slice(arr, idx), val, Colon())
end

import Base: getindex
"""
    getindex(arr :: NDArray, idx)

Shortcut for :func:`slice`. A typical use is to write

   .. code-block:: julia

      arr[:] += 5

   which translates into

   .. code-block:: julia

      arr[:] = arr[:] + 5

   which furthur translates into

   .. code-block:: julia

      setindex!(getindex(arr, Colon()), 5, Colon())

   .. note::

      The behavior is quite different from indexing into Julia's ``Array``. For example, ``arr[2:5]``
      create a **copy** of the sub-array for Julia ``Array``, while for :class:`NDArray`, this is
      a *slice* that shares the memory.
"""
function getindex(arr :: NDArray, ::Colon)
  return arr
end

"""
Shortcut for `slice`. **NOTE** the behavior for Julia's built-in index slicing is to create a
copy of the sub-array, while here we simply call `slice`, which shares the underlying memory.
"""
function getindex(arr :: NDArray, idx::UnitRange{Int})
  slice(arr, idx)
end

import Base: copy!, copy, convert
"""
.. function::
   copy!(dst :: Union{NDArray, Array}, src :: Union{NDArray, Array})

   Copy contents of ``src`` into ``dst``.
"""
function copy!(dst :: NDArray, src :: NDArray)
  @assert(dst.writable)
  if dst.handle == src.handle
    warn("Copying an NDArray to itself")
    return
  end

  _copyto(src, dst)
  return dst
end

function copy!{T<:DType}(dst :: Array{T}, src :: NDArray)
  @assert T == eltype(src)
  @assert size(dst) == size(src)
  @mxcall(:MXNDArraySyncCopyToCPU, (MX_handle, Ptr{Void}, Csize_t),
          src, pointer(dst), length(dst))
  return dst
end
function copy!{T<:Real}(dst :: Array{T}, src :: NDArray)
  copy!(dst, copy(src))
end

function copy!{T<:Real}(dst :: NDArray, src :: Array{T})
  @assert dst.writable
  @assert size(dst) == size(src)
  src = convert(Array{eltype(dst)}, src) # this might involve copying
  @mxcall(:MXNDArraySyncCopyFromCPU, (MX_handle, Ptr{Void}, Csize_t),
          dst.handle, pointer(src), length(src))
  return dst
end

function copy_ignore_shape!{T<:Real}(dst :: NDArray, src :: Array{T})
  @assert dst.writable
  @assert length(dst) == length(src)
  src = convert(Array{eltype(dst)}, src) # this might involve copying
  @mxcall(:MXNDArraySyncCopyFromCPU, (MX_handle, Ptr{Void}, Csize_t),
          dst.handle, pointer(src), length(src))
  return dst
end


"""
.. function::
   copy(arr :: NDArray)
   copy(arr :: NDArray, ctx :: Context)
   copy(arr :: Array, ctx :: Context)

   Create a copy of an array. When no :class:`Context` is given, create a Julia ``Array``.
   Otherwise, create an :class:`NDArray` on the specified context.
"""
# Create copy: NDArray -> Julia Array
function copy(arr :: NDArray)
  j_arr = Array{eltype(arr)}(size(arr))
  copy!(j_arr, arr)
end

# Create copy: NDArray -> NDArray in a given context
function copy(arr :: NDArray, ctx :: Context)
  dst = NDArray(_ndarray_alloc(eltype(arr), size(arr), ctx, true))
  copy!(dst, arr)
end

# Create copy: Julia Array -> NDArray in a given context
function copy{T<:DType}(arr :: Array{T}, ctx :: Context)
  dst = empty(T, size(arr), ctx)
  copy!(dst, arr)
end

"""
    convert(::Type{Array{T}}, arr :: NDArray)

   Convert an :class:`NDArray` into a Julia ``Array`` of specific type. Data will be copied.
"""
# Convert copy: NDArray -> Julia Array
function convert{T<:Real}(t::Type{Array{T}}, arr :: NDArray)
  convert(t, copy(arr))
end

"""
    @inplace

Julia does not support re-definiton of ``+=`` operator (like ``__iadd__`` in python),
When one write ``a += b``, it gets translated to ``a = a+b``. ``a+b`` will allocate new
memory for the results, and the newly allocated :class:`NDArray` object is then assigned
back to a, while the original contents in a is discarded. This is very inefficient
when we want to do inplace update.

This macro is a simple utility to implement this behavior. Write

   .. code-block:: julia

      @mx.inplace a += b

   will translate into

   .. code-block:: julia

      mx.add_to!(a, b)

   which will do inplace adding of the contents of ``b`` into ``a``.
"""
macro inplace(stmt)
  if stmt.head == :+= || stmt.head == :.+=
    Expr(:call, :add_to!, esc(stmt.args[1]), esc(stmt.args[2]))
  elseif stmt.head == :-= || stmt.head == :.-=
    Expr(:call, :sub_from!, esc(stmt.args[1]), esc(stmt.args[2]))
  elseif stmt.head == :.*=
    Expr(:call, :mul_to!, esc(stmt.args[1]), esc(stmt.args[2]))
  elseif stmt.head == :./=
    Expr(:call, :div_from!, esc(stmt.args[1]), esc(stmt.args[2]))
  else
    error("unsupported inplace translation for $stmt")
  end
end

"""
    add_to!(dst :: NDArray, args :: Union{Real, NDArray}...)

Add a bunch of arguments into ``dst``. Inplace updating.
"""
function add_to!(dst :: NDArray, args :: Union{Real, NDArray}...)
  @assert dst.writable
  for arg in args
    if isa(arg, Real)
      _plus_scalar(dst, convert(eltype(dst), arg), dst)
    else
      _plus(dst, arg, dst)
    end
  end
  return dst
end

import Base: +, .+

"""
    +(args...)
    .+(args...)

Summation. Multiple arguments of either scalar or :class:`NDArray` could be
added together. Note at least the first or second argument needs to be an :class:`NDArray` to
avoid ambiguity of built-in summation.
"""
function +(arg0 :: NDArray, args :: Union{Real, NDArray}...)
  ret = copy(arg0, context(arg0))
  add_to!(ret, args...)
end
function .+(arg0 :: NDArray, args :: Union{Real, NDArray}...)
  +(arg0, args...)
end
function +(arg0 :: Real, arg1 :: NDArray, args :: Union{Real, NDArray}...)
  +(arg1, arg0, args...)
end
function .+(arg0 :: Real, arg1 :: NDArray, args :: Union{Real, NDArray}...)
  .+(arg1, arg0, args...)
end

"""
    sub_from!(dst :: NDArray, args :: Union{Real, NDArray}...)

   Subtract a bunch of arguments from ``dst``. Inplace updating.
"""
function sub_from!(dst :: NDArray, arg :: Union{Real, NDArray})
  @assert dst.writable
  if isa(arg, Real)
    _minus_scalar(dst, convert(eltype(dst), arg), dst)
  else
    _minus(dst, arg, dst)
  end
end

import Base: -, .-

"""
    -(arg0, arg1)
    -(arg0)
    .-(arg0, arg1)

Subtraction ``arg0 - arg1``, of scalar types or :class:`NDArray`. Or create
the negative of ``arg0``.
"""
function -(arg0 :: NDArray, arg1 :: Union{Real, NDArray})
  ret = copy(arg0, context(arg0))
  sub_from!(ret, arg1)
end
function .-(arg0 :: NDArray, arg1 :: Union{Real, NDArray})
  -(arg0, arg1)
end
function -(arg0 :: Real, arg1 :: NDArray)
  ret = -arg1
  add_to!(ret, arg0)
  return ret
end
function .-(arg0 :: Real, arg1 :: NDArray)
  -(arg0, arg1)
end

function -(arg0 :: NDArray)
  _mul_scalar(arg0, -one(eltype(arg0)))
end

"""
    mul_to!(dst :: NDArray, arg :: Union{Real, NDArray})

   Elementwise multiplication into ``dst`` of either a scalar or an :class:`NDArray` of the same shape.
   Inplace updating.
"""
function mul_to!(dst :: NDArray, arg :: Union{Real, NDArray})
  @assert dst.writable
  if isa(arg, Real)
    _mul_scalar(dst, convert(eltype(dst), arg), dst)
  else
    _mul(dst, arg, dst)
  end
  return dst
end

import Base: .*, *

"""
    .*(arg0, arg1)

Elementwise multiplication of ``arg0`` and ``arg``, could be either scalar or :class:`NDArray`.
"""
function .*(arg0 :: NDArray, arg :: Union{Real, NDArray})
  ret = copy(arg0, context(arg0))
  mul_to!(ret, arg)
end
function .*(arg0 :: Real, arg :: NDArray)
  .*(arg, arg0)
end

"""
    *(arg0, arg1)

Currently only multiplication a scalar with an :class:`NDArray` is implemented. Matrix multiplication
is to be added soon.
"""
function *(arg0 :: NDArray, arg :: Real)
  ret = copy(arg0, context(arg0))
  mul_to!(ret, arg)
end
function *(arg0 :: Real, arg :: NDArray)
  *(arg, arg0)
end

"""
    div_from!(dst :: NDArray, arg :: Union{Real, NDArray})

Elementwise divide a scalar or an :class:`NDArray` of the same shape from ``dst``. Inplace updating.
"""
function div_from!(dst :: NDArray, arg :: Union{Real, NDArray})
  @assert dst.writable
  if isa(arg, Real)
    _div_scalar(dst, convert(eltype(dst), arg), dst)
  else
    _div(dst, arg, dst)
  end
end

import Base: ./, /
"""
    ./(arg0 :: NDArray, arg :: Union{Real, NDArray})

Elementwise dividing an :class:`NDArray` by a scalar or another :class:`NDArray` of the same shape.
"""
function ./(arg0 :: NDArray, arg :: Union{Real, NDArray})
  ret = copy(arg0, context(arg0))
  div_from!(ret, arg)
end

"""
    /(arg0 :: NDArray, arg :: Real)

Divide an :class:`NDArray` by a scalar. Matrix division (solving linear systems) is not implemented yet.
"""
function /(arg0 :: NDArray, arg :: Real)
  ./(arg0, arg)
end


"""
Manipulating as Julia Arrays
----------------------------

    @nd_as_jl(captures..., statement)

   A convenient macro that allows to operate :class:`NDArray` as Julia Arrays. For example,

   .. code-block:: julia

      x = mx.zeros(3,4)
      y = mx.ones(3,4)
      z = mx.zeros((3,4), mx.gpu())

      @mx.nd_as_jl ro=(x,y) rw=z begin
        # now x, y, z are just ordinary Julia Arrays
        z[:,1] = y[:,2]
        z[:,2] = 5
      end

   Under the hood, the macro convert all the declared captures from :class:`NDArray` into Julia
   Arrays, by using :func:`try_get_shared`. And automatically commit the modifications back into
   the :class:`NDArray` that is declared as ``rw``. This is useful for fast prototyping and when
   implement non-critical computations, such as :class:`AbstractEvalMetric`.

   .. note::

      - Multiple ``rw`` and / or ``ro`` capture declaration could be made.
      - The macro does **not** check to make sure that ``ro`` captures are not modified. If the
        original :class:`NDArray` lives in CPU memory, then it is very likely the corresponding
        Julia Array shares data with the :class:`NDArray`, so modifying the Julia Array will also
        modify the underlying :class:`NDArray`.
      - More importantly, since the :class:`NDArray` is
        asynchronized, we will wait for *writing* for ``rw`` variables but wait only for *reading*
        in ``ro`` variables. If we write into those ``ro`` variables, **and** if the memory is
        shared, racing condition might happen, and the behavior is undefined.
      - When an :class:`NDArray` is declared to be captured as ``rw``, its contents is always sync
        back in the end.
      - The execution results of the expanded macro is always ``nothing``.
      - The statements are wrapped in a ``let``, thus locally introduced new variables will not be
        available after the statements. So you will need to declare the variables before calling the
        macro if needed.
"""
macro nd_as_jl(m_args...)
  @assert(length(m_args) > 0)
  stmts = m_args[end]
  @assert(isa(stmts, Expr) && stmts.head == :block,
          "The last argument should be a statement block (begin-end); but get $stmts")
  stmts = esc(stmts)

  dclrs  = m_args[1:end-1]
  nd_ro  = []
  nd_rw  = []
  nd_all = []
  for declr in dclrs
    @assert(isa(declr, Expr) && declr.head == :(=) && length(declr.args)==2 && declr.args[1] ∈ (:ro,:rw),
            "Invalid declaration, should be rw=(x,y) or ro=z; but get $declr")

    declr_vars = declr.args[2]
    if isa(declr_vars, Symbol)
      declr_vars = (declr_vars,)
    elseif isa(declr_vars, Expr)
      @assert(declr_vars.head ∈ (:tuple, :vect),
              "Capture declaration should be a variable or a tuple of variables; but got $declr_vars")
      declr_vars = declr_vars.args
    else
      @assert(false, "Capture declaration should be a variable or a tuple of variables; but got $declr_vars")
    end
    for declr_var in declr_vars
      @assert(isa(declr_var, Symbol),
              "Captured ndarrays in ro/rw declaration should be variables, but get $(declr_var)")
    end
    append!(nd_all, [declr_vars...])
    if declr.args[1] == :ro
      append!(nd_ro, [declr_vars...])
    else
      append!(nd_rw, [declr_vars...])
    end
  end

  nd_ro    = map(esc, nd_ro)
  nd_rw    = map(esc, nd_rw)
  nd_all   = map(esc, nd_all)
  rw_origs = [gensym() for _ in nd_rw]

  save_statements  = Expr(:block, [:($v_orig = $v) for (v_orig, v) in zip(rw_origs, nd_rw)]...)
  wait_statements  = Expr(:block, [:(_wait_to_read($v)) for v in nd_ro]...,
                                  [:(_wait_to_write($v)) for v in nd_rw]...)
  clear_statements = Expr(:block, [:($v_orig = nothing) for v_orig in rw_origs]...)
  let_assignments  = [:($v = try_get_shared($v)) for v in nd_all]
  sync_statements  = map(rw_origs, nd_rw) do v_orig, v
    quote
      if !is_shared($v, $v_orig)
        # copy data back if not or no longer sharing data
        copy!($v_orig, $v)
      end
    end
  end
  sync_statements  = Expr(:block, sync_statements...)

  let_statement = Expr(:let, quote
    $stmts
    $sync_statements
  end, let_assignments...)
  m_body = quote
    $wait_statements
    $save_statements
    $let_statement
    $clear_statements
    nothing # the final results is always nothing
  end

  m_body
end

# NOTE: internal use only. Accessing pointers on a different device (e.g. accessing GPU
# pointers from CPU) leads to undefined behavior.
import Base.pointer
function pointer(arr :: NDArray)
  pdata = Ref{Ptr{MX_float}}(0)
  @mxcall(:MXNDArrayGetData, (MX_handle, Ref{Ptr{MX_float}}), arr, pdata)
  return convert(Ptr{eltype(arr)}, pdata[])
end
function _wait_to_read(arr :: NDArray)
  @mxcall(:MXNDArrayWaitToRead, (MX_handle,), arr)
end
function _wait_to_write(arr :: NDArray)
  @mxcall(:MXNDArrayWaitToWrite, (MX_handle,), arr)
end

"""
    try_get_shared(arr)

   Try to create a Julia array by sharing the data with the underlying :class:`NDArray`.

* NDArray arr: the array to be shared.

   .. warning::

      The returned array does not guarantee to share data with the underlying :class:`NDArray`.
      In particular, data sharing is possible only when the :class:`NDArray` lives on CPU.
"""
function try_get_shared(arr :: NDArray)
  if context(arr).device_type == CPU
    # try to do data sharing
    return unsafe_wrap(Array, pointer(arr), size(arr))
  else
    # impossible to share, just copying
    return copy(arr)
  end
end

"""
    is_shared(j_arr, arr)

   Test whether ``j_arr`` is sharing data with ``arr``.

* Array j_arr: the Julia Array.
* NDArray arr: the :class:`NDArray`.
"""
function is_shared(j_arr :: Array, arr :: NDArray)
  false
end
function is_shared{T<:DType}(j_arr :: Array{T}, arr :: NDArray)
  if length(j_arr) != length(arr)
    return false
  end
  if context(arr).device_type != CPU
    return false
  end
  return pointer(j_arr) == pointer(arr)
end

"""
    load(filename, ::Type{NDArray})

Load NDArrays from binary file.

* AbstractString filename: the path of the file to load. It could be S3 or HDFS address.
   :return: Either ``Dict{Base.Symbol, NDArray}`` or ``Vector{NDArray}``.

If the ``libmxnet`` is built with the corresponding component enabled. Examples
* ``s3://my-bucket/path/my-s3-ndarray``
* ``hdfs://my-bucket/path/my-hdfs-ndarray``
* ``/path-to/my-local-ndarray``
"""
function load(filename::AbstractString, ::Type{NDArray})
  out_size      = Ref{MX_uint}(0)
  out_hdrs      = Ref{Ptr{MX_handle}}(0)
  out_name_size = Ref{MX_uint}(0)
  out_names     = Ref{char_pp}(0)
  @mxcall(:MXNDArrayLoad, (char_p, Ref{MX_uint}, Ref{Ptr{MX_handle}}, Ref{MX_uint}, Ref{char_pp}),
          filename, out_size, out_hdrs, out_name_size, out_names)
  out_name_size = out_name_size[]
  out_size      = out_size[]
  if out_name_size == 0
    return [NDArray(MX_NDArrayHandle(hdr)) for hdr in unsafe_wrap(Array, out_hdrs[], out_size)]
  else
    @assert out_size == out_name_size
    return Dict([(Symbol(unsafe_wrap(String, k)), NDArray(MX_NDArrayHandle(hdr))) for (k,hdr) in
                 zip(unsafe_wrap(Array, out_names[], out_size), unsafe_wrap(Array, out_hdrs[], out_size))])
  end
end

"""
    save(filename :: AbstractString, data)

Save NDarrays to binary file. Filename could be S3 or HDFS address, if ``libmxnet`` is built
with corresponding support.

* AbstractString filename: path to the binary file to write to.
* data: data to save to file.
   :type data: :class:`NDArray`, or a ``Vector{NDArray}`` or a ``Dict{Base.Symbol, NDArray}``.
"""
function save(filename::AbstractString, data::NDArray)
  save(filename, [data])
end
function save(filename::AbstractString, data::Vector{NDArray})
  @mxcall(:MXNDArraySave, (char_p, MX_uint, Ptr{MX_handle}, char_pp),
          filename, length(data), MX_handle[data...], char_pp(0))
end
function save(filename::AbstractString, data::Dict{Base.Symbol,NDArray})
  names  = [k for k in keys(data)]
  arrays = MX_handle[data[k] for k in names]
  names  = AbstractString[string(k) for k in names]

  @mxcall(:MXNDArraySave, (char_p, MX_uint, Ptr{MX_handle}, char_pp),
          filename, length(names), arrays, names)
end

################################################################################
# NDArray functions dynamically imported from libmxnet
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

# Import corresponding math functions from base so the automatically defined libmxnet
# functions can overload them
import Base: sqrt

"""
The libxmnet APIs are automatically imported from ``libmxnet.so``. The functions listed
here operate on :class:`NDArray` objects. The arguments to the functions are typically ordered
as

.. code-block:: julia

   func_name(arg_in1, arg_in2, ..., scalar1, scalar2, ..., arg_out1, arg_out2, ...)

unless ``NDARRAY_ARG_BEFORE_SCALAR`` is not set. In this case, the scalars are put before the input arguments:

.. code-block:: julia

   func_name(scalar1, scalar2, ..., arg_in1, arg_in2, ..., arg_out1, arg_out2, ...)


If ``ACCEPT_EMPTY_MUTATE_TARGET`` is set. An overloaded function without the output arguments will also be defined:

.. code-block:: julia

   func_name(arg_in1, arg_in2, ..., scalar1, scalar2, ...)

Upon calling, the output arguments will be automatically initialized with empty NDArrays.

Those functions always return the output arguments. If there is only one output (the typical situation), that
object (:class:`NDArray`) is returned. Otherwise, a tuple containing all the outputs will be returned.
"""

function _get_ndarray_functions()
  n = Ref{MX_uint}(0)
  handles = Ref{Ptr{MX_handle}}(0)

  @mxcall(:MXListFunctions, (Ref{MX_uint}, Ref{Ptr{MX_handle}}), n, handles)

  funcs = unsafe_wrap(Array, handles[], n[])
  return funcs
end

const _function_cache = Dict{Symbol, MX_handle}()
function _get_function(name :: Symbol)
  if !haskey(_function_cache, name)
    handle = Ref{MX_handle}(0)

    @mxcall(:MXGetFunction, (Cstring, Ref{MX_handle}), name, handle)
    _function_cache[name] = handle[]
    return handle[]
  else
    return _function_cache[name]
  end
end

function _get_function_description(handle :: MX_handle)
  # get function information (human readable)
  ref_name = Ref{char_p}(0)
  ref_desc = Ref{char_p}(0)
  ref_narg = Ref{MX_uint}(0)

  ref_arg_names = Ref{char_pp}(0)
  ref_arg_types = Ref{char_pp}(0)
  ref_arg_descs = Ref{char_pp}(0)

  ref_ret_type  = Ref{char_p}(0)

  @mxcall(:MXFuncGetInfo,
         (MX_handle, Ref{char_p}, Ref{char_p}, Ref{MX_uint}, Ref{char_pp},
          Ref{char_pp}, Ref{char_pp}, Ref{char_p}),
          handle, ref_name, ref_desc, ref_narg, ref_arg_names,
          ref_arg_types, ref_arg_descs, ref_ret_type)

  name = Symbol(unsafe_wrap(String, ref_name[]))
  signature = _format_signature(Int(ref_narg[]), ref_arg_names)
  desc = "    " * string(name) * "(" * signature * ")\n\n"
  desc *= unsafe_wrap(String, ref_desc[]) * "\n\n"
  desc *= "# Arguments\n"
  desc *= _format_docstring(Int(ref_narg[]), ref_arg_names, ref_arg_types, ref_arg_descs)
  return name, desc
end

function _get_function_expressions(handle :: MX_handle, name)
  # get function specification
  ref_n_use_vars = Ref{MX_uint}(0)
  ref_n_scalars  = Ref{MX_uint}(0)
  ref_n_mut_vars = Ref{MX_uint}(0)
  ref_type_mask  = Ref{Cint}(0)
  @mxcall(:MXFuncDescribe,
         (MX_handle, Ref{MX_uint}, Ref{MX_uint}, Ref{MX_uint}, Ref{Cint}),
          handle, ref_n_use_vars, ref_n_scalars, ref_n_mut_vars, ref_type_mask)

  n_used_vars   = ref_n_use_vars[]
  n_scalars     = ref_n_scalars[]
  n_mutate_vars = ref_n_mut_vars[]
  type_mask     = ref_type_mask[]
  accept_empty_mutate = (type_mask & convert(Cint,ACCEPT_EMPTY_MUTATE_TARGET)) != 0
  arg_before_scalar   = (type_mask & convert(Cint,NDARRAY_ARG_BEFORE_SCALAR)) != 0

  # general ndarray function
  if arg_before_scalar
    args = vcat([Expr(:(::), Symbol("in$i"), NDArray) for i=1:n_used_vars],
                [Expr(:(::), Symbol("sca$i"), Real) for i=1:n_scalars],
                [Expr(:(::), Symbol("out$i"), NDArray) for i=1:n_mutate_vars])
  else
    args = vcat([Expr(:(::), Symbol("sca$i"), Real) for i=1:n_scalars],
                [Expr(:(::), Symbol("in$i"), NDArray) for i=1:n_used_vars],
                [Expr(:(::), Symbol("out$i"), NDArray) for i=1:n_mutate_vars])
  end

  _use_vars = Expr(:ref, :MX_handle, [Symbol("in$i") for i=1:n_used_vars]...)
  _scalars  = Expr(:ref, :MX_float, [Symbol("sca$i") for i=1:n_scalars]...)
  _mut_vars = Expr(:ref, :MX_handle, [Symbol("out$i") for i=1:n_mutate_vars]...)

  # XXX: hacky way of solving the problem that the arguments of `dot` should be swapped
  # See https://github.com/dmlc/MXNet.jl/issues/55
  if name == :dot
    _use_vars.args[2:end] = flipdim(_use_vars.args[2:end], 1)
  end
  stmt_call = quote
    local handle = _get_function($(QuoteNode(name)))
    _invoke_mxfunction(handle, $_use_vars, $_scalars, $_mut_vars)
  end
  if n_mutate_vars == 1
    stmt_ret = :(return out1)
  else
    stmt_ret = Expr(:return, Expr(:tuple, [Symbol("out$i") for i=1:n_mutate_vars]...))
  end

  func_body = Expr(:block, stmt_call, stmt_ret)
  func_head = Expr(:call, name, args...)

  func_def  = Expr(:function, func_head, func_body)
  exprs = Expr[func_def]

  if accept_empty_mutate
    args0      = args[1:n_used_vars+n_scalars]
    func_head0 = Expr(:call, name, args0...)
    _mut_vars0 = [:(NDArray(_ndarray_alloc())) for i=1:n_mutate_vars]
    stmt_call0 = Expr(:call, name, args0..., _mut_vars0...)
    func_body0 = Expr(:block, stmt_call0)
    func_head0 = Expr(:call, name, args0...)

    func_def0  = Expr(:function, func_head0, func_body0)
    push!(exprs, func_def0)
  end
  return exprs
end

macro _import_ndarray_functions()
  funcs = _get_ndarray_functions()
  func_exprs = Expr[]

  for i = 1:length(funcs)
    handle = funcs[i]

    name, desc = _get_function_description(handle)
    exprs = _get_function_expressions(handle, name)

    expr = quote
      $(exprs...)
      @doc $desc $name
    end

    push!(func_exprs, expr)
  end

  esc(quote
    $(func_exprs...)
  end)
end

@_import_ndarray_functions()
