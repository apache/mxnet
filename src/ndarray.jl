#=doc
NDArray
=======
=#

# create a NDArray handle of specific shape
function _ndarray_alloc{N}(shape :: NTuple{N, Int}, ctx :: Context, delay_alloc :: Bool)
  h_ref  = Ref{MX_handle}(0)
  shape  = flipdim(MX_uint[shape...],1)
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
#=doc
.. class:: NDArray

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
=#
type NDArray
  handle   :: MX_NDArrayHandle
  writable :: Bool

  function NDArray(handle, writable=true)
    new(handle, writable)
  end
end

function Base.show(io :: IO, arr :: NDArray)
  print(io, "mx.NDArray$(size(arr))")
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
#=doc
.. function:: context(arr :: NDArray)

   Get the context that this :class:`NDArray` lives on.
=#
function context(arr :: NDArray)
  ref_typeid = Ref{Cint}(0)
  ref_devid  = Ref{Cint}(0)
  @mxcall(:MXNDArrayGetContext, (MX_handle, Ref{Cint}, Ref{Cint}),
          arr, ref_typeid, ref_devid)
  return Context(ref_typeid[], ref_devid[])
end

#=doc
.. function::
   empty(shape :: Tuple, ctx :: Context)
   empty(shape :: Tuple)
   empty(dim1, dim2, ...)

   Allocate memory for an uninitialized :class:`NDArray` with specific shape.
=#
function empty{N}(shape :: NTuple{N, Int})
  empty(shape, cpu())
end
function empty{N}(shape :: NTuple{N, Int}, ctx :: Context)
  NDArray(_ndarray_alloc(shape, ctx, false))
end
function empty(shape :: Int...)
  empty(shape)
end

#=doc
Interface functions similar to Julia Arrays
-------------------------------------------
=#

#=doc
.. function::
   zeros(shape :: Tuple, ctx :: Context)
   zeros(shape :: Tuple)
   zeros(dim1, dim2, ...)

   Create zero-ed :class:`NDArray` with specific shape.
=#
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

#=doc
.. function::
   ones(shape :: Tuple, ctx :: Context)
   ones(shape :: Tuple)
   ones(dim1, dim2, ...)

   Create an :class:`NDArray` with specific shape and initialize with 1.
=#
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

#=doc
.. function::
   size(arr :: NDArray)
   size(arr :: NDArray, dim :: Int)

   Get the shape of an :class:`NDArray`. The shape is in Julia's column-major convention. See
   also the :ref:`notes on NDArray shapes <ndarray-shape-note>`.
=#
function size(arr :: NDArray)
  ref_ndim  = Ref{MX_uint}(0)
  ref_shape = Ref{Ptr{MX_uint}}(0)
  @mxcall(:MXNDArrayGetShape, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_uint}}),
          arr, ref_ndim, ref_shape)
  tuple(map(Int, flipdim(pointer_to_array(ref_shape[], ref_ndim[]),1))...)
end
function size(arr :: NDArray, dim :: Int)
  size(arr)[dim]
end

#=doc
.. function:: length(arr :: NDArray)

   Get the number of elements in an :class:`NDArray`.
=#
function length(arr :: NDArray)
  prod(size(arr))
end

#=doc
.. function:: ndims(arr :: NDArray)

   Get the number of dimensions of an :class:`NDArray`. Is equivalent to ``length(size(arr))``.
=#
function ndims(arr :: NDArray)
  length(size(arr))
end

#=doc
.. function:: eltype(arr :: NDArray)

   Get the element type of an :class:`NDArray`. Currently the element type is always ``mx.MX_float``.
=#
function eltype(arr :: NDArray)
  MX_float
end


import Base: slice
"""`slice` create a view into a sub-slice of an `NDArray`. Note only slicing at the slowest
changing dimension is supported. In Julia's column-major perspective, this is the last
dimension. For example, given an `NDArray` of shape (2,3,4), `sub(array, 2:3)` will create
a `NDArray` of shape (2,3,2), sharing the data with the original array. This operation is
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
"Assign all elements of an NDArray to a scalar"
function setindex!(arr :: NDArray, val :: Real, ::Colon)
  @assert(arr.writable)
  _set_value(val, arr)
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
"""Shortcut for `slice`. **NOTE** the behavior for Julia's built-in index slicing is to create a
copy of the sub-array, while here we simply call `slice`, which shares the underlying memory.
"""
function getindex(arr :: NDArray, ::Colon)
  return arr
end
function getindex(arr :: NDArray, idx::UnitRange{Int})
  slice(arr, idx)
end

#------------------------------------------------------------
# Copying functions
#------------------------------------------------------------
import Base: copy!, copy, convert
"Copy data between NDArrays"
function copy!(dst :: NDArray, src :: NDArray)
  @assert(dst.writable)
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
  @assert dst.writable
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

"Convert copy: NDArray -> Julia Array"
function convert{T<:Real}(t::Type{Array{T}}, arr :: NDArray)
  convert(t, copy(arr))
end


#------------------------------------------------------------
# Basic arithmetics
#------------------------------------------------------------
"""
Julia does not support re-definiton of `+=` operator (like `__iadd__` in python),
When one write `a += b`, it gets translated to `a = a+b`. `a+b` will allocate new
memory for the results, and the newly allocated `NDArray` object is then assigned
back to a, while the original contents in a is discarded. This is very inefficient
when we want to do inplace update.

This macro is a simple utility to implement this behavior. Write

```julia
@mx.inplace a += b
```

will translate into

```julia
mx.add_to!(a, b)
```

which will do inplace adding of the contents of b into a.
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

function add_to!(dst :: NDArray, args :: Union{Real, NDArray}...)
  @assert dst.writable
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
  @assert dst.writable
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

function mul_to!(dst :: NDArray, arg :: Union{Real, NDArray})
  @assert dst.writable
  if isa(arg, Real)
    _mul_scalar(dst, arg, dst)
  else
    _mul(dst, arg, dst)
  end
  return dst
end
import Base: .*, *
function .*(arg0 :: NDArray, arg :: Union{Real, NDArray})
  ret = copy(arg0, context(arg0))
  mul_to!(ret, arg)
end
function .*(arg0 :: Real, arg :: NDArray)
  .*(arg, arg0)
end
# unlike *, we only allow type Real in arguments, because array-array * operator
# means matrix multiplication in Julia
function *(arg0 :: NDArray, arg :: Real)
  ret = copy(arg0, context(arg0))
  mul_to!(ret, arg)
end
function *(arg0 :: Real, arg :: NDArray)
  *(arg, arg0)
end

function div_from!(dst :: NDArray, arg :: Union{Real, NDArray})
  @assert dst.writable
  if isa(arg, Real)
    _div_scalar(dst, arg, dst)
  else
    _div(dst, arg, dst)
  end
end
import Base: ./, /
function ./(arg0 :: NDArray, arg :: Union{Real, NDArray})
  ret = copy(arg0, context(arg0))
  div_from!(ret, arg)
end
function /(arg0 :: NDArray, arg :: Real)
  ./(arg0, arg)
end

#------------------------------------------------------------
# IO
#------------------------------------------------------------
"""Load NDArrays from binary file.

**Parameters**:

* `filename`: the path of the file to load. It could be S3 or HDFS address
  if the `libmxnet` is built with the corresponding component enabled. Examples

  * `s3://my-bucket/path/my-s3-ndarray`
  * `hdfs://my-bucket/path/my-hdfs-ndarray`
  * `/path-to/my-local-ndarray`

**Returns**:

  Either `Dict{Base.Symbol, NDArray}` or `Vector{NDArray}`.
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
    return [NDArray(MX_NDArrayHandle(hdr)) for hdr in pointer_to_array(out_hdrs[], out_size)]
  else
    @assert out_size == out_name_size
    return Dict([(symbol(bytestring(k)), NDArray(MX_NDArrayHandle(hdr))) for (k,hdr) in
                 zip(pointer_to_array(out_names[], out_size), pointer_to_array(out_hdrs[], out_size))])
  end
end

"""Save NDarrays to binary file.

**Parameters**:

* `filename`: path to the binary file to write to.
* `data`: an `NDArray`, or a `Vector{NDArray}` or a `Dict{Base.Symbol, NDArray}`.
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
Import dynamic functions for NDArrays. The arguments to the functions are typically ordered
as

```julia
func_name(arg_in1, arg_in2, ..., scalar1, scalar2, ..., arg_out1, arg_out2, ...)
```

unless NDARRAY_ARG_BEFORE_SCALAR is not set. In this case, the scalars are put before the input arguments:

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

    # TODO: add doc string
    # eval(:(@doc($doc_str, $func_name)))
  end
end

