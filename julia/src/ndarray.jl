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

# All the types supported by mshadow. See `mshadow/base.h`
const DType = Union{Float32, Float64, Float16, UInt8, Int32, Int8, Int64}
@enum TypeFlag kFloat32 kFloat64 kFloat16 kUint8 kInt32 kInt8 kInt64
const DEFAULT_DTYPE = Float32  # MSHADOW_DEFAULT_DTYPE

function toTypeFlag(T::Type{<:DType})
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
  elseif T == Int8
    return kInt8
  elseif T == Int64
    return kInt64
  else
    throw(ArgumentError("Can't convert $T to DType."))
  end
end

function fromTypeFlag(T::TypeFlag)
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
  elseif T == kInt8
    return Int8
  elseif T == kInt64
    return Int64
  else
    throw(ArgumentError("Can't convert DType $T."))
  end
end

# create a NDArray handle of specific shape
function _ndarray_alloc(shape :: NTuple{N, Int}, ctx :: Context, delay_alloc :: Bool) where N
  h_ref  = Ref{MX_handle}(0)
  shape  = flipdim(MX_uint[shape...],1)
  @mxcall(:MXNDArrayCreate, (Ptr{MX_uint}, MX_uint, Cint, Cint, Cint, Ref{MX_handle}),
      shape, length(shape), ctx.device_type, ctx.device_id, delay_alloc, h_ref)
  handle = MX_NDArrayHandle(h_ref[])
  return handle
end

# create a NDArray handle of specific shape type
function _ndarray_alloc(:: Type{T}, shape :: NTuple{N, Int}, ctx :: Context, delay_alloc :: Bool) where {T <: DType,N}
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
    NDArray{T,N}

Wrapper of the `NDArray` type in `libmxnet`. This is the basic building block
of tensor-based computation.

!!! note
      since C/C++ use row-major ordering for arrays while Julia follows a
      column-major ordering. To keep things consistent, we keep the underlying data
      in their original layout, but use *language-native* convention when we talk
      about shapes. For example, a mini-batch of 100 MNIST images is a tensor of
      C/C++/Python shape (100,1,28,28), while in Julia, the same piece of memory
      have shape (28,28,1,100).
"""
mutable struct NDArray{T,N}
  handle   :: MX_NDArrayHandle
  writable :: Bool

  NDArray{T,N}(handle, writable = true) where {T,N} = new(handle, writable)
end

NDArray(x::AbstractArray{T}) where {T<:DType} = copy(collect(x), cpu())
NDArray(x::Array{T}) where {T<:DType} = copy(x, cpu())
NDArray(::Type{T}, x::AbstractArray) where {T<:DType} =
  copy(convert(AbstractArray{T}, x), cpu())
NDArray(handle, writable = true) =
  NDArray{eltype(handle), ndims(handle)}(handle, writable)

# type aliases
const NDArrayOrReal = Union{NDArray, Real}
const VecOfNDArray = AbstractVector{<:NDArray}

@unfuse NDArray

function Base.show(io::IO, x::NDArray)
  print(io, "NDArray ")
  Base.showarray(io, try_get_shared(x, sync = :read), header = false)
end

# for REPL
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, x::NDArray{T, N}) where {T, N}
  type_ = split(string(typeof(x)), '.', limit=2)[end]
  size_ = N == 1 ? "$(length(x))-element" : join(size(x), "×")
  println(io, "$size_ $type_ @ $(context(x)):")
  Base.showarray(io, try_get_shared(x, sync = :read), false, header = false)
end

Base.unsafe_convert(::Type{MX_handle}, obj::NDArray) =
  Base.unsafe_convert(MX_handle, obj.handle)
Base.convert(T::Type{MX_handle}, obj::NDArray) = Base.unsafe_convert(T, obj)
Base.cconvert(T::Type{MX_handle}, obj::NDArray) = Base.unsafe_convert(T, obj)

################################################################################
# NDArray functions exported to the users
################################################################################
"""
    context(arr::NDArray)

Get the context that this `NDArray` lives on.
"""
function context(arr::NDArray)
  ref_typeid = Ref{Cint}(0)
  ref_devid  = Ref{Cint}(0)
  @mxcall(:MXNDArrayGetContext, (MX_handle, Ref{Cint}, Ref{Cint}),
          arr, ref_typeid, ref_devid)
  return Context(ref_typeid[], ref_devid[])
end

"""
    empty(DType, dims[, ctx::Context = cpu()])
    empty(DType, dims)
    empty(DType, dim1, dim2, ...)

Allocate memory for an uninitialized `NDArray` with a specified type.
"""
empty(::Type{T}, dims::NTuple{N,Int}, ctx::Context = cpu()) where {N,T<:DType} =
  NDArray{T, N}(_ndarray_alloc(T, dims, ctx, false))
empty(::Type{T}, dims::Int...) where {T<:DType} = empty(T, dims)

"""
    empty(dims::Tuple[, ctx::Context = cpu()])
    empty(dim1, dim2, ...)

Allocate memory for an uninitialized `NDArray` with specific shape of type Float32.
"""
empty(dims::NTuple{N,Int}, ctx::Context = cpu()) where N =
  NDArray(_ndarray_alloc(dims, ctx, false))
empty(dims::Int...) = empty(dims)

"""
    similar(x::NDArray)

Create an `NDArray` with similar shape, data type,
and context with the given one.
Note that the returned `NDArray` is uninitialized.
"""
Base.similar(x::NDArray{T}) where {T} = empty(T, size(x), context(x))

"""
    zeros([DType], dims, [ctx::Context = cpu()])
    zeros([DType], dims...)
    zeros(x::NDArray)

Create zero-ed `NDArray` with specific shape and type.
"""
function zeros(::Type{T}, dims::NTuple{N,Int}, ctx::Context = cpu()) where {N,T<:DType}
  arr = empty(T, dims, ctx)
  arr[:] = zero(T)
  arr
end

zeros(::Type{T}, dims::Int...) where {T<:DType} = zeros(T, dims)

zeros(dims::NTuple{N,Int}, ctx::Context = cpu()) where N =
  zeros(MX_float, dims, ctx)
zeros(dims::Int...) = zeros(dims)

zeros(x::NDArray)::typeof(x)      = zeros_like(x)
Base.zeros(x::NDArray)::typeof(x) = zeros_like(x)

"""
    ones([DType], dims, [ctx::Context = cpu()])
    ones([DType], dims...)
    ones(x::NDArray)

Create an `NDArray` with specific shape & type, and initialize with 1.
"""
function ones(::Type{T}, dims::NTuple{N,Int}, ctx::Context = cpu()) where {N,T<:DType}
  arr = empty(T, dims, ctx)
  arr[:] = one(T)
  arr
end

ones(::Type{T}, dims::Int...) where T<:DType = ones(T, dims)

ones(dims::NTuple{N,Int}, ctx::Context = cpu()) where N =
  ones(MX_float, dims, ctx)
ones(dims::Int...) = ones(dims)

ones(x::NDArray)::typeof(x)      = ones_like(x)
Base.ones(x::NDArray)::typeof(x) = ones_like(x)

import Base: size, length, ndims, eltype

"""
    size(x::NDArray)
    size(x::NDArray, dims...)

Get the shape of an `NDArray`. The shape is in Julia's column-major convention.
See also the notes on NDArray shapes [`NDArray`](@ref).
"""
function size(x::NDArray)
  ref_ndim  = Ref{MX_uint}(0)
  ref_shape = Ref{Ptr{MX_uint}}(0)
  @mxcall(:MXNDArrayGetShape, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_uint}}),
          x, ref_ndim, ref_shape)
  tuple(map(Int, flipdim(unsafe_wrap(Array, ref_shape[], ref_ndim[]),1))...)
end

function size(x::NDArray{T,N}, dim::Int) where {T,N}
  if dim > N
    1
  else
    size(x)[dim]
  end
end

size(x::NDArray, dims::Int...) = map(d -> size(x, d), dims)

"""
    length(x::NDArray)

Get the number of elements in an `NDArray`.
"""
length(x::NDArray) = prod(size(x))

"""
    ndims(x::NDArray)

Get the number of dimensions of an `NDArray`.
Is equivalent to `length(size(arr))`.
"""
ndims(x::NDArray) = ndims(x.handle)

function ndims(x::MX_NDArrayHandle)::Int
  ref_ndim  = Ref{MX_uint}(0)
  ref_shape = Ref{Ptr{MX_uint}}(0)
  @mxcall(:MXNDArrayGetShape, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_uint}}),
          x, ref_ndim, ref_shape)
  ref_ndim[]
end

"""
    eltype(x::NDArray)

Get the element type of an `NDArray`.
"""
function eltype(x::Union{NDArray, MX_NDArrayHandle})
  dtype_ref = Ref{Cint}(0)
  @mxcall(:MXNDArrayGetDType, (MX_handle, Ptr{Cint}), x, dtype_ref)

  if dtype_ref[] == -1 # x->is_none()
    warn("Eltype of $x is not defined")
    Base.show_backtrace(STDOUT, backtrace())
    println()
    Float32
  else
    fromTypeFlag(TypeFlag(dtype_ref[]))
  end
end

@inline _first(x::NDArray) = try_get_shared(x, sync = :read) |> first

Base.first(x::NDArray) = _first(x)

Base.endof(x::NDArray) = length(x)

"""
    slice(arr :: NDArray, start:stop)

Create a view into a sub-slice of an `NDArray`. Note only slicing at the slowest
changing dimension is supported. In Julia's column-major perspective, this is the last
dimension. For example, given an `NDArray` of shape (2,3,4), `slice(array, 2:3)` will create
a `NDArray` of shape (2,3,2), sharing the data with the original array. This operation is
used in data parallelization to split mini-batch into sub-batches for different devices.
"""
function slice(arr::NDArray, ::Colon)
  arr
end
function slice(arr::NDArray, slice::UnitRange{Int})
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

function _at(handle::Union{MX_NDArrayHandle, MX_handle}, idx::Integer)
  h_ref = Ref{MX_handle}(C_NULL)
  @mxcall(:MXNDArrayAt, (MX_handle, MX_uint, Ref{MX_handle}),
          handle, idx, h_ref)
  h_ref[]
end

import Base: setindex!

"""
    setindex!(arr::NDArray, val, idx)

Assign values to an `NDArray`.
The following scenarios are supported

* single value assignment via linear indexing: `arr[42] = 24`

* `arr[:] = val`: whole array assignment, `val` could be a scalar or an array (Julia `Array`
  or `NDArray`) of the same shape.
* `arr[start:stop] = val`: assignment to a *slice*, `val` could be a scalar or an array of
  the same shape to the slice. See also [`slice`](@ref).
"""
function setindex!(arr::NDArray, val::Real, idx::Integer)
  # linear indexing
  @assert arr.writable
  _set_value(out=arr[idx], src=val)
end

function setindex!(arr::NDArray, val::Real, ::Colon)
  @assert arr.writable
  _set_value(out = arr, src = dump_mx_param(val))
end

function setindex!(arr::NDArray, val::Array{T}, ::Colon) where T<:Real
  @assert arr.writable
  copy!(arr, val)
end

function setindex!(arr::NDArray, val::NDArray, ::Colon)
  @assert arr.writable
  copy!(arr, val)
end

function setindex!(arr::NDArray, val::Union{T,Array{T},NDArray},
                   idx::UnitRange{Int}) where T<:Real
  @assert arr.writable
  setindex!(slice(arr, idx), val, Colon())
end

import Base: getindex
"""
    getindex(arr::NDArray, idx)

Shortcut for [`slice`](@ref). A typical use is to write

```julia
  arr[:] += 5
```

which translates into

```julia
  arr[:] = arr[:] + 5
```

which furthur translates into

```julia
  setindex!(getindex(arr, Colon()), 5, Colon())
```

!!! note
    The behavior is quite different from indexing into Julia's `Array`. For example, `arr[2:5]`
    create a **copy** of the sub-array for Julia `Array`, while for `NDArray`, this is
    a *slice* that shares the memory.
"""
getindex(arr::NDArray, ::Colon) = arr

"""
Shortcut for [`slice`](@ref).
**NOTE** the behavior for Julia's built-in index slicing is to create a
copy of the sub-array, while here we simply call `slice`,
which shares the underlying memory.
"""
getindex(arr::NDArray, idx::UnitRange{Int}) = slice(arr, idx)

getindex(arr::NDArray) = _first(arr)

function getindex(arr::NDArray, idx::Integer)
  # linear indexing
  len = length(arr)
  size_ = size(arr)

  if idx <= 0 || idx > len
    throw(BoundsError(
      "attempt to access $(join(size_, 'x')) NDArray at index $(idx)"))
  end

  idx -= 1
  offsets = size_[1:end-1] |> reverse ∘ cumprod ∘ collect
  handle = arr.handle
  for offset ∈ offsets
    handle = _at(handle, idx ÷ offset)
    idx %= offset
  end

  _at(handle, idx) |> MX_NDArrayHandle |> x -> NDArray(x, arr.writable)
end

import Base: copy!, copy, convert, deepcopy

"""
    copy!(dst::Union{NDArray, Array}, src::Union{NDArray, Array})

Copy contents of `src` into `dst`.
"""
function copy!(dst::NDArray, src::NDArray)
  @assert(dst.writable)
  if dst.handle == src.handle
    warn("Copying an NDArray to itself")
    return
  end

  _copyto(src, out=dst)
  return dst
end

function copy!(dst::Array{T}, src::NDArray{T}) where T<:DType
  @assert size(dst) == size(src)
  @mxcall(:MXNDArraySyncCopyToCPU, (MX_handle, Ptr{Void}, Csize_t),
          src, pointer(dst), length(dst))
  dst
end

copy!(dst::Array{<:Real}, src::NDArray) = copy!(dst, copy(src))
copy!(dst::NDArray, src::AbstractArray) = copy!(dst, collect(src))

function copy!(dst::NDArray{T}, src::Array{<:Real}) where {T}
  @assert dst.writable
  @assert size(dst) == size(src)
  src = convert(Array{T}, src) # this might involve copying
  @mxcall(:MXNDArraySyncCopyFromCPU, (MX_handle, Ptr{Void}, Csize_t),
          dst.handle, pointer(src), length(src))
  dst
end

function copy_ignore_shape!(dst::NDArray{T}, src::Array{<:Real}) where {T}
  @assert dst.writable
  @assert length(dst) == length(src)
  src = convert(Array{T}, src) # this might involve copying
  @mxcall(:MXNDArraySyncCopyFromCPU, (MX_handle, Ptr{Void}, Csize_t),
          dst.handle, pointer(src), length(src))
  dst
end


"""
    copy(arr :: NDArray)
    copy(arr :: NDArray, ctx :: Context)
    copy(arr :: Array, ctx :: Context)

Create a copy of an array. When no `Context` is given, create a Julia `Array`.
Otherwise, create an `NDArray` on the specified context.
"""
# Create copy: NDArray -> Julia Array
copy(x::NDArray{T,D}) where{T,D} = copy!(Array{T,D}(size(x)), x)

# Create copy: NDArray -> NDArray in a given context
copy(x::NDArray{T,D}, ctx::Context) where {T,D} =
  copy!(NDArray{T,D}(_ndarray_alloc(T, size(x), ctx, true)), x)

# Create copy: Julia Array -> NDArray in a given context
copy(x::Array{T}, ctx::Context) where {T<:DType} =
  copy!(empty(T, size(x), ctx), x)

copy(x::AbstractArray, ctx::Context) =
  copy!(empty(eltype(x), size(x), ctx), collect(x))

"""
    convert(::Type{Array{<:Real}}, x::NDArray)

Convert an `NDArray` into a Julia `Array` of specific type.
Data will be copied.
"""
convert(T::Type{Array{<:Real}}, x::NDArray) = convert(T, copy(x))

"""
    deepcopy(arr::NDArray)

Get a deep copy of the data blob in the form of an NDArray of default storage
type. This function blocks. Do not use it in performance critical code.
"""
function deepcopy(arr::NDArray)
  out_ref = Ref{MX_handle}(C_NULL)
  @mxcall(:MXNDArrayGetDataNDArray, (MX_handle, Ref{MX_handle}), arr, out_ref)
  NDArray(MX_NDArrayHandle(out_ref[]))
end

"""
    hcat(x::NDArray...)
"""
Base.hcat(xs::NDArray{T}...) where T = cat(2, xs...)

"""
    vcat(x::NDArray...)
"""
Base.vcat(xs::NDArray{T}...) where T = cat(1, xs...)

"""
    cat(dim, xs::NDArray...)

Concate the `NDArray`s which have the same element type along the `dim`.
Building a diagonal matrix is not supported yet.
"""
function Base.cat(dim::Int, xs::NDArray{T}...) where T
  ns = ndims.(xs)
  d = Base.max(dim, maximum(ns))
  xs′ = map(zip(ns, xs)) do i
    n, x = i
    (d > n) ? reshape(x, -2, Base.ones(Int, d - n)...) : x
  end
  concat(xs′..., dim = d - dim)
end

"""
    @inplace

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

which will do inplace adding of the contents of `b` into `a`.
"""
macro inplace(ex)
  f = if ex.head == :+= || ex.head == :.+=
    :add_to!
  elseif ex.head == :-= || ex.head == :.-=
    :sub_from!
  elseif ex.head == :.*=
    :mul_to!
  elseif ex.head == :./=
    :div_from!
  elseif ex.head == :.%=
    :mod_from!
  else
    error("unsupported inplace translation for $ex")
  end
  Expr(:call, f, esc(ex.args[1]), esc(ex.args[2]))
end

"""
    add_to!(dst::NDArray, args::NDArrayOrReal...)

Add a bunch of arguments into `dst`. Inplace updating.
"""
function add_to!(dst::NDArray, args::NDArrayOrReal...)
  @assert dst.writable
  for arg in args
    if isa(arg, Real)
      _plus_scalar(dst, scalar = arg, out = dst)
    else
      _plus!(dst, arg)
    end
  end
  dst
end

import Base: +

"""
    +(args...)
    .+(args...)

Summation. Multiple arguments of either scalar or `NDArray` could be
added together. Note at least the first or second argument needs to be an
`NDArray` to avoid ambiguity of built-in summation.
"""
+(x::NDArray)             = x
+(x::NDArray, y::NDArray) = _plus(x, y)
+(x::NDArray, y::Real)    = _plus_scalar(x, scalar = y)
+(y::Real,    x::NDArray) = _plus_scalar(x, scalar = y)

broadcast_(::typeof(+), x::NDArray, y::Real) = x + y
broadcast_(::typeof(+), x::Real, y::NDArray) = x + y

broadcast_(::typeof(+), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N}   = x + y
broadcast_(::typeof(+), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_add(x, y)

"""
    sub_from!(dst::NDArray, args::NDArrayOrReal...)

Subtract a bunch of arguments from `dst`. Inplace updating.
"""
function sub_from!(dst::NDArray, arg::NDArrayOrReal)
  @assert dst.writable
  if isa(arg, Real)
    _minus_scalar(dst, scalar = arg, out = dst)
  else
    _minus!(dst, arg)
  end
  dst
end

import Base: -

"""
    -(x::NDArray)
    -(x, y)
    .-(x, y)

Subtraction `x - y`, of scalar types or `NDArray`.
Or create the negative of `x`.
"""
-(x::NDArray) = _mul_scalar(x, scalar = -one(eltype(x)))
-(x::NDArray, y::NDArray) = _minus(x, y)
-(x::NDArray, y::Real)    = _minus_scalar(x, scalar = y)
-(y::Real, x::NDArray)    = _rminus_scalar(x, scalar = y)

broadcast_(::typeof(-), x::NDArray, y::Real) = x - y
broadcast_(::typeof(-), x::Real, y::NDArray) = x - y

broadcast_(::typeof(-), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N}   = x - y
broadcast_(::typeof(-), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_minus(x, y)

"""
    mul_to!(dst::NDArray, arg::NDArrayOrReal)

Elementwise multiplication into `dst` of either a scalar or an `NDArray` of the same shape.
Inplace updating.
"""
function mul_to!(dst::NDArray, arg::NDArrayOrReal)
  @assert dst.writable
  if isa(arg, Real)
    _mul_scalar(dst, scalar = arg, out = dst)
  else
    _mul(dst, arg, out = dst)
  end
  dst
end

import Base: *

"""
    .*(x, y)

Elementwise multiplication for `NDArray`.
"""
*(x::NDArray, y::Real)  = _mul_scalar(x, scalar = y)
*(y::Real, x::NDArray)  = _mul_scalar(x, scalar = y)

broadcast_(::typeof(*), x::NDArray, y::Real) = x * y
broadcast_(::typeof(*), y::Real, x::NDArray) = x * y

broadcast_(::typeof(*), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N} =
  _mul(x, y)
broadcast_(::typeof(*), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_mul(x, y)

"""
    *(A::NDArray, B::NDArray)

Matrix/tensor multiplication.
"""
*(x::NDArray{T}, y::NDArray{T}) where T = x ⋅ y

"""
    div_from!(dst::NDArray, arg::NDArrayOrReal)

Elementwise divide a scalar or an `NDArray` of the same shape from `dst`. Inplace updating.
"""
function div_from!(dst::NDArray, arg::NDArrayOrReal)
  @assert dst.writable
  if isa(arg, Real)
    _div_scalar(dst, scalar = arg, out = dst)
  else
    _div(dst, arg, out = dst)
  end
  dst
end

function div_from!(dst::NDArray{T}, arg::Real) where {T<:Integer}
  @assert dst.writable
  @assert(round(T, arg) != zero(T), "Integer divided by zero")
  _div_scalar(dst, scalar = arg, out = dst)
  dst
end

"""
    rdiv_from!(x:: Real, y::NDArray)

Elementwise divide a scalar by an `NDArray`. Inplace updating.
"""
function rdiv_from!(x::Real, y::NDArray)
  @assert y.writable
  _rdiv_scalar(y, scalar = x, out = y)
  y
end

import Base: /

"""
    ./(x::NDArray, y::NDArray)
    ./(x::NDArray, y::Real)
    ./(x::Real, y::NDArray)

* Elementwise dividing an `NDArray` by a scalar or another `NDArray`
of the same shape.

* Elementwise divide a scalar by an `NDArray`.

* Matrix division (solving linear systems) is not implemented yet.
"""
/(x::NDArray, y::Real) = _div_scalar(x, scalar = y)

broadcast_(::typeof(/), x::NDArray, y::Real)    = _div_scalar(x, scalar = y)
broadcast_(::typeof(/), y::Real, x::NDArray)    = _rdiv_scalar(x, scalar = y)

broadcast_(::typeof(/), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N} =
  _div(x, y)
broadcast_(::typeof(/), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_div(x, y)

function broadcast_(::typeof(/), x::NDArray{T}, y::Real) where {T<:Integer}
  @assert(round(T, y) != zero(T), "Integer divided by zero")
  _div_scalar(x, scalar = y)
end

"""
    mod_from!(x::NDArray, y::NDArray)
    mod_from!(x::NDArray, y::Real)

Elementwise modulo for `NDArray`.
Inplace updating.
"""
mod_from!(x::NDArray, y::NDArray) = _mod!(x, y)
mod_from!(x::NDArray, y::Real)    = _mod_scalar!(x, y)

"""
    rmod_from!(y::Real, x::NDArray)

Elementwise modulo for `NDArray`.
Inplace updating.
"""
rmod_from!(y::Real, x::NDArray) = _rmod_scalar!(x, y)

import Base: %

"""
    .%(x::NDArray, y::NDArray)
    .%(x::NDArray, y::Real)
    .%(x::Real, y::NDArray)

Elementwise modulo for `NDArray`.
"""
%(x::NDArray, y::Real) = _mod_scalar(x, scalar = y)

broadcast_(::typeof(%), x::NDArray, y::Real)    = _mod_scalar(x, y)
broadcast_(::typeof(%), y::Real, x::NDArray)    = _rmod_scalar(x, y)

broadcast_(::typeof(%), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N} =
  _mod(x, y)
broadcast_(::typeof(%), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_mod(x, y)

import Base: ^

# document of `.^` is merged into SymbolicNode's

broadcast_(::typeof(^), x::NDArray, s::Real)    = _power_scalar(x, scalar = s)
broadcast_(::typeof(^), s::Real, x::NDArray)    = _rpower_scalar(x, scalar = s)

broadcast_(::typeof(^), ::Irrational{:e}, x::NDArray) = exp(x)
broadcast_(::typeof(^), x::NDArray, s::Irrational)    = _power_scalar(x, scalar = s)
broadcast_(::typeof(^), s::Irrational, x::NDArray)    = _rpower_scalar(x, scalar = s)

broadcast_(::typeof(^), x::NDArray{T,N}, y::NDArray{T,N}) where {T,N} =
  _power(x, y)
broadcast_(::typeof(^), x::NDArray{T,N}, y::NDArray{T,M}) where {T,N,M} =
  _broadcast_power(x, y)

###############################################################################
# comparison
###############################################################################

broadcast_(::typeof(==), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_equal(x, y)

broadcast_(::typeof(!=), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_not_equal(x, y)

broadcast_(::typeof(>), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_greater(x, y)

broadcast_(::typeof(>=), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_greater_equal(x, y)

broadcast_(::typeof(<), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_lesser(x, y)

broadcast_(::typeof(<=), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_lesser_equal(x, y)


###############################################################################
# min/max
###############################################################################

import Base: min, max

broadcast_(::typeof(max), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_maximum(x, y)

broadcast_(::typeof(min), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_minimum(x, y)

"""
    fill!(arr::NDArray, x)

Create an `NDArray` filled with the value `x`, like `Base.fill!`.
"""
function Base.fill!(arr::NDArray, x)
  arr[:] = x
  arr
end

"""
    fill(x, dims, ctx=cpu())
    fill(x, dims...)

Create an `NDArray` filled with the value `x`, like `Base.fill`.
"""
function fill(x, dims::NTuple{N,Integer}, ctx::Context=cpu()) where N
  arr = empty(typeof(x), dims, ctx)
  arr[:] = x
  arr
end

fill(x, dims::Integer...) = fill(x, dims)

import Base: hypot

broadcast_(::typeof(hypot), x::NDArray{T}, y::NDArray{T}) where {T} =
  _broadcast_hypot(x, y)

"""
Manipulating as Julia Arrays
----------------------------

    @nd_as_jl(captures..., statement)

A convenient macro that allows to operate `NDArray` as Julia Arrays. For example,

```julia
  x = mx.zeros(3,4)
  y = mx.ones(3,4)
  z = mx.zeros((3,4), mx.gpu())

  @mx.nd_as_jl ro=(x,y) rw=z begin
    # now x, y, z are just ordinary Julia Arrays
    z[:,1] = y[:,2]
    z[:,2] = 5
  end
```

Under the hood, the macro convert all the declared captures from `NDArray` into Julia
Arrays, by using `try_get_shared`. And automatically commit the modifications back into
the `NDArray` that is declared as `rw`. This is useful for fast prototyping and when
implement non-critical computations, such as `AbstractEvalMetric`.

!!! note
* Multiple `rw` and / or `ro` capture declaration could be made.
* The macro does **not** check to make sure that `ro` captures are not modified. If the
  original `NDArray` lives in CPU memory, then it is very likely the corresponding
  Julia Array shares data with the `NDArray`, so modifying the Julia Array will also
  modify the underlying `NDArray`.
* More importantly, since the `NDArray` is
  asynchronized, we will wait for *writing* for `rw` variables but wait only for *reading*
  in `ro` variables. If we write into those `ro` variables, **and** if the memory is
  shared, racing condition might happen, and the behavior is undefined.
* When an `NDArray` is declared to be captured as `rw`, its contents is always sync
  back in the end.
* The execution results of the expanded macro is always `nothing`.
* The statements are wrapped in a `let`, thus locally introduced new variables will not be
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
  pdata = Ref{Ptr{Void}}(0)
  @mxcall(:MXNDArrayGetData, (MX_handle, Ref{Ptr{Void}}), arr, pdata)
  return convert(Ptr{eltype(arr)}, pdata[])
end

@inline _wait_to_read(arr :: NDArray) =
  @mxcall(:MXNDArrayWaitToRead, (MX_handle,), arr)
@inline _wait_to_write(arr :: NDArray) =
  @mxcall(:MXNDArrayWaitToWrite, (MX_handle,), arr)

"""
    try_get_shared(arr; sync=:nop)

Try to create a Julia array by sharing the data with the underlying `NDArray`.

# Arguments:

* `arr::NDArray`: the array to be shared.

!!! note
    The returned array does not guarantee to share data with the underlying `NDArray`.
    In particular, data sharing is possible only when the `NDArray` lives on CPU.

* `sync::Symbol`: `:nop`,`:write`, `:read`
  On CPU, invoke `_wait_to_read` if `:read`;
  invoke `_wait_to_write` if `:write`.
"""
function try_get_shared(x::NDArray; sync::Symbol=:nop)
  if context(x).device_type == CPU
    # try to do data sharing
    if sync == :read
      _wait_to_read(x)
    elseif sync == :write
      _wait_to_write(x)
    end

    unsafe_wrap(Array, pointer(x), size(x))
  else
    # impossible to share, just copying
    copy(x)
  end
end

"""
    is_shared(j_arr, arr)

Test whether `j_arr` is sharing data with `arr`.

# Arguments:

* `j_arr::Array`: the Julia Array.
* `arr::NDArray`: the `NDArray`.
"""
is_shared(::Array, ::NDArray) = false

function is_shared(j_arr::Array{T}, arr::NDArray{T}) where {T<:DType}
  if length(j_arr) != length(arr)
    return false
  end
  if context(arr).device_type != CPU
    return false
  end
  pointer(j_arr) == pointer(arr)
end

"""
    load(filename, ::Type{NDArray})

Load NDArrays from binary file.

# Arguments:
* `filename::String`: the path of the file to load. It could be S3 or HDFS address.

Returns either `Dict{Symbol, NDArray}` or `Vector{NDArray}`.

`filename` can point to `s3` or `hdfs` resources if the `libmxnet` is built with the
corresponding components enabled. Examples:
* `s3://my-bucket/path/my-s3-ndarray`
* `hdfs://my-bucket/path/my-hdfs-ndarray`
* `/path-to/my-local-ndarray`
"""
function load(filename::AbstractString, ::Type{<:NDArray})
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
    return Dict([(Symbol(unsafe_string(k)), NDArray(MX_NDArrayHandle(hdr))) for (k,hdr) in
                 zip(unsafe_wrap(Array, out_names[], out_size), unsafe_wrap(Array, out_hdrs[], out_size))])
  end
end

"""
    save(filename::AbstractString, data)

Save NDarrays to binary file. Filename could be S3 or HDFS address, if `libmxnet` is built
with corresponding support (see `load`).

* `filename::String`: path to the binary file to write to.
* `data`: data to save to file. Data can be a`NDArray`, a `Vector` of `NDArray`,
  or a `Dict{Symbol}` contains `NDArray`s.
"""
save(filename::String, data::NDArray) = save(filename, [data])

save(filename::String, data::VecOfNDArray) =
  @mxcall(:MXNDArraySave, (char_p, MX_uint, Ptr{MX_handle}, char_pp),
          filename, length(data), MX_handle[data...], char_pp(0))

function save(filename::String, data::Dict{Symbol})
  names  = keys(data)
  arrays = MX_handle.(collect(values(data)))
  names  = String.(collect(names))

  @mxcall(:MXNDArraySave, (char_p, MX_uint, Ptr{MX_handle}, char_pp),
          filename, length(names), arrays, names)
end

################################################################################
# Mapping NDArray functions to Base-like API
################################################################################

const _ndsig = Dict{Symbol,Expr}()
const _nddoc = Dict{Symbol,Any}()

function _autoimport(name::Symbol, sig::Expr)
  if name == :broadcast_
    name = _broadcast_target(sig)
  end

  if isdefined(Base, name)
    :(import Base: $name)
  else
    :()
  end
end

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
  if fname !== :broadcast_
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

macro _remap(sig::Expr, imp::Expr)
  fname = (sig.head == :call) ? sig.args[1] : sig.args[1].args[1]  # case of `where`
  opname = string(imp.args[1])

  import_expr = _autoimport(fname, sig)

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
    $import_expr
    @doc $docstr ->
    $func_def
  end)
end

macro _remap(sig::Expr, imp::Symbol)
  imp = _ndsig[imp]

  esc(quote
    @_remap($sig, $imp)
  end)
end

_ndsig[:reshape] = :(reshape(arr; shape = dim, reverse = !reverse))
@_remap reshape(arr::NDArray, dim...; reverse = false) reshape
@_remap reshape(arr::NDArray, dim; reverse = false)    reshape

@_remap mean(arr::NDArray)         mean(arr)
@_remap mean(arr::NDArray, region) mean(arr; axis = 0 .- region, keepdims = true)

@_remap sum(arr::NDArray)       sum(arr)
@_remap sum(arr::NDArray, dims) sum(arr; axis = 0 .- dims, keepdims = true)

@_remap maximum(arr::NDArray)       max(arr)
@_remap maximum(arr::NDArray, dims) max(arr; axis = 0 .- dims, keepdims = true)

@_remap minimum(arr::NDArray)       min(arr)
@_remap minimum(arr::NDArray, dims) min(arr; axis = 0 .- dims, keepdims = true)

# See https://github.com/dmlc/MXNet.jl/issues/55
@_remap dot(x::NDArray, y::NDArray) dot(y, x)

# See https://github.com/dmlc/MXNet.jl/pull/123
@_remap transpose(arr::NDArray{T,1}) where T reshape(arr; shape = (1, length(arr)), reverse = true)
@_remap transpose(arr::NDArray{T,2}) where T transpose(arr)
@_remap permutedims(arr::NDArray, axes) transpose(arr; axes = length(axes) .- tuple(axes...))

@_remap prod(arr::NDArray)       prod(arr)
@_remap prod(arr::NDArray, dims) prod(arr; axis = 0 .- dims, keepdims = true)

_nddoc[:clip] = _nddoc[:clip!] =
"""
    clip(x::NDArray, min, max)
    clip!(x::NDArray, min, max)

Clips (limits) the values in `NDArray`.
Given an interval, values outside the interval are clipped to the interval edges.
Clipping `x` between `min` and `x` would be:

```julia
clip(x, min_, max_) = max(min(x, max_), min_))
```

```jldoctest
julia> x = NDArray(1:9);

julia> mx.clip(x, 2, 8)'
1×9 mx.NDArray{Int64,2} @ CPU0:
 2  2  3  4  5  6  7  8  8
```

The storage type of clip output depends on storage types of inputs and the
`min`, `max` parameter values:

- clip(default) = default
- clip(row_sparse, min <= 0, max >= 0) = row_sparse
- clip(csr, min <= 0, max >= 0) = csr
- clip(row_sparse, min < 0, max < 0) = default
- clip(row_sparse, min > 0, max > 0) = default
- clip(csr, min < 0, max < 0) = csr
- clip(csr, min > 0, max > 0) = csr
"""
@_remap clip(x::NDArray, min::Real, max::Real) clip(x; a_min = min, a_max = max)
@_remap clip!(x::NDArray, min::Real, max::Real) clip(x; a_min = min, a_max = max)

_nddoc[:expand_dims] =
"""
    expand_dims(x::NDArray, dim)

Insert a new axis into `dim`.

```julia
julia> x
4 mx.NDArray{Float64,1} @ CPU0:
 1.0
 2.0
 3.0
 4.0

julia> mx.expand_dims(x, 1)
1×4 mx.NDArray{Float64,2} @ CPU0:
 1.0  2.0  3.0  4.0

julia> mx.expand_dims(x, 2)
4×1 mx.NDArray{Float64,2} @ CPU0:
 1.0
 2.0
 3.0
 4.0
```
"""
@_remap expand_dims(x::NDArray, dim) expand_dims(x; axis = -dim)

# trigonometric functions, remap to keep consistent with Base
@_remap broadcast_(::typeof(sin),  x::NDArray) sin(x)
@_remap broadcast_(::typeof(cos),  x::NDArray) cos(x)
@_remap broadcast_(::typeof(tan),  x::NDArray) tan(x)
@_remap broadcast_(::typeof(asin), x::NDArray) arcsin(x)
@_remap broadcast_(::typeof(acos), x::NDArray) arccos(x)
@_remap broadcast_(::typeof(atan), x::NDArray) arctan(x)

# hyperbolic funcs, remap to keep consistent with Base
@_remap broadcast_(::typeof(sinh),  x::NDArray) sinh(x)
@_remap broadcast_(::typeof(cosh),  x::NDArray) cosh(x)
@_remap broadcast_(::typeof(tanh),  x::NDArray) tanh(x)
@_remap broadcast_(::typeof(asinh), x::NDArray) arcsinh(x)
@_remap broadcast_(::typeof(acosh), x::NDArray) arccosh(x)
@_remap broadcast_(::typeof(atanh), x::NDArray) arctanh(x)

# activation functions
_nddoc[:σ] = _nddoc[:sigmoid] = doc"""
    σ.(x::NDArray)
    sigmoid.(x::NDArray)

Computes sigmoid of x element-wise.

```math
σ(x) = \frac{1}{(1 + exp(-x))}
```

The storage type of `sigmoid` output is always dense.
"""
@_remap broadcast_(::typeof(σ), x::NDArray)       sigmoid(x)
@_remap broadcast_(::typeof(sigmoid), x::NDArray) sigmoid(x)

_nddoc[:relu] = doc"""
    relu.(x::NDArray)

Computes rectified linear.

```math
\max(x, 0)
```
"""
@_remap broadcast_(::typeof(relu), x::NDArray) relu(x)

_nddoc[:softmax] = doc"""
    softmax.(x::NDArray, [dim = ndims(x)])

Applies the softmax function.

The resulting array contains elements in the range `(0, 1)`
and the elements along the given axis sum up to 1.

```math
softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
```
"""
@_remap broadcast_(::typeof(softmax), x::NDArray) softmax(x; axis = -ndims(x))
@_remap broadcast_(::typeof(softmax), x::NDArray, dim::Int) softmax(x; axis = -dim)

_nddoc[:log_softmax] = """
    log_softmax.(x::NDArray, [dim = ndims(x)])

Computes the log softmax of the input.
This is equivalent to computing softmax followed by log.

julia> x
2×3 mx.NDArray{Float64,2} @ CPU0:
 1.0  2.0  0.1
 0.1  2.0  1.0

julia> mx.log_softmax.(x)
2×3 mx.NDArray{Float64,2} @ CPU0:
 -1.41703  -0.41703  -2.31703
 -2.31703  -0.41703  -1.41703
"""
@_remap broadcast_(::typeof(log_softmax), x::NDArray) log_softmax(x; axis = -ndims(x))
@_remap broadcast_(::typeof(log_softmax), x::NDArray, dim::Int) log_softmax(x; axis = -dim)

################################################################################
# remapping to solving type unstablility
################################################################################

@_remap _plus(x::NDArray, y::NDArray)  _plus(x, y)
@_remap _plus!(x::NDArray, y::NDArray) _plus(x, y)

@_remap _minus(x::NDArray, y::NDArray)  _minus(x, y)
@_remap _minus!(x::NDArray, y::NDArray) _minus(x, y)

@_remap _mod(x::NDArray, y::NDArray)  _mod(x, y)
@_remap _mod!(x::NDArray, y::NDArray) _mod(x, y)

@_remap _mod_scalar(x::NDArray, y::Real)  _mod_scalar(x; scalar = y)
@_remap _mod_scalar!(x::NDArray, y::Real) _mod_scalar(x; scalar = y)

@_remap _rmod_scalar(x::NDArray, y::Real)  _rmod_scalar(x; scalar = y)
@_remap _rmod_scalar!(x::NDArray, y::Real) _rmod_scalar(x; scalar = y)

@_remap _broadcast_add(x::NDArray, y::NDArray)  broadcast_add(x, y)
@_remap _broadcast_add!(x::NDArray, y::NDArray) broadcast_add(x, y)

@_remap _broadcast_minus(x::NDArray, y::NDArray)  broadcast_minus(x, y)
@_remap _broadcast_minus!(x::NDArray, y::NDArray) broadcast_minus(x, y)

@_remap _broadcast_mul(x::NDArray, y::NDArray)  broadcast_mul(x, y)
@_remap _broadcast_mul!(x::NDArray, y::NDArray) broadcast_mul(x, y)

@_remap _broadcast_div(x::NDArray, y::NDArray)  broadcast_div(x, y)
@_remap _broadcast_div!(x::NDArray, y::NDArray) broadcast_div(x, y)

@_remap _broadcast_mod(x::NDArray, y::NDArray)  broadcast_mod(x, y)
@_remap _broadcast_mod!(x::NDArray, y::NDArray) broadcast_mod(x, y)

@_remap _broadcast_power(x::NDArray, y::NDArray)  broadcast_power(x, y)
@_remap _broadcast_power!(x::NDArray, y::NDArray) broadcast_power(x, y)

@_remap _broadcast_equal(x::NDArray, y::NDArray)  broadcast_equal(x, y)
@_remap _broadcast_equal!(x::NDArray, y::NDArray) broadcast_equal(x, y)

@_remap _broadcast_not_equal(x::NDArray, y::NDArray)  broadcast_not_equal(x, y)
@_remap _broadcast_not_equal!(x::NDArray, y::NDArray) broadcast_not_equal(x, y)

@_remap _broadcast_greater(x::NDArray, y::NDArray)  broadcast_greater(x, y)
@_remap _broadcast_greater!(x::NDArray, y::NDArray) broadcast_greater(x, y)

@_remap _broadcast_greater_equal(x::NDArray, y::NDArray)  broadcast_greater_equal(x, y)
@_remap _broadcast_greater_equal!(x::NDArray, y::NDArray) broadcast_greater_equal(x, y)

@_remap _broadcast_lesser(x::NDArray, y::NDArray)  broadcast_lesser(x, y)
@_remap _broadcast_lesser!(x::NDArray, y::NDArray) broadcast_lesser(x, y)

@_remap _broadcast_lesser_equal(x::NDArray, y::NDArray)  broadcast_lesser_equal(x, y)
@_remap _broadcast_lesser_equal!(x::NDArray, y::NDArray) broadcast_lesser_equal(x, y)

@_remap _broadcast_maximum(x::NDArray, y::NDArray)  broadcast_maximum(x, y)
@_remap _broadcast_maximum!(x::NDArray, y::NDArray) broadcast_maximum(x, y)

@_remap _broadcast_minimum(x::NDArray, y::NDArray)  broadcast_minimum(x, y)
@_remap _broadcast_minimum!(x::NDArray, y::NDArray) broadcast_minimum(x, y)

@_remap _broadcast_hypot(x::NDArray, y::NDArray)  broadcast_hypot(x, y)
@_remap _broadcast_hypot!(x::NDArray, y::NDArray) broadcast_hypot(x, y)

_nddoc[:broadcast_to] = """
    broadcast_to(x::NDArray, dims)
    broadcast_to(x::NDArray, dims...)

Broadcasts the input array to a new shape.

In the case of broacasting doesn't work out of box,
you can expand the NDArray first.

```jldoctest
julia> x = mx.ones(2, 3, 4);

julia> y = mx.ones(1, 1, 4);

julia> x .+ mx.broadcast_to(y, 2, 3, 4)
2×3×4 mx.NDArray{Float32,3} @ CPU0:
[:, :, 1] =
 2.0  2.0  2.0
 2.0  2.0  2.0

[:, :, 2] =
 2.0  2.0  2.0
 2.0  2.0  2.0

[:, :, 3] =
 2.0  2.0  2.0
 2.0  2.0  2.0

[:, :, 4] =
 2.0  2.0  2.0
 2.0  2.0  2.0
```
"""
@_remap broadcast_to(x::NDArray, dims)    broadcast_to(x; shape = dims)
@_remap broadcast_to(x::NDArray, dims...) broadcast_to(x; shape = dims)

_nddoc[:broadcast_axis] = _nddoc[:broadcast_axes] = """
    broadcast_axis(x::NDArray, dim, size)
    broadcast_axes(x::NDArray, dim, size)

Broadcasts the input array over particular axis(axes).
Parameter `dim` and `size` could be a scalar, a Tuple or an Array.

`broadcast_axes` is just an alias.

```jldoctest
julia> x
1×2×1 mx.NDArray{Int64,3} @ CPU0:
[:, :, 1] =
 1  2

julia> mx.broadcast_axis(x, 1, 2)
2×2×1 mx.NDArray{Int64,3} @ CPU0:
[:, :, 1] =
 1  2
 1  2

julia> mx.broadcast_axis(x, 3, 2)
1×2×2 mx.NDArray{Int64,3} @ CPU0:
[:, :, 1] =
 1  2

[:, :, 2] =
 1  2
```
"""
@_remap(broadcast_axis(x::NDArray, dim, size),
        broadcast_axis(x; axis = ndims(x) .- dim, size = size))
@_remap(broadcast_axes(x::NDArray, dim, size),
        broadcast_axes(x; axis = ndims(x) .- dim, size = size))

################################################################################
# NDArray functions dynamically imported from libmxnet
################################################################################
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
function _get_ndarray_function_def(name :: String)
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
]

macro _import_ndarray_functions()
  names = filter(n -> ∉(lowercase(n), _op_import_bl), _get_libmx_op_names())

  func_exprs = map(names) do name
    op_handle = _get_libmx_op_handle(name)

    desc, key_narg = _get_libmx_op_description(name, op_handle)
    func_def, func_def2 = _get_ndarray_function_def(name)

    func_name = Symbol(name)
    expr = quote
      # TODO the explicit exclusion of take will no longer be necessary when it is removed from Base
      $((isdefined(Base, func_name) && func_name ≠ :take) ? :(import Base.$func_name) : :())
      $func_def
      @doc $desc ->
      $func_def2
    end
  end

  esc(quote
    $(func_exprs...)
  end)
end

@_import_ndarray_functions()
