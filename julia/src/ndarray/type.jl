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
function _ndarray_alloc(shape::NTuple{N,Int}, ctx::Context, delay_alloc::Bool) where N
  h_ref  = Ref{MX_handle}(0)
  shape  = collect(reverse(MX_uint.(shape)))
  @mxcall(:MXNDArrayCreate, (Ptr{MX_uint}, MX_uint, Cint, Cint, Cint, Ref{MX_handle}),
      shape, N, ctx.device_type, ctx.device_id, delay_alloc, h_ref)
  handle = MX_NDArrayHandle(h_ref[])
  return handle
end

# create a NDArray handle of specific shape type
function _ndarray_alloc(::Type{T}, shape::NTuple{N,Int}, ctx::Context, delay_alloc::Bool) where {T<:DType,N}
  h_ref  = Ref{MX_handle}(0)
  shape  = collect(reverse(MX_uint.(shape)))
  dtype  = toTypeFlag(T)
  @mxcall(:MXNDArrayCreateEx, (Ptr{MX_uint}, MX_uint, Cint, Cint, Cint, Cint, Ref{MX_handle}),
      shape, N, ctx.device_type, ctx.device_id, delay_alloc, dtype, h_ref)
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

  NDArray{T,N}(handle::MX_NDArrayHandle, writable::Bool = true) where {T,N} =
    new(handle, writable)
end

# UndefInitializer constructors
NDArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer};
             writable = true, ctx::Context = current_context()) where {T,N} =
  NDArray{T,N}(_ndarray_alloc(T, dims, ctx, false), writable)
NDArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}; kw...) where {T,N} =
  NDArray{T,N}(undef, dims; kw...)

NDArray{T}(::UndefInitializer, dims::NTuple{N,Integer}; kw...) where {T,N} =
  NDArray{T,N}(undef, dims; kw...)
NDArray{T}(::UndefInitializer, dims::Vararg{Integer,N}; kw...) where {T,N} =
  NDArray{T,N}(undef, dims; kw...)

NDArray(::UndefInitializer, dims::NTuple{N,Integer}; kw...) where {N} =
  NDArray{DEFAULT_DTYPE,N}(undef, dims; kw...)
NDArray(::UndefInitializer, dims::Vararg{Integer,N}; kw...) where {N} =
  NDArray{DEFAULT_DTYPE,N}(undef, dims; kw...)

NDArray(x::AbstractArray{<:DType}) = copy(collect(x), cpu())
NDArray(x::Array{<:DType})         = copy(x, cpu())

NDArray(::Type{T}, x::AbstractArray) where {T<:DType} =
  copy(convert(AbstractArray{T}, x), cpu())

NDArray(handle, writable = true) =
  NDArray{eltype(handle), ndims(handle)}(handle, writable)

# type aliases
const NDArrayOrReal = Union{NDArray,Real}
const VecOfNDArray = AbstractVector{<:NDArray}

Base.unsafe_convert(::Type{MX_handle}, x::NDArray) =
  Base.unsafe_convert(MX_handle, x.handle)
Base.convert(T::Type{MX_handle}, x::NDArray) = Base.unsafe_convert(T, x)
Base.cconvert(T::Type{MX_handle}, x::NDArray) = Base.unsafe_convert(T, x)

MX_handle(x::NDArray) = Base.convert(MX_handle, x)
