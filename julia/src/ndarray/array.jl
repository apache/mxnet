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

# Julia Array related interface

"""
    similar(x::NDArray; writable, ctx)

Create an `NDArray` with similar shape, data type,
and context with the given one.
Note that the returned `NDArray` is uninitialized.
"""
Base.similar(x::NDArray{T,N}; writable = x.writable, ctx = context(x)) where {T,N} =
  NDArray{T,N}(undef, size(x)...; writable = writable, ctx = ctx)

"""
    zeros([DType], dims, ctx::Context = current_context())
    zeros([DType], dims...)
    zeros(x::NDArray)

Create zero-ed `NDArray` with specific shape and type.
"""
function zeros(::Type{T}, dims::NTuple{N,Int},
               ctx::Context = current_context()) where {N,T<:DType}
  x = NDArray{T}(undef, dims..., ctx = ctx)
  x[:] = zero(T)
  x
end

zeros(::Type{T}, dims::Int...) where {T<:DType} = zeros(T, dims)

zeros(dims::NTuple{N,Int}, ctx::Context = current_context()) where N =
  zeros(MX_float, dims, ctx)
zeros(dims::Int...) = zeros(dims)

zeros(x::NDArray)::typeof(x)      = zeros_like(x)
Base.zeros(x::NDArray)::typeof(x) = zeros_like(x)

"""
    ones([DType], dims, ctx::Context = current_context())
    ones([DType], dims...)
    ones(x::NDArray)

Create an `NDArray` with specific shape & type, and initialize with 1.
"""
function ones(::Type{T}, dims::NTuple{N,Int},
              ctx::Context = current_context()) where {N,T<:DType}
  arr = NDArray{T}(undef, dims..., ctx = ctx)
  arr[:] = one(T)
  arr
end

ones(::Type{T}, dims::Int...) where T<:DType = ones(T, dims)

ones(dims::NTuple{N,Int}, ctx::Context = current_context()) where N =
  ones(MX_float, dims, ctx)
ones(dims::Int...) = ones(dims)

ones(x::NDArray)::typeof(x)      = ones_like(x)
Base.ones(x::NDArray)::typeof(x) = ones_like(x)

import Base: length, ndims

"""
    size(x::NDArray)
    size(x::NDArray, dims)

Get the shape of an `NDArray`. The shape is in Julia's column-major convention.
See also the notes on NDArray shapes [`NDArray`](@ref).
"""
function Base.size(x::NDArray)
  ref_ndim  = Ref{MX_uint}(0)
  ref_shape = Ref{Ptr{MX_uint}}(0)
  @mxcall(:MXNDArrayGetShape, (MX_handle, Ref{MX_uint}, Ref{Ptr{MX_uint}}),
          x, ref_ndim, ref_shape)
  tuple(map(Int, reverse(unsafe_wrap(Array, ref_shape[], ref_ndim[])))...)
end

Base.size(x::NDArray{T,N}, dims::Integer) where {T,N} = (dims > N) ? 1 : size(x)[dims]

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
function Base.eltype(x::Union{NDArray,MX_NDArrayHandle})
  dtype_ref = Ref{Cint}(0)
  @mxcall(:MXNDArrayGetDType, (MX_handle, Ptr{Cint}), x, dtype_ref)

  if dtype_ref[] == -1 # x->is_none()
    # TODO: unit test for this branch
    throw(MXError("Eltype of $x is not defined"))
  end

  fromTypeFlag(TypeFlag(dtype_ref[]))
end

@inline _first(x::NDArray) = try_get_shared(x, sync = :read) |> first

Base.first(x::NDArray) = _first(x)

Base.lastindex(x::NDArray) = length(x)

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
    @warn("Copying an NDArray to itself")
    return
  end

  _copyto(src, out=dst)
  return dst
end

function copy!(dst::Array{T}, src::NDArray{T}) where T<:DType
  @assert size(dst) == size(src)
  @mxcall(:MXNDArraySyncCopyToCPU, (MX_handle, Ptr{Cvoid}, Csize_t),
          src, pointer(dst), length(dst))
  dst
end

copy!(dst::Array{<:Real}, src::NDArray) = copy!(dst, copy(src))
copy!(dst::NDArray, src::AbstractArray) = copy!(dst, collect(src))

function copy!(dst::NDArray{T}, src::Array{<:Real}) where {T}
  @assert dst.writable
  @assert size(dst) == size(src)
  src = convert(Array{T}, src) # this might involve copying
  @mxcall(:MXNDArraySyncCopyFromCPU, (MX_handle, Ptr{Cvoid}, Csize_t),
          dst.handle, pointer(src), length(src))
  dst
end

function copy_ignore_shape!(dst::NDArray{T}, src::Array{<:Real}) where {T}
  @assert dst.writable
  @assert length(dst) == length(src)
  src = convert(Array{T}, src) # this might involve copying
  @mxcall(:MXNDArraySyncCopyFromCPU, (MX_handle, Ptr{Cvoid}, Csize_t),
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
copy

# Create copy: NDArray -> Julia Array
copy(x::NDArray{T,D}) where{T,D} = copy!(Array{T,D}(undef, size(x)), x)

# Create copy: NDArray -> NDArray in a given context
copy(x::NDArray{T,D}, ctx::Context) where {T,D} =
  copy!(NDArray{T,D}(_ndarray_alloc(T, size(x), ctx, true)), x)

# Create copy: Julia Array -> NDArray in a given context
copy(x::Array{T}, ctx::Context) where {T<:DType} =
  copy!(NDArray{T}(undef, size(x); ctx = ctx), x)

copy(x::AbstractArray, ctx::Context) =
  copy!(NDArray{eltype(x)}(undef, size(x); ctx = ctx), collect(x))

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
Base.hcat(xs::NDArray{T}...) where T = cat(xs..., dims = 2)

"""
    vcat(x::NDArray...)
"""
Base.vcat(xs::NDArray{T}...) where T = cat(xs..., dims = 1)

"""
    cat(xs::NDArray...; dims)

Concate the `NDArray`s which have the same element type along the `dims`.
Building a diagonal matrix is not supported yet.
"""
function Base.cat(xs::NDArray{T}...; dims) where T
  ns = ndims.(xs)
  d = Base.max(dims, maximum(ns))
  xs′ = map(zip(ns, xs)) do i
    n, x = i
    (d > n) ? reshape(x, -2, Base.ones(Int, d - n)...) : x
  end
  concat(xs′..., dim = d - dims)
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

"""
    fill!(arr::NDArray, x)

Create an `NDArray` filled with the value `x`, like `Base.fill!`.
"""
function Base.fill!(arr::NDArray, x)
  arr[:] = x
  arr
end

"""
    fill(x, dims, ctx = current_context())
    fill(x, dims...)

Create an `NDArray` filled with the value `x`, like `Base.fill`.
"""
function fill(x::T, dims::NTuple{N,Integer}, ctx::Context = current_context()) where {T,N}
  arr = NDArray{T}(undef, dims, ctx = ctx)
  arr[:] = x
  arr
end

fill(x, dims::Integer...) = fill(x, dims)

import Base: hypot

broadcasted(::typeof(hypot), x::NDArray{T}, y::NDArray{T}) where {T} =
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
  let_assignments  = Expr(:block, [:($v = try_get_shared($v)) for v in nd_all]...)
  sync_statements  = map(rw_origs, nd_rw) do v_orig, v
    quote
      if !is_shared($v, $v_orig)
        # copy data back if not or no longer sharing data
        copy!($v_orig, $v)
      end
    end
  end
  sync_statements  = Expr(:block, sync_statements...)

  let_statement = Expr(:let, let_assignments, quote
    $stmts
    $sync_statements
  end)
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
  pdata = Ref{Ptr{Cvoid}}(0)
  @mxcall(:MXNDArrayGetData, (MX_handle, Ref{Ptr{Cvoid}}), arr, pdata)
  return convert(Ptr{eltype(arr)}, pdata[])
end

_ndsig[:reshape] = :(reshape(x; shape = dim, reverse = !reverse))
@_remap Base.reshape(x::NDArray, dim...; reverse = false) reshape
@_remap Base.reshape(x::NDArray, dim   ; reverse = false) reshape

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

@_remap Base.permutedims(x::NDArray, axes) transpose(x; axes = length(axes) .- tuple(axes...))

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
@_remap(Base.broadcast_axes(x::NDArray, dim, size),
        broadcast_axes(x; axis = ndims(x) .- dim, size = size))

################################################################################
# remapping to solving type unstablility
################################################################################

@_remap _broadcast_hypot(x::NDArray, y::NDArray)  broadcast_hypot(x, y)
@_remap _broadcast_hypot!(x::NDArray, y::NDArray) broadcast_hypot(x, y)
