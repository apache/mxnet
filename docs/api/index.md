# API-INDEX


## MODULE: MXNet.mx

---

## Methods [Internal]

[_compose!(sym::MXNet.mx.Symbol)](MXNet.md#method___compose.1)  Compose symbol on inputs

[_default_get_name!(counter::Dict{Symbol, Int64},  name::Union{AbstractString, Symbol},  hint::Union{AbstractString, Symbol})](MXNet.md#method___default_get_name.1)  Default implementation for generating a name for a symbol.

[_import_ndarray_functions()](MXNet.md#method___import_ndarray_functions.1)  Import dynamic functions for NDArrays. The arguments to the functions are typically ordered

[_split_inputs(batch_size::Int64,  n_split::Int64)](MXNet.md#method___split_inputs.1)  Get a split of `batch_size` into `n_split` pieces for data parallelization. Returns a vector

[copy!(dst::Array{Float32, N},  src::MXNet.mx.NDArray)](MXNet.md#method__copy.1)  Copy data from NDArray to Julia Array

[copy!(dst::MXNet.mx.NDArray,  src::MXNet.mx.NDArray)](MXNet.md#method__copy.2)  Copy data between NDArrays

[copy!{T<:Real}(dst::MXNet.mx.NDArray,  src::Array{T<:Real, N})](MXNet.md#method__copy.3)  Copy data from Julia Array to NDArray

[copy(arr::MXNet.mx.NDArray)](MXNet.md#method__copy.4)  Create copy: NDArray -> Julia Array

[copy(arr::MXNet.mx.NDArray,  ctx::MXNet.mx.Context)](MXNet.md#method__copy.5)  Create copy: NDArray -> NDArray in a given context

[copy{T<:Real}(arr::Array{T<:Real, N},  ctx::MXNet.mx.Context)](MXNet.md#method__copy.6)  Create copy: Julia Array -> NDArray in a given context

[get_internals(self::MXNet.mx.Symbol)](MXNet.md#method__get_internals.1)  Get a new grouped symbol whose output contains all the internal outputs of this symbol.

[group(symbols::MXNet.mx.Symbol...)](MXNet.md#method__group.1)  Create a symbol that groups symbols together

[list_auxiliary_states(self::MXNet.mx.Symbol)](MXNet.md#method__list_auxiliary_states.1)  List all auxiliary states in the symbool.

[ones{N}(shape::NTuple{N, Int64})](MXNet.md#method__ones.1)  Create NDArray and initialize with 1

[ones{N}(shape::NTuple{N, Int64},  ctx::MXNet.mx.Context)](MXNet.md#method__ones.2)  Create NDArray and initialize with 1

[setindex!(arr::MXNet.mx.NDArray,  val::Real,  ::Colon)](MXNet.md#method__setindex.1)  Assign all elements of an NDArray to a scalar

[size(arr::MXNet.mx.NDArray)](MXNet.md#method__size.1)  Get the shape of an `NDArray`. Note the shape is converted to Julia convention.

[slice(arr::MXNet.mx.NDArray,  ::Colon)](MXNet.md#method__slice.1)  `slice` create a view into a sub-slice of an `NDArray`. Note only slicing at the slowest

[variable(name::Union{AbstractString, Symbol})](MXNet.md#method__variable.1)  Create a symbolic variable with the given name

[zeros{N}(shape::NTuple{N, Int64})](MXNet.md#method__zeros.1)  Create zero-ed NDArray of specific shape

[zeros{N}(shape::NTuple{N, Int64},  ctx::MXNet.mx.Context)](MXNet.md#method__zeros.2)  Create zero-ed NDArray of specific shape

---

## Types [Internal]

[MXNet.mx.AbstractDataBatch](MXNet.md#type__abstractdatabatch.1)  Root type for data batch

[MXNet.mx.AbstractDataProvider](MXNet.md#type__abstractdataprovider.1)  Root type for data provider

[MXNet.mx.AbstractDataProviderState](MXNet.md#type__abstractdataproviderstate.1)  Root type for states of data provider

[MXNet.mx.MXDataProvider](MXNet.md#type__mxdataprovider.1)  Wrapper of built-in `libmxnet` data iterators.

[MXNet.mx.MXError](MXNet.md#type__mxerror.1)  Exception thrown when an error occurred calling MXNet API.

[MXNet.mx.NDArray](MXNet.md#type__ndarray.1)  Wrapper of the `NDArray` type in `libmxnet`. This is the basic building block

---

## Typealiass [Internal]

[SlicedNDArray](MXNet.md#typealias__slicedndarray.1)  A tuple of (slice, NDArray). Usually each NDArray resides on a different device, and each

---

## Macros [Internal]

[@inplace(stmt)](MXNet.md#macro___inplace.1)  Julia does not support re-definiton of += operator (like __iadd__ in python),

[@mxcall(fv, argtypes, args...)](MXNet.md#macro___mxcall.1)  Utility macro to call MXNet API functions

