# MXNet.mx

## Internal

---

<a id="method___import_ndarray_functions.1" class="lexicon_definition"></a>
#### _import_ndarray_functions()
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


*source:*
[MXNet/src/ndarray.jl:367](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L367)

---

<a id="method___split_inputs.1" class="lexicon_definition"></a>
#### _split_inputs(batch_size::Int64,  n_split::Int64)
Get a split of `batch_size` into `n_split` pieces for data parallelization. Returns a vector
    of length `n_split`, with each entry a `UnitRange{Int}` indicating the slice index for that
    piece.


*source:*
[MXNet/src/estimator.jl:18](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/estimator.jl#L18)

---

<a id="method__copy.1" class="lexicon_definition"></a>
#### copy!(dst::Array{Float32, N},  src::MXNet.mx.NDArray)
Copy data from NDArray to Julia Array

*source:*
[MXNet/src/ndarray.jl:178](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L178)

---

<a id="method__copy.2" class="lexicon_definition"></a>
#### copy!(dst::MXNet.mx.NDArray,  src::MXNet.mx.NDArray)
Copy data between NDArrays

*source:*
[MXNet/src/ndarray.jl:166](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L166)

---

<a id="method__copy.3" class="lexicon_definition"></a>
#### copy!{T<:Real}(dst::MXNet.mx.NDArray,  src::Array{T<:Real, N})
Copy data from Julia Array to NDArray

*source:*
[MXNet/src/ndarray.jl:186](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L186)

---

<a id="method__copy.4" class="lexicon_definition"></a>
#### copy(arr::MXNet.mx.NDArray)
Create copy: NDArray -> Julia Array

*source:*
[MXNet/src/ndarray.jl:196](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L196)

---

<a id="method__copy.5" class="lexicon_definition"></a>
#### copy(arr::MXNet.mx.NDArray,  ctx::MXNet.mx.Context)
Create copy: NDArray -> NDArray in a given context

*source:*
[MXNet/src/ndarray.jl:202](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L202)

---

<a id="method__copy.6" class="lexicon_definition"></a>
#### copy{T<:Real}(arr::Array{T<:Real, N},  ctx::MXNet.mx.Context)
Create copy: Julia Array -> NDArray in a given context

*source:*
[MXNet/src/ndarray.jl:208](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L208)

---

<a id="method__get_internals.1" class="lexicon_definition"></a>
#### get_internals(self::MXNet.mx.Symbol)
Get a new grouped symbol whose output contains all the internal outputs of this symbol.

*source:*
[MXNet/src/symbol.jl:63](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/symbol.jl#L63)

---

<a id="method__group.1" class="lexicon_definition"></a>
#### group(symbols::MXNet.mx.Symbol...)
Create a symbol that groups symbols together

*source:*
[MXNet/src/symbol.jl:77](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/symbol.jl#L77)

---

<a id="method__list_auxiliary_states.1" class="lexicon_definition"></a>
#### list_auxiliary_states(self::MXNet.mx.Symbol)
List all auxiliary states in the symbool.

Auxiliary states are special states of symbols that do not corresponds to an argument,
and do not have gradient. But still be useful for the specific operations.
A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
Most operators do not have Auxiliary states.


*source:*
[MXNet/src/symbol.jl:58](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/symbol.jl#L58)

---

<a id="method__ones.1" class="lexicon_definition"></a>
#### ones{N}(shape::NTuple{N, Int64})
Create NDArray and initialize with 1

*source:*
[MXNet/src/ndarray.jl:112](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L112)

---

<a id="method__ones.2" class="lexicon_definition"></a>
#### ones{N}(shape::NTuple{N, Int64},  ctx::MXNet.mx.Context)
Create NDArray and initialize with 1

*source:*
[MXNet/src/ndarray.jl:112](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L112)

---

<a id="method__setindex.1" class="lexicon_definition"></a>
#### setindex!(arr::MXNet.mx.NDArray,  val::Real,  ::Colon)
Assign all elements of an NDArray to a scalar

*source:*
[MXNet/src/ndarray.jl:146](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L146)

---

<a id="method__size.1" class="lexicon_definition"></a>
#### size(arr::MXNet.mx.NDArray)
Get the shape of an `NDArray`. Note the shape is converted to Julia convention.
    So the same piece of memory, in Julia (column-major), with shape (K, M, N), will be of the
    shape (N, M, K) in the Python (row-major) binding.


*source:*
[MXNet/src/ndarray.jl:81](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L81)

---

<a id="method__slice.1" class="lexicon_definition"></a>
#### slice(arr::MXNet.mx.NDArray,  ::Colon)
`slice` create a view into a sub-slice of an `NDArray`. Note only slicing at the slowest
    changing dimension is supported. In Julia's column-major perspective, this is the last
    dimension. For example, given an `NDArray` of shape (2,3,4), `sub(array, 2:3)` will create
    a `NDArray` of shape (2,3,2), sharing the data with the original array. This operation is
    used in data parallelization to split mini-batch into sub-batches for different devices.


*source:*
[MXNet/src/ndarray.jl:128](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L128)

---

<a id="method__variable.1" class="lexicon_definition"></a>
#### variable(name::Union{AbstractString, Symbol})
Create a symbolic variable with the given name

*source:*
[MXNet/src/symbol.jl:70](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/symbol.jl#L70)

---

<a id="method__zeros.1" class="lexicon_definition"></a>
#### zeros{N}(shape::NTuple{N, Int64})
Create zero-ed NDArray of specific shape

*source:*
[MXNet/src/ndarray.jl:102](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L102)

---

<a id="method__zeros.2" class="lexicon_definition"></a>
#### zeros{N}(shape::NTuple{N, Int64},  ctx::MXNet.mx.Context)
Create zero-ed NDArray of specific shape

*source:*
[MXNet/src/ndarray.jl:102](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L102)

---

<a id="type__abstractdatabatch.1" class="lexicon_definition"></a>
#### MXNet.mx.AbstractDataBatch
Root type for data batch

A data batch must implement the following interface function to actually provide the data and label.

```julia
load_data!(batch :: AbstractDataBatch, targets :: Vector{Vector{SlicedNDArray}})
load_label!(batch :: AbstractDataBatch, targets :: Vector{Vector{SlicedNDArray}})
```

Load data and label into targets. The targets is a list of target that the data/label should be
copied into. The order in the list is guaranteed to be the same as returned by `provide_data` and
`provide_label`. Each entry in the list is again a list of `SlicedNDArray`, corresponding the
memory buffer for each device.

The `SlicedNDArray` is used in data parallelization to run different sub-batch on different devices.

The following function should also be implemented to handle the case when the mini-batch size does not
divide the size of the whole dataset. So in the last mini-batch, the actual data copied might be fewer
than the mini-batch size. This is usually not an issue during the training as the remaining space may
contain the data and label copied during the previous mini-batch are still valid data. However, during
testing, especially when doing feature extraction, we need to be precise about the number of samples
processed.

```julia
get_pad(batch :: AbstractDataBatch)
```

Return the number of *dummy samples* in this mini-batch.


*source:*
[MXNet/src/io.jl:110](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/io.jl#L110)

---

<a id="type__abstractdataprovider.1" class="lexicon_definition"></a>
#### MXNet.mx.AbstractDataProvider
Root type for data provider

A data provider provides interface to iterate over a dataset. It should implement the following functions:

```julia
provide_data(provider :: AbstractDataProvider) => Vector{Tuple{Base.Symbol, Tuple}}
provide_label(provider :: AbstractDataProvider) => Vector{Tuple{Base.Symbol, Tuple}}
```

Returns a list of name-shape pairs, indicating the name and shape of the each data stream. For example,
`[(:data, (100,1,28,28))]` or `[(:softmax_label, (100,1))]`. It should also implement the following convenient
function

```julia
get_batch_size(provider :: AbstractDataProvider) => Int
```

which returns the batch size used in this data provider.

A data provider should implement the standard Julia iteration interface, including `Base.start`,
`Base.next`, `Base.done` and `Base.eltype`. It could safely assume that the interface functions will
always be called like

```julia
for batch in provider
  # ...
  load_data!(batch, targets)
end
```

which translates into

```julia
state = Base.start(provider)
while !done(provider, state)
  (batch, state) = next(provider, state)
  # ...
  load_data!(batch, targets)
end
```

In other words, it could safely assume that `Base.next` is always called after `Base.done`. And neither
of those function will be called twice consequtively. The detailed interfaces are list below:

```julia
Base.start(provider :: AbstractDataProvider) => AbstractDataProviderState
```

Initialize or reset the data iteration.

```julia
Base.next(provider :: AbstractDataProvider, state :: AbstractDataProviderState)
    => (AbstractDataBatch, AbstractDataProviderState)
```

Return one batch of data. Actual data can be retrieved from the batch by interface functions described
in the document of type `AbstractDataBatch`.

```julia
Base.done(provider :: AbstractDataProvider, state :: AbstractDataProviderState) => Bool
```

Return `false` if there is more batch to get.

```julia
Base.eltype(::Type{MyDataProvider}) => MyDataProviderState
```

Return the type of the data provider state.


*source:*
[MXNet/src/io.jl:71](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/io.jl#L71)

---

<a id="type__abstractdataproviderstate.1" class="lexicon_definition"></a>
#### MXNet.mx.AbstractDataProviderState
Root type for states of data provider

*source:*
[MXNet/src/io.jl:74](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/io.jl#L74)

---

<a id="type__mxdataprovider.1" class="lexicon_definition"></a>
#### MXNet.mx.MXDataProvider
Wrapper of built-in `libmxnet` data iterators.


*source:*
[MXNet/src/io.jl:119](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/io.jl#L119)

---

<a id="type__mxerror.1" class="lexicon_definition"></a>
#### MXNet.mx.MXError
Exception thrown when an error occurred calling MXNet API.

*source:*
[MXNet/src/init.jl:2](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/init.jl#L2)

---

<a id="type__ndarray.1" class="lexicon_definition"></a>
#### MXNet.mx.NDArray
Wrapper of the `NDArray` type in `libmxnet`. This is the basic building block
    of tensor-based computation.

    **Note** since C/C++ use row-major ordering for arrays while Julia follows a
    column-major ordering. To keep things consistent, we keep the underlying data
    in their original layout, but use *language-native* convention when we talk
    about shapes. For example, a mini-batch of 100 MNIST images is a tensor of
    C/C++/Python shape (100,1,28,28), while in Julia, the same piece of memory
    have shape (28,28,1,100).


*source:*
[MXNet/src/ndarray.jl:32](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L32)

---

<a id="typealias__slicedndarray.1" class="lexicon_definition"></a>
#### SlicedNDArray
A tuple of (slice, NDArray). Usually each NDArray resides on a different device, and each
    slice describe which part of a larger piece of data should goto that device.


*source:*
[MXNet/src/io.jl:79](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/io.jl#L79)

---

<a id="macro___inplace.1" class="lexicon_definition"></a>
#### @inplace(stmt)
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


*source:*
[MXNet/src/ndarray.jl:234](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/ndarray.jl#L234)

---

<a id="macro___mxcall.1" class="lexicon_definition"></a>
#### @mxcall(fv, argtypes, args...)
Utility macro to call MXNet API functions

*source:*
[MXNet/src/init.jl:41](https://github.com/dmlc/MXNet.jl/tree/34a1c89bf2b65351914e00ccd12a033df724a721/src/init.jl#L41)

