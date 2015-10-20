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
[MXNet/src/ndarray.jl:380](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L380)

---

<a id="method__copy.1" class="lexicon_definition"></a>
#### copy!(dst::Array{Float32, N},  src::MXNet.mx.NDArray)
Copy data from NDArray to Julia Array

*source:*
[MXNet/src/ndarray.jl:187](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L187)

---

<a id="method__copy.2" class="lexicon_definition"></a>
#### copy!(dst::MXNet.mx.NDArray,  src::MXNet.mx.NDArray)
Copy data between NDArrays

*source:*
[MXNet/src/ndarray.jl:175](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L175)

---

<a id="method__copy.3" class="lexicon_definition"></a>
#### copy!{T<:Real}(dst::MXNet.mx.NDArray,  src::Array{T<:Real, N})
Copy data from Julia Array to NDArray

*source:*
[MXNet/src/ndarray.jl:195](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L195)

---

<a id="method__copy.4" class="lexicon_definition"></a>
#### copy(arr::MXNet.mx.NDArray)
Create copy: NDArray -> Julia Array

*source:*
[MXNet/src/ndarray.jl:205](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L205)

---

<a id="method__copy.5" class="lexicon_definition"></a>
#### copy(arr::MXNet.mx.NDArray,  ctx::MXNet.mx.Context)
Create copy: NDArray -> NDArray in a given context

*source:*
[MXNet/src/ndarray.jl:211](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L211)

---

<a id="method__copy.6" class="lexicon_definition"></a>
#### copy{T<:Real}(arr::Array{T<:Real, N},  ctx::MXNet.mx.Context)
Create copy: Julia Array -> NDArray in a given context

*source:*
[MXNet/src/ndarray.jl:217](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L217)

---

<a id="method__ones.1" class="lexicon_definition"></a>
#### ones{N}(shape::NTuple{N, Int64})
Create NDArray and initialize with 1

*source:*
[MXNet/src/ndarray.jl:118](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L118)

---

<a id="method__setindex.1" class="lexicon_definition"></a>
#### setindex!(arr::MXNet.mx.NDArray,  val::Real,  ::Colon)
Assign all elements of an NDArray to a scalar

*source:*
[MXNet/src/ndarray.jl:155](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L155)

---

<a id="method__size.1" class="lexicon_definition"></a>
#### size(arr::MXNet.mx.NDArray)
Get the shape of an `NDArray`. Note the shape is converted to Julia convention.
    So the same piece of memory, in Julia (column-major), with shape (K, M, N), will be of the
    shape (N, M, K) in the Python (row-major) binding.


*source:*
[MXNet/src/ndarray.jl:84](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L84)

---

<a id="method__slice.1" class="lexicon_definition"></a>
#### slice(arr::MXNet.mx.NDArray,  ::Colon)
`slice` create a view into a sub-slice of an `NDArray`. Note only slicing at the slowest
changing dimension is supported. In Julia's column-major perspective, this is the last
dimension. For example, given an `NDArray` of shape (2,3,4), `sub(array, 2:3)` will create
a `NDArray` of shape (2,3,2), sharing the data with the original array. This operation is
used in data parallelization to split mini-batch into sub-batches for different devices.


*source:*
[MXNet/src/ndarray.jl:137](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L137)

---

<a id="method__zeros.1" class="lexicon_definition"></a>
#### zeros{N}(shape::NTuple{N, Int64})
Create zero-ed NDArray of specific shape

*source:*
[MXNet/src/ndarray.jl:105](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L105)

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
[MXNet/src/ndarray.jl:32](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L32)

---

<a id="macro___inplace.1" class="lexicon_definition"></a>
#### @inplace(stmt)
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


*source:*
[MXNet/src/ndarray.jl:247](https://github.com/dmlc/MXNet.jl/tree/d13ddc6542bdb00e26b87e721a9b0e79a22bbd66/src/ndarray.jl#L247)

