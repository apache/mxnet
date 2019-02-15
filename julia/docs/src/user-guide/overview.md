<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Overview

## MXNet.jl Namespace

Most the functions and types in MXNet.jl are organized in a flat
namespace. Because many some functions are conflicting with existing
names in the Julia Base module, we wrap them all in a `mx` module. The
convention of accessing the MXNet.jl interface is the to use the `mx.`
prefix explicitly:

```julia
julia> using MXNet

julia> x = mx.zeros(2, 3)             # MXNet NDArray
2×3 mx.NDArray{Float32} @ CPU0:
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> y = zeros(eltype(x), size(x))  # Julia Array
2×3 Array{Float32,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> copy!(y, x)                    # Overloaded function in Julia Base
2×3 Array{Float32,2}:
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> z = mx.ones(size(x), mx.gpu()) # MXNet NDArray on GPU
2×3 mx.NDArray{Float32} @ GPU0:
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> mx.copy!(z, y)                 # Same as copy!(z, y)
2×3 mx.NDArray{Float32} @ GPU0:
 0.0  0.0  0.0
 0.0  0.0  0.0
```

Note functions like `size`, `copy!` that is extensively overloaded for
various types works out of the box. But functions like `zeros` and
`ones` will be ambiguous, so we always use the `mx.` prefix. If you
prefer, the `mx.` prefix can be used explicitly for all MXNet.jl
functions, including `size` and `copy!` as shown in the last line.

## Low Level Interface

### `NDArray`

`NDArray` is the basic building blocks of the actual computations in
MXNet. It is like a Julia `Array` object, with some important
differences listed here:

- The actual data could live on different `Context` (e.g. GPUs). For
  some contexts, iterating into the elements one by one is very slow,
  thus indexing into NDArray is not recommanded in general. The easiest
  way to inspect the contents of an NDArray is to use the `copy`
  function to copy the contents as a Julia `Array`.
- Operations on `NDArray` (including basic arithmetics and neural
  network related operators) are executed in parallel with automatic
  dependency tracking to ensure correctness.
- There is no generics in `NDArray`, the `eltype` is always
  `mx.MX_float`. Because for applications in machine learning, single
  precision floating point numbers are typical a best choice balancing
  between precision, speed and portability. Also since libmxnet is
  designed to support multiple languages as front-ends, it is much
  simpler to implement with a fixed data type.

While most of the computation is hidden in libmxnet by operators
corresponding to various neural network layers. Getting familiar with
the `NDArray` API is useful for implementing `Optimizer` or customized
operators in Julia directly.

The followings are common ways to create `NDArray` objects:

- `NDArray(undef, shape...; ctx = context, writable = true)`:
  create an uninitialized array of a given shape on a specific device.
  For example,
  `NDArray(undef, 2, 3)`, `NDArray(undef, 2, 3, ctx = mx.gpu(2))`.
- `NDArray(undef, shape; ctx = context, writable = true)`
- `NDArray{T}(undef, shape...; ctx = context, writable = true)`:
  create an uninitialized with the given type `T`.
- `mx.zeros(shape[, context])` and `mx.ones(shape[, context])`:
  similar to the Julia's built-in `zeros` and `ones`.
- `mx.copy(jl_arr, context)`: copy the contents of a Julia `Array` to
  a specific device.

Most of the convenient functions like `size`, `length`, `ndims`,
`eltype` on array objects should work out-of-the-box. Although indexing
is not supported, it is possible to take *slices*:

```@repl
using MXNet
a = mx.ones(2, 3)
b = mx.slice(a, 1:2)
b[:] = 2
a
```

A slice is a sub-region sharing the same memory with the original
`NDArray` object. A slice is always a contiguous piece of memory, so only
slicing on the *last* dimension is supported. The example above also
shows a way to set the contents of an `NDArray`.

```@repl
using MXNet
mx.srand(42)
a = NDArray(undef, 2, 3)
a[:] = 0.5              # set all elements to a scalar
a[:] = rand(size(a))    # set contents with a Julia Array
copy!(a, rand(size(a))) # set value by copying a Julia Array
b = NDArray(undef, size(a))
b[:] = a                # copying and assignment between NDArrays
```

Note due to the intrinsic design of the Julia language, a normal
assignment

```julia
a = b
```

does **not** mean copying the contents of `b` to `a`. Instead, it just
make the variable `a` pointing to a new object, which is `b`.
Similarly, inplace arithmetics does not work as expected:

```@repl inplace-macro
using MXNet
a = mx.ones(2)
r = a           # keep a reference to a
b = mx.ones(2)
a += b          # translates to a = a + b
a
r
```

As we can see, `a` has expected value, but instead of inplace updating,
a new `NDArray` is created and `a` is set to point to this new object. If
we look at `r`, which still reference to the old `a`, its content has
not changed. There is currently no way in Julia to overload the
operators like `+=` to get customized behavior.

Instead, you will need to write `a[:] = a + b`, or if you want *real*
inplace `+=` operation, MXNet.jl provides a simple macro `@mx.inplace`:

```@repl inplace-macro
@mx.inplace a += b
macroexpand(:(@mx.inplace a += b))
```

As we can see, it translate the `+=` operator to an explicit `add_to!`
function call, which invokes into libmxnet to add the contents of `b`
into `a` directly. For example, the following is the update rule in the
`SGD Optimizer` (both gradient `∇` and weight `W` are `NDArray` objects):

```julia
@inplace W .+= -η .* (∇ + λ .* W)
```

Note there is no much magic in `mx.inplace`: it only does a shallow
translation. In the SGD update rule example above, the computation like
scaling the gradient by `grad_scale` and adding the weight decay all
create temporary `NDArray` objects. To mitigate this issue, libmxnet has a
customized memory allocator designed specifically to handle this kind of
situations. The following snippet does a simple benchmark on allocating
temp `NDArray` vs. pre-allocating:

```julia
using Benchmark
using MXNet

N_REP = 1000
SHAPE = (128, 64)
CTX   = mx.cpu()
LR    = 0.1

function inplace_op()
  weight = mx.zeros(SHAPE, CTX)
  grad   = mx.ones(SHAPE, CTX)

  # pre-allocate temp objects
  grad_lr = NDArray(undef, SHAPE, ctx = CTX)

  for i = 1:N_REP
    copy!(grad_lr, grad)
    @mx.inplace grad_lr .*= LR
    @mx.inplace weight -= grad_lr
  end
  return weight
end

function normal_op()
  weight = mx.zeros(SHAPE, CTX)
  grad   = mx.ones(SHAPE, CTX)

  for i = 1:N_REP
    weight[:] -= LR * grad
  end
  return weight
end

# make sure the results are the same
@assert(maximum(abs(copy(normal_op() - inplace_op()))) < 1e-6)

println(compare([inplace_op, normal_op], 100))
```

The comparison on my laptop shows that `normal_op` while allocating a
lot of temp NDArray in the loop (the performance gets worse when
increasing `N_REP`), is only about twice slower than the pre-allocated
one.

| Row    | Function        | Average      | Relative    | Replications    |
| ------ | --------------- | ------------ | ----------- | --------------- |
| 1      | "inplace\_op"   | 0.0074854    | 1.0         | 100             |
| 2      | "normal\_op"    | 0.0174202    | 2.32723     | 100             |

So it will usually not be a big problem unless you are at the bottleneck
of the computation.

### Distributed Key-value Store

The type `KVStore` and related methods are used for data sharing across
different devices or machines. It provides a simple and efficient
integer - NDArray key-value storage system that each device can pull or
push.

The following example shows how to create a local `KVStore`, initialize
a value and then pull it back.

```@setup kv
using MXNet
```

```@example kv
kv    = mx.KVStore(:local)
shape = (2, 3)
key   = 3

mx.init!(kv, key, mx.ones(shape) * 2)
a = NDArray(undef, shape)
mx.pull!(kv, key, a) # pull value into a
a
```

## Intermediate Level Interface

### Symbols and Composition

The way we build deep learning models in MXNet.jl is to use the powerful
symbolic composition system. It is like
[Theano](http://deeplearning.net/software/theano/), except that we
avoided long expression compilation time by providing *larger* neural
network related building blocks to guarantee computation performance.
See also [this note](http://mxnet.readthedocs.org/en/latest/program_model.html)
for the design and trade-off of the MXNet symbolic composition system.

The basic type is `mx.SymbolicNode`. The following is a trivial example of
composing two symbols with the `+` operation.

```@setup sym1
using MXNet
```

```@example sym1
A = mx.Variable(:A)
B = mx.Variable(:B)
C = A + B
print(C)  # debug printing
```

We get a new `SymbolicNode` by composing existing `SymbolicNode`s by some
*operations*. A hierarchical architecture of a deep neural network could
be realized by recursive composition. For example, the following code
snippet shows a simple 2-layer MLP construction, using a hidden layer of
128 units and a `ReLU` activation function.

```@setup fcnet
using MXNet
```

```@example fcnet
net = mx.Variable(:data)
net = mx.FullyConnected(net, name=:fc1, num_hidden=128)
net = mx.Activation(net, name=:relu1, act_type=:relu)
net = mx.FullyConnected(net, name=:fc2, num_hidden=64)
net = mx.SoftmaxOutput(net, name=:out)
print(net)  # debug printing
```

Each time we take the previous symbol, and compose with an operation.
Unlike the simple `+` example above, the *operations* here are "bigger"
ones, that correspond to common computation layers in deep neural
networks.

Each of those operation takes one or more input symbols for composition,
with optional hyper-parameters (e.g. `num_hidden`, `act_type`) to
further customize the composition results.

When applying those operations, we can also specify a `name` for the
result symbol. This is convenient if we want to refer to this symbol
later on. If not supplied, a name will be automatically generated.

Each symbol takes some arguments. For example, in the `+` case above, to
compute the value of `C`, we will need to know the values of the two
inputs `A` and `B`. For neural networks, the arguments are primarily two
categories: *inputs* and *parameters*. *inputs* are data and labels for
the networks, while *parameters* are typically trainable *weights*,
*bias*, *filters*.

When composing symbols, their arguments accumulates.
We can list all the arguments by

```@example fcnet
mx.list_arguments(net)
```

Note the names of the arguments are generated according to the provided
name for each layer. We can also specify those names explicitly:

```@repl
using MXNet
net = mx.Variable(:data)
w   = mx.Variable(:myweight)
net = mx.FullyConnected(net, weight=w, name=:fc1, num_hidden=128)
mx.list_arguments(net)
```

The simple fact is that a `Variable` is just a placeholder `mx.SymbolicNode`.
In composition, we can use arbitrary symbols for arguments. For example:

```@repl
using MXNet
net  = mx.Variable(:data)
net  = mx.FullyConnected(net, name=:fc1, num_hidden=128)
net2 = mx.Variable(:data2)
net2 = mx.FullyConnected(net2, name=:net2, num_hidden=128)
mx.list_arguments(net2)
composed_net = net2(data2=net, name=:composed)
mx.list_arguments(composed_net)
```

Note we use a composed symbol, `net` as the argument `data2` for `net2`
to get a new symbol, which we named `:composed`. It also shows that a
symbol itself is a call-able object, which can be invoked to fill in
missing arguments and get more complicated symbol compositions.

### Shape Inference

Given enough information, the shapes of all arguments in a composed
symbol could be inferred automatically. For example, given the input
shape, and some hyper-parameters like `num_hidden`, the shapes for the
weights and bias in a neural network could be inferred.

```@repl infer-shape
using MXNet
net = mx.Variable(:data)
net = mx.FullyConnected(net, name=:fc1, num_hidden=10)
arg_shapes, out_shapes, aux_shapes = mx.infer_shape(net, data=(10, 64))
```

The returned shapes corresponds to arguments with the same order as
returned by `mx.list_arguments`. The `out_shapes` are shapes for
outputs, and `aux_shapes` can be safely ignored for now.

```@repl infer-shape
for (n, s) in zip(mx.list_arguments(net), arg_shapes)
  println("$n\t=> $s")
end
```
```@repl infer-shape
for (n, s) in zip(mx.list_outputs(net), out_shapes)
  println("$n\t=> $s")
end
```

### Binding and Executing

In order to execute the computation graph specified a composed symbol,
we will *bind* the free variables to concrete values, specified as
`mx.NDArray`. This will create an `mx.Executor` on a given `mx.Context`.
A context describes the computation devices (CPUs, GPUs, etc.) and an
executor will carry out the computation (forward/backward) specified in
the corresponding symbolic composition.

```@repl
using MXNet
A = mx.Variable(:A)
B = mx.Variable(:B)
C = A .* B
a = mx.ones(3) * 4
b = mx.ones(3) * 2
c_exec = mx.bind(C, context=mx.cpu(), args=Dict(:A => a, :B => b));

mx.forward(c_exec)
c_exec.outputs[1]
copy(c_exec.outputs[1])  # copy turns NDArray into Julia Array
```

For neural networks, it is easier to use `simple_bind`. By providing the
shape for input arguments, it will perform a shape inference for the
rest of the arguments and create the NDArray automatically. In practice,
the binding and executing steps are hidden under the `Model` interface.

**TODO** Provide pointers to model tutorial and further details about
binding and symbolic API.

## High Level Interface

The high level interface include model training and prediction API, etc.
