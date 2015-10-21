# MXNet.jl Namespace

Most the functions and types in MXNet.jl are organized in a flat namespace. Because many some functions are conflicting with existing names in the Julia Base module, we wrap them all in a `mx` module. The convention of accessing the MXNet.jl interface is the to use the `mx.` prefix explicitly:
```julia
using MXNet

x = mx.zeros(2,3)              # MXNet NDArray
y = zeros(eltype(x), size(x))  # Julia Array
copy!(y, x)                    # Overloaded function in Julia Base
z = mx.ones(size(x), mx.gpu()) # MXNet NDArray on GPU
mx.copy!(z, y)                 # Same as copy!(z, y)
```
Note functions like `size`, `copy!` that is extensively overloaded for various types works out of the box. But functions like `zeros` and `ones` will be ambiguous, so we always use the `mx.` prefix. If you prefer, the `mx.` prefix can be used explicitly for all MXNet.jl functions, including `size` and `copy!` as shown in the last line.

# High Level Interface

## Symbols and Composition

The way we build deep learning models in MXNet.jl is to use the powerful symbolic composition system. It is like [Theano](http://deeplearning.net/software/theano/), except that we avoided long expression compiliation time by providing *larger* neural network related building blocks to guarantee computation performance. See also [this note](http://mxnet.readthedocs.org/en/latest/program_model.html) for the design and trade-off of the MXNet symbolic composition system.

The basic type is `mx.Symbol`. The following is a trivial example of composing two symbols with the `+` operation.
```julia
A = mx.variable(:A)
B = mx.variable(:B)
C = A + B
```
We get a new *symbol* by composing existing *symbols* by some *operations*. A hierarchical architecture of a deep neural network could be realized by recursive composition. For example, the following code snippet shows a simple 2-layer MLP construction, using a hidden layer of 128 units and a ReLU activation function.
```julia
net = mx.variable(:data)
net = mx.FullyConnected(data=net, name=:fc1, num_hidden=128)
net = mx.Activation(data=net, name=:relu1, act_type=:relu)
net = mx.FullyConnected(data=net, name=:fc2, num_hidden=64)
net = mx.Softmax(data=net, name=:out)
```
Each time we take the previous symbol, and compose with an operation. Unlike the simple `+` example above, the *operations* here are "bigger" ones, that correspond to common computation layers in deep neural networks.

Each of those operation takes one or more input symbols for composition, with optional hyper-parameters (e.g. `num_hidden`, `act_type`) to further customize the composition results.

When applying those operations, we can also specify a `name` for the result symbol. This is convenient if we want to refer to this symbol later on. If not supplied, a name will be automatically generated.

Each symbol takes some arguments. For example, in the `+` case above, to compute the value of `C`, we will need to know the values of the two inputs `A` and `B`. For neural networks, the arguments are primarily two categories: *inputs* and *parameters*. *inputs* are data and labels for the networks, while *parameters* are typically trainable *weights*, *bias*, *filters*.

When composing symbols, their arguments accumulates. We can list all the arguments by
```julia
julia> mx.list_arguments(net)
6-element Array{Symbol,1}:
 :data         # Input data, name from the first data variable
 :fc1_weight   # Weights of the fully connected layer named :fc1
 :fc1_bias     # Bias of the layer :fc1
 :fc2_weight   # Weights of the layer :fc2
 :fc2_bias     # Bias of the layer :fc2
 :out_label    # Input label, required by the softmax layer named :out
```
Note the names of the arguments are generated according to the provided name for each layer. We can also specify those names explicitly:
```julia
net = mx.variable(:data)
w   = mx.variable(:myweight)
net = mx.FullyConnected(data=data, weight=w, name=:fc1, num_hidden=128)
mx.list_arguments(net)
# =>
# 3-element Array{Symbol,1}:
#  :data
#  :myweight
#  :fc1_bias
```
The simple fact is that a `variable` is just a placeholder `mx.Symbol`. In composition, we can use arbitrary symbols for arguments. For example:
```julia
net  = mx.variable(:data)
net  = mx.FullyConnected(data=net, name=:fc1, num_hidden=128)
net2 = mx.variable(:data2)
net2 = mx.FullyConnected(data=net2, name=:net2, num_hidden=128)
mx.list_arguments(net2)
# =>
# 3-element Array{Symbol,1}:
#  :data2
#  :net2_weight
#  :net2_bias
composed_net = net2(data2=net, name=:composed)
mx.list_arguments(composed_net)
# =>
# 5-element Array{Symbol,1}:
#  :data
#  :fc1_weight
#  :fc1_bias
#  :net2_weight
#  :net2_bias
```
Note we use a composed symbol, `net` as the argument `data2` for `net2` to get a new symbol, which we named `:composed`. It also shows that a symbol itself is a call-able object, which can be invoked to fill in missing arguments and get more complicated symbol compositions.

## Shape Inference

Given enough information, the shapes of all arguments in a composed symbol could be inferred automatically. For example, given the input shape, and some hyper-parameters like `num_hidden`, the shapes for the weights and bias in a neural network could be inferred.
```julia
net = mx.variable(:data)
net = mx.FullyConnected(data=net, name=:fc1, num_hidden=10)
arg_shapes, out_shapes, aux_shapes = mx.infer_shape(net, data=(10, 64))
```
The returned shapes corresponds to arguments with the same order as returned by `mx.list_arguments`. The `out_shapes` are shapes for outputs, and `aux_shapes` can be safely ignored for now.
```julia
for (n,s) in zip(mx.list_arguments(net), arg_shapes)
  println("$n => $s")
end
# =>
# data => (10,64)
# fc1_weight => (10,10)
# fc1_bias => (10,)
for (n,s) in zip(mx.list_outputs(net), out_shapes)
  println("$n => $s")
end
# =>
# fc1_output => (10,64)
```

# Low Level Interface
