# MXNet.jl Namespace

Most the functions and types in MXNet.jl are organized in a flat namespace. Because many some functions are conflicting with existing names in the Julia Base module, we wrap them all in a `mx` module. The convention of accessing the MXNet.jl interface is the to use the `mx.` prefix explicitly:
```jl
using MXNet

x = mx.zeros(2,3)              # MXNet NDArray
y = zeros(eltype(x), size(x))  # Julia Array
copy!(y, x)                    # Overloaded function in Julia Base
z = mx.ones(size(x), mx.gpu()) # MXNet NDArray on GPU
mx.copy!(z, y)                 # Same as copy!(z, y)
```
Note functions like `size`, `copy!` that is extensively overloaded for various types works out of the box. But functions like `zeros` and `ones` will be ambiguous, so we always use the `mx.` prefix. If you prefer, the `mx.` prefix can be used explicitly for all MXNet.jl functions, including `size` and `copy!` as shown in the last line.

# High Level Interface

The way we build deep learning models in MXNet.jl is to use the powerful symbolic composition system. It is like [Theano](http://deeplearning.net/software/theano/), except that we avoided long expression compiliation time by providing *larger* neural network related building blocks to guarantee computation performance. See also [this note](http://mxnet.readthedocs.org/en/latest/program_model.html) for the design and trade-off of the MXNet symbolic composition system.

The basic type is `mx.Symbol`. The following is a trivial example of composing two symbols with the `+` operation.
```jl
A = mx.variable(:A)
B = mx.variable(:B)
C = A + B
```
We get a new *symbol* by composing existing *symbols* by some *operations*. A hierarchical architecture of a deep neural network could be realized by recursive composition. For example, the following code snippet shows a simple 2-layer MLP construction, using a hidden layer of 128 units and a ReLU activation function.
```jl
net = mx.variable(:data)
net = mx.FullyConnected(data=net, name=:fc1, num_hidden=128)
net = mx.Activation(data=net, name=:relu1, act_type=:relu)
net = mx.FullyConnected(data=net, name=:fc2, num_hidden=64)
net = mx.Softmax(data=net, name=:out)
```
Each time we take the previous symbol, and compose with an operation. Unlike the simple `+` example above, the *operations* here are "bigger" ones, that correspond to common computation layers in deep neural networks.

Each of those operation takes one or more input symbols for composition, with optional hyper-parameters (e.g. `num_hidden`, `act_type`) to further customize the composition results.

When applying those operations, we can also specify a `name` for the result symbol. This is convenient if we want to refer to this symbol later on. If not supplied, a name will be automatically generated.

# Low Level Interface
