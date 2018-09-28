# MXNet

[![MXNet](http://pkg.julialang.org/badges/MXNet_0.6.svg)](http://pkg.julialang.org/?pkg=MXNet)


MXNet.jl is the [dmlc/mxnet](https://github.com/apache/incubator-mxnet) [Julia](http://julialang.org/) package. MXNet.jl brings flexible and efficient GPU computing and state-of-art deep learning to Julia. Some highlight of its features include:

* Efficient tensor/matrix computation across multiple devices, including multiple CPUs, GPUs and distributed server nodes.
* Flexible symbolic manipulation to composite and construction of state-of-the-art deep learning models.

Here is an example of how training a simple 3-layer MLP on MNIST:

```julia
using MXNet

mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size)

# setup model
model = mx.FeedForward(mlp, context=mx.cpu())

# optimization algorithm
# where η is learning rate and μ is momentum
optimizer = mx.SGD(η=0.1, μ=0.9)

# fit parameters
mx.fit(model, optimizer, train_provider, n_epoch=20, eval_data=eval_provider)
```

You can also predict using the `model` in the following way:

```julia
probs = mx.predict(model, eval_provider)

# collect all labels from eval data
labels = reduce(
  vcat,
  copy(mx.get(eval_provider, batch, :softmax_label)) for batch ∈ eval_provider)
# labels are 0...9
labels .= labels .+ 1

# Now we use compute the accuracy
pred = map(i -> indmax(probs[1:10, i]), 1:size(probs, 2))
correct = sum(pred .== labels)
accuracy = 100correct/length(labels)
@printf "Accuracy on eval set: %.2f%%\n" accuracy
```

For more details, please refer to the
[documentation](https://dmlc.github.io/MXNet.jl/latest) and [examples](examples).
