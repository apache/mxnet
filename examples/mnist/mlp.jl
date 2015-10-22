using MXNet

#--------------------------------------------------------------------------------
# define MLP
# the following two ways are equivalent

#-- Option 1: explicit composition
# data = mx.Variable(:data)
# fc1  = mx.FullyConnected(data = data, name=:fc1, num_hidden=128)
# act1 = mx.Activation(data = fc1, name=:relu1, act_type=:relu)
# fc2  = mx.FullyConnected(data = act1, name=:fc2, num_hidden=64)
# act2 = mx.Activation(data = fc2, name=:relu2, act_type=:relu)
# fc3  = mx.FullyConnected(data = act2, name=:fc3, num_hidden=10)
# mlp  = mx.Softmax(data = fc3, name=:softmax)

#-- Option 2: using the mx.chain macro
mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.Softmax(name=:softmax)

# data provider
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size)

# setup estimator
estimator = mx.FeedForward(mlp, context=mx.cpu())

# optimizer
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)

# fit parameters
mx.fit(estimator, optimizer, train_provider, epoch_stop=20, eval_data=eval_provider)
