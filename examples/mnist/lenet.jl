using MXNet

#--------------------------------------------------------------------------------
# define lenet

# input
data = mx.variable(:data)

# first conv
conv1 = mx.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.Activation(data=conv1, act_type=:tanh)
pool1 = mx.Pooling(data=tanh1, pool_type=:max, kernel=(2,2), stride=(2,2))

# second conv
conv2 = mx.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.Activation(data=conv2, act_type=:tanh)
pool2 = mx.Pooling(data=tanh2, pool_type=:max, kernel=(2,2), stride=(2,2))

# first fully-connected
flat  = mx.Flatten(data=pool2)
fc1   = mx.FullyConnected(data=flat, num_hidden=500)
tanh3 = mx.Activation(data=fc1, act_type=:tanh)

# second fully-connected
fc2   = mx.FullyConnected(data=tanh3, num_hidden=10)

# softmax loss
lenet = mx.Softmax(data=fc2, name=:softmax)


#--------------------------------------------------------------------------------
# load data
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size; flat=false)

#--------------------------------------------------------------------------------
# fit model
dev = mx.Context(mx.GPU)
estimator = mx.FeedForward(lenet, context=dev)

# optimizer
optimizer = mx.SGD(lr_scheduler=mx.FixedLearningRateScheduler(0.05),
                   mom_scheduler=mx.FixedMomentumScheduler(0.9),
                   weight_decay=0.00001)

# fit parameters
mx.fit(estimator, optimizer, train_provider, epoch_stop=20, eval_data=eval_provider)
