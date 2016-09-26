using MXNet

#--------------------------------------------------------------------------------
# define MLP
# the following two ways are equivalent

#-- Option 1: explicit composition
# data = mx.Variable(:data)
# fc1  = mx.FullyConnected(data, name=:fc1, num_hidden=128)
# act1 = mx.Activation(fc1, name=:relu1, act_type=:relu)
# fc2  = mx.FullyConnected(act1, name=:fc2, num_hidden=64)
# act2 = mx.Activation(fc2, name=:relu2, act_type=:relu)
# fc3  = mx.FullyConnected(act2, name=:fc3, num_hidden=10)
# mlp  = mx.SoftmaxOutput(fc3, name=:softmax)

#-- Option 2: using the mx.chain macro
# mlp = @mx.chain mx.Variable(:data)             =>
#   mx.FullyConnected(name=:fc1, num_hidden=128) =>
#   mx.Activation(name=:relu1, act_type=:relu)   =>
#   mx.FullyConnected(name=:fc2, num_hidden=64)  =>
#   mx.Activation(name=:relu2, act_type=:relu)   =>
#   mx.FullyConnected(name=:fc3, num_hidden=10)  =>
#   mx.SoftmaxOutput(name=:softmax)

#-- Option 3: using nn-factory
mlp = @mx.chain mx.Variable(:data) =>
  mx.MLP([128, 64, 10])            =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size)

# setup model
model = mx.FeedForward(mlp, context=mx.cpu())

# optimizer
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)

# fit parameters
mx.fit(model, optimizer, train_provider, eval_data=eval_provider, n_epoch=20)

#--------------------------------------------------------------------------------
# Optional, demonstration of the predict API
probs = mx.predict(model, eval_provider)

# collect all labels from eval data
labels = Array[]
for batch in eval_provider
  push!(labels, copy(mx.get(eval_provider, batch, :softmax_label)))
end
labels = cat(1, labels...)

# Now we use compute the accuracy
correct = 0
for i = 1:length(labels)
  # labels are 0...9
  if indmax(probs[:,i]) == labels[i]+1
    correct += 1
  end
end
println(mx.format("Accuracy on eval set: {1:.2f}%", 100correct/length(labels)))
