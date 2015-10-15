# This is an example of training using R's array

require(mxnet)

# Network configuration
batch.size <- 100
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.Softmax(fc3, name = "sm")

dtrain = mx.varg.io.MNISTIter(list(
  image="data/train-images-idx3-ubyte",
  label="data/train-labels-idx1-ubyte",
  data.shape=c(784),
  batch.size=batch.size,
  flat=TRUE,
  silent=0,
  seed=10))

dtest = mx.varg.io.MNISTIter(list(
  image="data/t10k-images-idx3-ubyte",
  label="data/t10k-labels-idx1-ubyte",
  data.shape=c(784),
  batch.size=batch.size,
  shuffle=FALSE,
  flat=TRUE,
  silent=0))
# X is R's array, we load from mxnet's native iter structure, but you don't have to
X = mx.io.extract(dtrain, "data")
y = mx.io.extract(dtrain, "label")

devices = lapply(1:2, function(i) {
  mx.cpu(i)
})
# create the model
model <- mx.model.FeedForward.create(softmax, X=X, y=y,
                                     ctx=devices, num.round=1,
                                     learning.rate=0.1, momentum=0.9,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
