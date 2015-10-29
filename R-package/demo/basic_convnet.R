require(mxnet)

batch.size = 100
data = mx.symbol.Variable("data")
conv1= mx.symbol.Convolution(data = data, name="conv1", num_filter=32, kernel=c(3,3), stride=c(2,2))

bn1 = mx.symbol.BatchNorm(data = conv1, name="bn1")
act1 = mx.symbol.Activation(data = bn1, name="relu1", act_type="relu")

mp1 = mx.symbol.Pooling(data = act1, name = "mp1", kernel=c(2,2), stride=c(2,2), pool_type="max")

conv2= mx.symbol.Convolution(data = mp1, name="conv2", num_filter=32, kernel=c(3,3), stride=c(2,2))
bn2 = mx.symbol.BatchNorm(data = conv2, name="bn2")
act2 = mx.symbol.Activation(data = bn2, name="relu2", act_type="relu")

mp2 = mx.symbol.Pooling(data = act2, name = "mp2", kernel=c(2,2), stride=c(2,2), pool_type="max")


fl = mx.symbol.Flatten(data = mp2, name="flatten")
fc2 = mx.symbol.FullyConnected(data = fl, name="fc2", num_hidden=10)
softmax = mx.symbol.Softmax(data = fc2, name = "sm")

dtrain = mx.varg.io.MNISTIter(list(
  image="data/train-images-idx3-ubyte",
  label="data/train-labels-idx1-ubyte",
  data.shape=c(1, 28, 28),
  batch.size=batch.size,
  shuffle=TRUE,
  flat=FALSE,
  silent=0,
  seed=10))

dtest = mx.varg.io.MNISTIter(list(
  image="data/t10k-images-idx3-ubyte",
  label="data/t10k-labels-idx1-ubyte",
  data.shape=c(1, 28, 28),
  batch.size=batch.size,
  shuffle=FALSE,
  flat=TRUE,
  silent=0))

mx.set.seed(0)
devices = lapply(1:2, function(i) {
  mx.cpu(i)
})
model <- mx.model.FeedForward.create(softmax, X=dtrain, eval.data=dtest,
                                     ctx=devices, num.round=1,
                                     learning.rate=0.1, momentum=0.9,
                                     initializer=mx.init.uniform(0.07),
                                     batch.end.callback=mx.callback.log.train.metric(100))

