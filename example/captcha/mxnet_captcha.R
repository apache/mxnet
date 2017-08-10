library(mxnet)

data <- mx.symbol.Variable('data')
label <- mx.symbol.Variable('label')
conv1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 32)
pool1 <- mx.symbol.Pooling(data = conv1, pool_type = "max", kernel = c(2, 2), stride = c(1, 1))
relu1 <- mx.symbol.Activation(data = pool1, act_type = "relu")

conv2 <- mx.symbol.Convolution(data = relu1, kernel = c(5, 5), num_filter = 32)
pool2 <- mx.symbol.Pooling(data = conv2, pool_type = "avg", kernel = c(2, 2), stride = c(1, 1))
relu2 <- mx.symbol.Activation(data = pool2, act_type = "relu")

flatten <- mx.symbol.Flatten(data = relu2)
fc1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 120)
fc21 <- mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
fc22 <- mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
fc23 <- mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
fc24 <- mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
fc2 <- mx.symbol.Concat(c(fc21, fc22, fc23, fc24), dim = 0, num.args = 4)
label <- mx.symbol.transpose(data = label)
label <- mx.symbol.Reshape(data = label, target_shape = c(0))
captcha_net <- mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")

mx.metric.acc2 <- mx.metric.custom("accuracy", function(label, pred) {
    ypred <- max.col(t(pred)) - 1
    ypred <- matrix(ypred, nrow = nrow(label), ncol = ncol(label), byrow = TRUE)
    return(sum(colSums(label == ypred) == 4) / ncol(label))
  })

data.shape <- c(80, 30, 3)

batch_size <- 40

train <- mx.io.ImageRecordIter(
  path.imgrec     = "train.rec",
  path.imglist    = "train.lst",
  batch.size      = batch_size,
  label.width     = 4,
  data.shape      = data.shape,
  mean.img        = "mean.bin"
)

val <- mx.io.ImageRecordIter(
  path.imgrec     = "test.rec",
  path.imglist    = "test.lst",
  batch.size      = batch_size,
  label.width     = 4,
  data.shape      = data.shape,
  mean.img        = "mean.bin"
)

mx.set.seed(42)

model <- mx.model.FeedForward.create(
  X                  = train,
  eval.data          = val,
  ctx                = mx.gpu(),
  symbol             = captcha_net,
  eval.metric        = mx.metric.acc2,
  num.round          = 10,
  learning.rate      = 0.0001,
  momentum           = 0.9,
  wd                 = 0.00001,
  batch.end.callback = mx.callback.log.train.metric(50),
  initializer        = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
  optimizer          = "sgd",
  clip_gradient = 10
)
