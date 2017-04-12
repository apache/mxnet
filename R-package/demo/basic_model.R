list.of.packages <- c("R.utils")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "https://cloud.r-project.org/")

setwd(tempdir())

download.file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", destfile="train-images-idx3-ubyte.gz")

download.file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", destfile="train-labels-idx1-ubyte.gz")

download.file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", destfile="t10k-images-idx3-ubyte.gz")

download.file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", destfile="t10k-labels-idx1-ubyte.gz")

require(R.utils)

gunzip("train-images-idx3-ubyte.gz")

gunzip("train-labels-idx1-ubyte.gz")

gunzip("t10k-images-idx3-ubyte.gz")

gunzip("t10k-labels-idx1-ubyte.gz")

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

dtrain = mx.io.MNISTIter(
  image="train-images-idx3-ubyte",
  label="train-labels-idx1-ubyte",
  data.shape=c(784),
  batch.size=batch.size,
  shuffle=TRUE,
  flat=TRUE,
  silent=0,
  seed=10)

dtest = mx.io.MNISTIter(
  image="t10k-images-idx3-ubyte",
  label="t10k-labels-idx1-ubyte",
  data.shape=c(784),
  batch.size=batch.size,
  shuffle=FALSE,
  flat=TRUE,
  silent=0)

mx.set.seed(0)
devices = lapply(1:2, function(i) {
  mx.cpu(i)
})

# create the model
model <- mx.model.FeedForward.create(softmax, X=dtrain, eval.data=dtest,
                                     ctx=devices, num.round=1,
                                     learning.rate=0.1, momentum=0.9,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.save.checkpoint("chkpt"),
                                     batch.end.callback=mx.callback.log.train.metric(100))

# do prediction
pred <- predict(model, dtest)
label <- mx.io.extract(dtest, "label")
dataX <- mx.io.extract(dtest, "data")
# Predict with R's array
pred2 <- predict(model, X=dataX)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(label, pred)))
print(paste0("Finish prediction... accuracy2=", accuracy(label, pred2)))



# load the model
model <- mx.model.load("chkpt", 1)

#continue training with some new arguments
model <- mx.model.FeedForward.create(model$symbol, X=dtrain, eval.data=dtest,
                                     ctx=devices, num.round=5,
                                     learning.rate=0.1, momentum=0.9,
                                     epoch.end.callback=mx.callback.save.checkpoint("reload_chkpt"),
                                     batch.end.callback=mx.callback.log.train.metric(100),
                                     arg.params=model$arg.params, aux.params=model$aux.params)

# do prediction
pred <- predict(model, dtest)
label <- mx.io.extract(dtest, "label")
dataX <- mx.io.extract(dtest, "data")
# Predict with R's array
pred2 <- predict(model, X=dataX)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(label, pred)))
print(paste0("Finish prediction... accuracy2=", accuracy(label, pred2)))




