require(mxnet)
# A basic neural net training
# To run this, run python/mxnet/test_io.py to get data first

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
  image="data/train-images-idx3-ubyte",
  label="data/train-labels-idx1-ubyte",
  data.shape=c(784),
  batch.size=batch.size,
  shuffle=TRUE,
  flat=TRUE,
  silent=0,
  seed=10)

accuracy <- function(label, pred) {
  ypred = max.col(as.array(pred))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}
mx.set.seed(0)
# Training parameters
ctx <- mx.cpu()
input.shape <- c(batch.size, 784)
symbol <- softmax
init <- mx.init.uniform(0.07)
opt <- mx.opt.create("sgd", learning.rate=0.05, momentum=0.9, rescale.grad=1.0/batch.size)

# Training procedure
texec <- mx.simple.bind(symbol, ctx=ctx, data=input.shape, grad.req=TRUE)
shapes <- lapply(texec$ref.arg.arrays, dim)
names(shapes) <- names(texec$arg.arrays)
arg.arrays <- mx.init.create(init, shapes, ctx)
mx.exec.update.arg.arrays(texec, arg.arrays, match.name=TRUE)

updater <- mx.opt.get.updater(opt, texec$ref.arg.arrays)
nround <- 10
tic <- proc.time()

for (iteration in 1 : nround) {
  nbatch <- 0
  train.acc <- 0
  while (dtrain$iter.next()) {
    batch <- dtrain$value()
    label <- batch$label
    names(batch) <- c("data", "sm_label")
    # copy data arguments to executor
    mx.exec.update.arg.arrays(texec, batch, match.name=TRUE)
    # forward pass
    mx.exec.forward(texec, is.train=TRUE)
    # copy prediction out
    out.pred <- mx.nd.copyto(texec$outputs[[1]], mx.cpu())
    # backward pass
    mx.exec.backward(texec)
    arg.arrays <- updater(texec$arg.arrays, texec$ref.grad.arrays)
    mx.exec.update.arg.arrays(texec, arg.arrays, skip.null=TRUE)
    nbatch <- nbatch + 1
    train.acc <- train.acc + accuracy(label, out.pred)
    if (nbatch %% 100 == 0) {
      print(paste("Train-acc=", train.acc / nbatch))
      print(proc.time() - tic)
    }
  }
  dtrain$reset()
  print(paste("Train-acc=", train.acc / nbatch))
}
