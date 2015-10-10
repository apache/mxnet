require(mxnet)
# To run this, run python/mxnet/test_io.py to get data first
iter = mx.varg.io.MNISTIter(list(
  image="data/train-images-idx3-ubyte",
  label="data/train-labels-idx1-ubyte",
  data.shape=c(784),
  batch.size=3,
  shuffle=TRUE,
  flat=TRUE,
  silent=0,
  seed=10))

iter$reset()
print(iter$iter.next())
data = iter$value()

print(as.array(data$label))
print(dim(data$data))
