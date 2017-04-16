library(mxnet)

get_symbol <- function(num_classes = 1000) {
  input_data <- mx.symbol.Variable(name = "data")
  # stage 1
  conv1 <- mx.symbol.Convolution(data = input_data, kernel = c(11, 11), stride = c(4, 4), num_filter = 96)
  relu1 <- mx.symbol.Activation(data = conv1, act_type = "relu")
  pool1 <- mx.symbol.Pooling(data = relu1, pool_type = "max", kernel = c(3, 3), stride = c(2, 2))
  lrn1 <- mx.symbol.LRN(data = pool1, alpha = 0.0001, beta = 0.75, knorm = 1, nsize = 5)
  # stage 2
  conv2 <- mx.symbol.Convolution(data = lrn1, kernel = c(5, 5), pad = c(2, 2), num_filter = 256)
  relu2 <- mx.symbol.Activation(data = conv2, act_type = "relu")
  pool2 <- mx.symbol.Pooling(data = relu2, kernel = c(3, 3), stride = c(2, 2), pool_type = "max")
  lrn2 <- mx.symbol.LRN(data = pool2, alpha = 0.0001, beta = 0.75, knorm = 1, nsize = 5)
  # stage 3
  conv3 <- mx.symbol.Convolution(data = lrn2, kernel = c(3, 3), pad = c(1, 1), num_filter = 384)
  relu3 <- mx.symbol.Activation(data = conv3, act_type = "relu")
  conv4 <- mx.symbol.Convolution(data = relu3, kernel = c(3, 3), pad = c(1, 1), num_filter = 384)
  relu4 <- mx.symbol.Activation(data = conv4, act_type = "relu")
  conv5 <- mx.symbol.Convolution(data = relu4, kernel = c(3, 3), pad = c(1, 1), num_filter = 256)
  relu5 <- mx.symbol.Activation(data = conv5, act_type = "relu")
  pool3 <- mx.symbol.Pooling(data = relu5, kernel = c(3, 3), stride = c(2, 2), pool_type = "max")
  # stage 4
  flatten <- mx.symbol.Flatten(data = pool3)
  fc1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 4096)
  relu6 <- mx.symbol.Activation(data = fc1, act_type = "relu")
  dropout1 <- mx.symbol.Dropout(data = relu6, p = 0.5)
  # stage 5
  fc2 <- mx.symbol.FullyConnected(data = dropout1, num_hidden = 4096)
  relu7 <- mx.symbol.Activation(data = fc2, act_type = "relu")
  dropout2 <- mx.symbol.Dropout(data = relu7, p = 0.5)
  # stage 6
  fc3 <- mx.symbol.FullyConnected(data = dropout2, num_hidden = num_classes)
  softmax <- mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
  return(softmax)
}
