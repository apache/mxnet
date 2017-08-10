library(mxnet)

get_symbol <- function(num_classes = 1000) {
  data <- mx.symbol.Variable('data')
  fc1 <- mx.symbol.FullyConnected(data = data, name = 'fc1', num_hidden = 128)
  act1 <- mx.symbol.Activation(data = fc1, name = 'relu1', act_type = "relu")
  fc2 <- mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
  act2 <- mx.symbol.Activation(data = fc2, name = 'relu2', act_type = "relu")
  fc3 <- mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = num_classes)
  mlp <- mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
  return(mlp)
}
