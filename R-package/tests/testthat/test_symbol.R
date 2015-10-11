require(mxnet)

context("symbol")

test_that("basic symbol operation", {
  data = mx.symbol.Variable('data')
  net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
  net1 = mx.symbol.FullyConnected(data=net1, name='fc2', num_hidden=100)
  
  expect_equal(arguments(net1), c('data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias'))
  
  net2 = mx.symbol.FullyConnected(name='fc3', num_hidden=10)
  net2 = mx.symbol.Activation(data=net2, act_type='relu')
  net2 = mx.symbol.FullyConnected(data=net2, name='fc4', num_hidden=20)
  
  composed = mx.apply(net2, fc3_data=net1, name='composed')
  
  expect_equal(arguments(composed), c('data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'fc3_weight', 'fc3_bias', 'fc4_weight', 'fc4_bias'))
})


