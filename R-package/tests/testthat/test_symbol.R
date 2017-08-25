require(mxnet)

context("symbol")

test_that("basic symbol operation", {
  data = mx.symbol.Variable('data')
  net1 = mx.symbol.FullyConnected(data = data, name = 'fc1', num_hidden = 10)
  net1 = mx.symbol.FullyConnected(data = net1, name = 'fc2', num_hidden = 100)
  
  expect_equal(arguments(net1), c('data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias'))
  expect_equal(outputs(net1), 'fc2_output')
  
  net2 = mx.symbol.FullyConnected(name = 'fc3', num_hidden = 10)
  net2 = mx.symbol.Activation(data = net2, act_type = 'relu')
  net2 = mx.symbol.FullyConnected(data = net2, name = 'fc4', num_hidden = 20)
  
  composed = mx.apply(net2, fc3_data = net1, name = 'composed')
  
  expect_equal(arguments(composed), c('data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'fc3_weight', 'fc3_bias', 'fc4_weight', 'fc4_bias'))
  expect_equal(outputs(composed), 'composed_output')
  
  multi_out = mx.symbol.Group(c(composed, net1))
  expect_equal(outputs(multi_out), c('composed_output', 'fc2_output'))
})

test_that("symbol internal", {
  data = mx.symbol.Variable('data')
  oldfc = mx.symbol.FullyConnected(data = data, name = 'fc1', num_hidden = 10)
  net1 = mx.symbol.FullyConnected(data = oldfc, name = 'fc2', num_hidden = 100)
  
  expect_equal(arguments(net1), c("data", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"))
  
  internal = internals(net1)
  fc1 = internal[[match("fc1_output", internal$outputs)]]
  
  expect_equal(arguments(fc1), arguments(oldfc))
})

test_that("symbol children", {
  data = mx.symbol.Variable('data')
  oldfc = mx.symbol.FullyConnected(data = data,
                                   name = 'fc1',
                                   num_hidden = 10)
  net1 = mx.symbol.FullyConnected(data = oldfc, name = 'fc2', num_hidden = 100)
  
  expect_equal(outputs(children(net1)), c('fc1_output', 'fc2_weight', 'fc2_bias'))
  expect_equal(outputs(children(children(net1))), c('data', 'fc1_weight', 'fc1_bias'))
  
  net2 = net1$get.children()
  expect_equal(net2[[match('fc2_weight', net2$outputs)]]$arguments, 'fc2_weight')
  
  data = mx.symbol.Variable('data')
  sliced = mx.symbol.SliceChannel(data, num_outputs = 3, name = 'slice')
  expect_equal(outputs(children(sliced)), 'data')
})

test_that("symbol infer type", {
  num_hidden = 128
  num_dim    = 64
  num_sample = 10
  
  data = mx.symbol.Variable('data')
  prev = mx.symbol.Variable('prevstate')
  x2h  = mx.symbol.FullyConnected(data = data, name = 'x2h', num_hidden = num_hidden)
  h2h  = mx.symbol.FullyConnected(data = prev, name = 'h2h', num_hidden = num_hidden)
  
  out  = mx.symbol.Activation(data = mx.symbol.elemwise_add(x2h, h2h), name = 'out', act_type = 'relu')
  
  # shape inference will fail because information is not available for h2h
  ret = mx.symbol.infer.shape(out, data = c(num_dim, num_sample))
  
  expect_equal(ret, NULL)
})

test_that("symbol save/load", {
  data <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data, num_hidden = 1)
  lro <- mx.symbol.LinearRegressionOutput(fc1)
  mx.symbol.save(lro, "tmp_r_sym.json")
  data2 = mx.symbol.load("tmp_r_sym.json")
  
  expect_equal(data2$as.json(), lro$as.json())
  file.remove("tmp_r_sym.json")
})

test_that("symbol attributes access", {
  str <- "(1, 1, 1, 1)"
  x = mx.symbol.Variable('x')
  x$attributes <- list(`__shape__` = str)
  
  expect_equal(x$attributes$`__shape__`, str)
  
  y = mx.symbol.Variable('y')
  y$attributes$`__shape__` <- str
  
  expect_equal(y$attributes$`__shape__`, str)
})

test_that("symbol concat", {
  s1 <- mx.symbol.Variable("data1")
  s2 <- mx.symbol.Variable("data2")
  s3 <- mx.symbol.concat(data = c(s1, s2), num.args = 2, name = "concat")
  expect_equal(outputs(s3), "concat_output")
  expect_equal(outputs(children(s3)), c("data1", "data2"))
  expect_equal(arguments(s3), c("data1", "data2"))
  
  s4 <- mx.symbol.Concat(data = c(s1, s2), num.args = 2, name = "concat")
  expect_equal(outputs(s3), outputs(s4))
  expect_equal(outputs(children(s3)), outputs(children(s4)))
  expect_equal(arguments(s3), arguments(s4))
})
