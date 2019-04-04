# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

context("optimizer")

test_that("sgd", {
  
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  fc_weight <- mx.symbol.Variable("fc_weight")
  fc <- mx.symbol.FullyConnected(data = data, weight = fc_weight, no.bias = T, 
    name = "fc1", num_hidden = 1)
  loss <- mx.symbol.LinearRegressionOutput(data = fc, label = label, name = "loss")
  
  x <- mx.nd.array(array(1:6, dim = 2:3))
  y <- mx.nd.array(c(5, 11, 16))
  w1 <- mx.nd.array(array(c(1.1, 1.8), dim = c(2, 1)))
  
  exec <- mxnet:::mx.symbol.bind(symbol = loss, ctx = mx.cpu(), arg.arrays = list(data = x, 
    fc1_weight = w1, label = y), aux.arrays = NULL, grad.reqs = c("null", "write", 
    "null"))
  
  optimizer <- mx.opt.create("sgd", learning.rate = 1, momentum = 0, wd = 0, rescale.grad = 1, 
    clip_gradient = -1)
  
  updaters <- mx.opt.get.updater(optimizer, exec$ref.arg.arrays, ctx = mx.cpu())
  
  mx.exec.forward(exec, is.train = T)
  mx.exec.backward(exec)
  
  arg.blocks <- updaters(exec$ref.arg.arrays, exec$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec, arg.blocks, skip.null = TRUE)
  
  expect_equal(as.array(arg.blocks[[2]]), array(c(1.4, 2.6), dim = c(2, 1)), tolerance = 0.1)
  
})


test_that("rmsprop", {
  
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  fc_weight <- mx.symbol.Variable("fc_weight")
  fc <- mx.symbol.FullyConnected(data = data, weight = fc_weight, no.bias = T, 
    name = "fc1", num_hidden = 1)
  loss <- mx.symbol.LinearRegressionOutput(data = fc, label = label, name = "loss")
  
  x <- mx.nd.array(array(1:6, dim = 2:3))
  y <- mx.nd.array(c(5, 11, 16))
  w1 <- mx.nd.array(array(c(1.1, 1.8), dim = c(2, 1)))
  
  exec <- mxnet:::mx.symbol.bind(symbol = loss, ctx = mx.cpu(), arg.arrays = list(data = x, 
    fc1_weight = w1, label = y), aux.arrays = NULL, grad.reqs = c("null", "write", 
    "null"))
  
  optimizer <- mx.opt.create("rmsprop", learning.rate = 1, centered = TRUE, gamma1 = 0.95, 
    gamma2 = 0.9, epsilon = 1e-04, wd = 0, rescale.grad = 1, clip_gradient = -1)
  
  updaters <- mx.opt.get.updater(optimizer, exec$ref.arg.arrays, ctx = mx.cpu())
  
  mx.exec.forward(exec, is.train = T)
  mx.exec.backward(exec)
  
  arg.blocks <- updaters(exec$ref.arg.arrays, exec$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec, arg.blocks, skip.null = TRUE)
  
  expect_equal(as.array(arg.blocks[[2]]), array(c(5.64, 6.38), dim = c(2, 1)), 
    tolerance = 0.1)
  
})


test_that("adam", {
  
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  fc_weight <- mx.symbol.Variable("fc_weight")
  fc <- mx.symbol.FullyConnected(data = data, weight = fc_weight, no.bias = T, 
    name = "fc1", num_hidden = 1)
  loss <- mx.symbol.LinearRegressionOutput(data = fc, label = label, name = "loss")
  
  x <- mx.nd.array(array(1:6, dim = 2:3))
  y <- mx.nd.array(c(5, 11, 16))
  w1 <- mx.nd.array(array(c(1.1, 1.8), dim = c(2, 1)))
  
  exec <- mxnet:::mx.symbol.bind(symbol = loss, ctx = mx.cpu(), arg.arrays = list(data = x, 
    fc1_weight = w1, label = y), aux.arrays = NULL, grad.reqs = c("null", "write", 
    "null"))
  
  optimizer <- mx.opt.create("adam", learning.rate = 1, beta1 = 0.9, beta2 = 0.999, 
    epsilon = 1e-08, wd = 0, rescale.grad = 1, clip_gradient = -1)
  
  updaters <- mx.opt.get.updater(optimizer, exec$ref.arg.arrays, ctx = mx.cpu())
  
  mx.exec.forward(exec, is.train = T)
  mx.exec.backward(exec)
  
  arg.blocks <- updaters(exec$ref.arg.arrays, exec$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec, arg.blocks, skip.null = TRUE)
  
  expect_equal(as.array(arg.blocks[[2]]), array(c(4.26, 4.96), dim = c(2, 1)), 
    tolerance = 0.1)
  
})


test_that("adagrad", {
  
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  fc_weight <- mx.symbol.Variable("fc_weight")
  fc <- mx.symbol.FullyConnected(data = data, weight = fc_weight, no.bias = T, 
    name = "fc1", num_hidden = 1)
  loss <- mx.symbol.LinearRegressionOutput(data = fc, label = label, name = "loss")
  
  x <- mx.nd.array(array(1:6, dim = 2:3))
  y <- mx.nd.array(c(5, 11, 16))
  w1 <- mx.nd.array(array(c(1.1, 1.8), dim = c(2, 1)))
  
  exec <- mxnet:::mx.symbol.bind(symbol = loss, ctx = mx.cpu(), arg.arrays = list(data = x, 
    fc1_weight = w1, label = y), aux.arrays = NULL, grad.reqs = c("null", "write", 
    "null"))
  
  optimizer <- mx.opt.create("adagrad", learning.rate = 1, epsilon = 1e-08, wd = 0, 
    rescale.grad = 1, clip_gradient = -1)
  
  updaters <- mx.opt.get.updater(optimizer, exec$ref.arg.arrays, ctx = mx.cpu())
  
  mx.exec.forward(exec, is.train = T)
  mx.exec.backward(exec)
  
  arg.blocks <- updaters(exec$ref.arg.arrays, exec$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec, arg.blocks, skip.null = TRUE)
  
  expect_equal(as.array(arg.blocks[[2]]), array(c(2.1, 2.8), dim = c(2, 1)), tolerance = 0.1)
  
})


test_that("adadelta", {
  
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  fc_weight <- mx.symbol.Variable("fc_weight")
  fc <- mx.symbol.FullyConnected(data = data, weight = fc_weight, no.bias = T, 
    name = "fc1", num_hidden = 1)
  loss <- mx.symbol.LinearRegressionOutput(data = fc, label = label, name = "loss")
  
  x <- mx.nd.array(array(1:6, dim = 2:3))
  y <- mx.nd.array(c(5, 11, 16))
  w1 <- mx.nd.array(array(c(1.1, 1.8), dim = c(2, 1)))
  
  exec <- mxnet:::mx.symbol.bind(symbol = loss, ctx = mx.cpu(), arg.arrays = list(data = x, 
    fc1_weight = w1, label = y), aux.arrays = NULL, grad.reqs = c("null", "write", 
    "null"))
  
  optimizer <- mx.opt.create("adadelta", rho = 0.9, epsilon = 1e-05, wd = 0, rescale.grad = 1, 
    clip_gradient = -1)
  
  updaters <- mx.opt.get.updater(optimizer, exec$ref.arg.arrays, ctx = mx.cpu())
  
  mx.exec.forward(exec, is.train = T)
  mx.exec.backward(exec)
  
  arg.blocks <- updaters(exec$ref.arg.arrays, exec$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec, arg.blocks, skip.null = TRUE)
  
  expect_equal(as.array(arg.blocks[[2]]), array(c(1.11, 1.81), dim = c(2, 1)), 
    tolerance = 0.1)
  
})
