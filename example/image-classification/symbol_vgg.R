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

library(mxnet)

get_symbol <- function(num_classes = 1000) {
  ## define alexnet
  data = mx.symbol.Variable(name = "data")
  # group 1
  conv1_1 = mx.symbol.Convolution(data = data, kernel = c(3, 3), pad = c(1, 1),
                                  num_filter = 64, name = "conv1_1")
  relu1_1 = mx.symbol.Activation(data = conv1_1, act_type = "relu", name = "relu1_1")
  pool1 = mx.symbol.Pooling(data = relu1_1, pool_type = "max", kernel = c(2, 2),
                            stride = c(2, 2), name = "pool1")
  # group 2
  conv2_1 = mx.symbol.Convolution(data = pool1, kernel = c(3, 3), pad = c(1, 1),
                                  num_filter = 128, name = "conv2_1")
  relu2_1 = mx.symbol.Activation(data = conv2_1, act_type = "relu", name = "relu2_1")
  pool2 = mx.symbol.Pooling(data = relu2_1, pool_type = "max", kernel = c(2, 2),
                            stride = c(2, 2), name = "pool2")
  # group 3
  conv3_1 = mx.symbol.Convolution(data = pool2, kernel = c(3, 3), pad = c(1, 1),
                                  num_filter = 256, name = "conv3_1")
  relu3_1 = mx.symbol.Activation(data = conv3_1, act_type = "relu", name = "relu3_1")
  conv3_2 = mx.symbol.Convolution(data = relu3_1, kernel = c(3, 3), pad = c(1, 1),
                                  num_filter = 256, name = "conv3_2")
  relu3_2 = mx.symbol.Activation(data = conv3_2, act_type = "relu", name = "relu3_2")
  pool3 = mx.symbol.Pooling(data = relu3_2, pool_type = "max", kernel = c(2, 2),
                            stride = c(2, 2), name = "pool3")
  # group 4
  conv4_1 = mx.symbol.Convolution(data = pool3, kernel = c(3, 3), pad = c(1, 1),
                                  num_filter = 512, name = "conv4_1")
  relu4_1 = mx.symbol.Activation(data = conv4_1, act_type = "relu", name = "relu4_1")
  conv4_2 = mx.symbol.Convolution(data = relu4_1, kernel = c(3, 3), pad = c(1, 1),
                                  num_filter = 512, name = "conv4_2")
  relu4_2 = mx.symbol.Activation(data = conv4_2, act_type = "relu", name = "relu4_2")
  pool4 = mx.symbol.Pooling(data = relu4_2, pool_type = "max",
                            kernel = c(2, 2), stride = c(2, 2), name = "pool4")
  # group 5
  conv5_1 = mx.symbol.Convolution(data = pool4, kernel = c(3, 3),
                                  pad = c(1, 1), num_filter = 512, name = "conv5_1")
  relu5_1 = mx.symbol.Activation(data = conv5_1, act_type = "relu", name = "relu5_1")
  conv5_2 = mx.symbol.Convolution(data = relu5_1, kernel = c(3, 3),
                                  pad = c(1, 1), num_filter = 512, name = "conv5_2")
  relu5_2 = mx.symbol.Activation(data = conv5_2, act_type = "relu", name = "relu5_2")
  pool5 = mx.symbol.Pooling(data = relu5_2, pool_type = "max",
                            kernel = c(2, 2), stride = c(2, 2), name = "pool5")
  # group 6
  flatten = mx.symbol.Flatten(data = pool5, name = "flatten")
  fc6 = mx.symbol.FullyConnected(data = flatten, num_hidden = 4096, name = "fc6")
  relu6 = mx.symbol.Activation(data = fc6, act_type = "relu", name = "relu6")
  drop6 = mx.symbol.Dropout(data = relu6, p = 0.5, name = "drop6")
  # group 7
  fc7 = mx.symbol.FullyConnected(data = drop6, num_hidden = 4096, name = "fc7")
  relu7 = mx.symbol.Activation(data = fc7, act_type = "relu", name = "relu7")
  drop7 = mx.symbol.Dropout(data = relu7, p = 0.5, name = "drop7")
  # output
  fc8 = mx.symbol.FullyConnected(data = drop7, num_hidden = num_classes, name = "fc8")
  softmax = mx.symbol.SoftmaxOutput(data = fc8, name = 'softmax')
  return(softmax)
}
