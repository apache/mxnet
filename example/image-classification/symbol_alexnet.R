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
  input_data <- mx.symbol.Variable(name = "data")
  # stage 1
  conv1 <- mx.symbol.Convolution(data = input_data, kernel = c(11, 11), stride = c(4, 4), num_filter = 96)
  relu1 <- mx.symbol.Activation(data = conv1, act_type = "relu")
  lrn1 <- mx.symbol.LRN(data = relu1, alpha = 0.0001, beta = 0.75, knorm = 2, nsize = 5)
  pool1 <- mx.symbol.Pooling(data = lrn1, kernel = c(3, 3), stride = c(2, 2), pool_type = "max")
  # stage 2
  conv2 <- mx.symbol.Convolution(data = lrn1, kernel = c(5, 5), pad = c(2, 2), num_filter = 256)
  relu2 <- mx.symbol.Activation(data = conv2, act_type = "relu")
  lrn2 <- mx.symbol.LRN(data = relu2, alpha = 0.0001, beta = 0.75, knorm = 2, nsize = 5)
  pool2 <- mx.symbol.Pooling(data = lrn2, kernel = c(3, 3), stride = c(2, 2), pool_type = "max")  
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
