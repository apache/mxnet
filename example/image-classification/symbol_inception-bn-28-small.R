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

# Basic Conv + BN + ReLU factory
ConvFactory <- function(data, num_filter, kernel, stride = c(1,1),
                        pad = c(0, 0), act_type = "relu") {
  conv = mx.symbol.Convolution(
    data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad =
      pad
  )
  bn = mx.symbol.BatchNorm(data = conv)
  act = mx.symbol.Activation(data = bn, act_type = act_type)
  return(act)
}

# A Simple Downsampling Factory
DownsampleFactory <- function(data, ch_3x3) {
  # conv 3x3
  conv = ConvFactory(
    data = data, kernel = c(3, 3), stride = c(2, 2), num_filter = ch_3x3, pad =
      c(1, 1)
  )
  
  # pool
  pool = mx.symbol.Pooling(
    data = data, kernel = c(3, 3), stride = c(2, 2), pad = c(1, 1), pool_type =
      'max'
  )
  # concat
  concat = mx.symbol.Concat(c(conv, pool), num.args = 2)
  return(concat)
}

# A Simple module
SimpleFactory <- function(data, ch_1x1, ch_3x3) {
  # 1x1
  conv1x1 = ConvFactory(
    data = data, kernel = c(1, 1), pad = c(0, 0), num_filter = ch_1x1
  )
  # 3x3
  conv3x3 = ConvFactory(
    data = data, kernel = c(3, 3), pad = c(1, 1), num_filter = ch_3x3
  )
  #concat
  concat = mx.symbol.Concat(c(conv1x1, conv3x3), num.args = 2)
  return(concat)
}

get_symbol <- function(num_classes = 10) {
  data = mx.symbol.Variable(name = "data")
  conv1 = ConvFactory(
    data = data, kernel = c(3,3), pad = c(1,1), num_filter = 96,
    act_type = "relu"
  )
  in3a = SimpleFactory(conv1, 32, 32)
  in3b = SimpleFactory(in3a, 32, 48)
  in3c = DownsampleFactory(in3b, 80)
  in4a = SimpleFactory(in3c, 112, 48)
  in4b = SimpleFactory(in4a, 96, 64)
  in4c = SimpleFactory(in4b, 80, 80)
  in4d = SimpleFactory(in4c, 48, 96)
  in4e = DownsampleFactory(in4d, 96)
  in5a = SimpleFactory(in4e, 176, 160)
  in5b = SimpleFactory(in5a, 176, 160)
  pool = mx.symbol.Pooling(
    data = in5b, pool_type = "avg", kernel = c(7,7), name = "global_pool"
  )
  flatten = mx.symbol.Flatten(data = pool, name = "flatten1")
  fc = mx.symbol.FullyConnected(data = flatten, num_hidden = num_classes, name =
                                  "fc1")
  softmax = mx.symbol.SoftmaxOutput(data = fc, name = "softmax")
  return(softmax)
}