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

ConvFactory <- function(data, num_filter, kernel, stride = c(1, 1), pad = c(0, 0),
                        name = '', suffix = '') {
    conv <- mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride,
                                  pad = pad, name = paste('conv_', name, suffix, sep = ""))
    act <- mx.symbol.Activation(data = conv, act_type = 'relu', name = paste('relu_', name, suffix, sep = ''))
    return(act)
}

InceptionFactory <- function(data, num_1x1, num_3x3red, num_3x3,
                             num_d5x5red, num_d5x5, pool, proj, name) {
    # 1x1
    c1x1 <- ConvFactory(data = data, num_filter = num_1x1, kernel = c(1, 1),
                        name = paste(name, '_1x1', sep = ''))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data = data, num_filter = num_3x3red, kernel = c(1, 1),
                        name = paste(name, '_3x3', sep = ''), suffix = '_reduce')
    c3x3 = ConvFactory(data = c3x3r, num_filter = num_3x3, kernel = c(3, 3),
                       pad = c(1, 1), name = paste(name, '_3x3', sep = ''))
    # double 3x3 reduce + double 3x3
    cd5x5r = ConvFactory(data = data, num_filter = num_d5x5red, kernel = c(1, 1),
                         name = paste(name, '_5x5', sep = ''), suffix = '_reduce')
    cd5x5 = ConvFactory(data = cd5x5r, num_filter = num_d5x5, kernel = c(5, 5), pad = c(2, 2),
                        name = paste(name, '_5x5', sep = ''))
    # pool + proj
    pooling = mx.symbol.Pooling(data = data, kernel = c(3, 3), stride = c(1, 1), 
                                pad = c(1, 1), pool_type = pool,
                                name = paste(pool, '_pool_', name, '_pool', sep = ''))

    cproj = ConvFactory(data = pooling, num_filter = proj, kernel = c(1, 1), 
                        name = paste(name, '_proj', sep = ''))
    # concat
    concat_lst <- list()
    concat_lst <- c(c1x1, c3x3, cd5x5, cproj)
    concat_lst$num.args = 4
    concat_lst$name = paste('ch_concat_', name, '_chconcat', sep = '')
    concat = mxnet:::mx.varg.symbol.Concat(concat_lst)
    return(concat)
}


get_symbol <- function(num_classes = 1000) {
  data <- mx.symbol.Variable("data")
  conv1 <- ConvFactory(data, 64, kernel = c(7, 7), stride = c(2, 2), pad = c(3, 3), name = "conv1")
  pool1 <- mx.symbol.Pooling(conv1, kernel = c(3, 3), stride = c(2, 2), pool_type = "max")
  conv2 <- ConvFactory(pool1, 64, kernel = c(1, 1), stride = c(1, 1), name = "conv2")
  conv3 <- ConvFactory(conv2, 192, kernel = c(3, 3), stride = c(1, 1), pad = c(1, 1), name = "conv3")
  pool3 <- mx.symbol.Pooling(conv3, kernel = c(3, 3), stride = c(2, 2), pool_type = "max")
  
  in3a <- InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name = "in3a")
  in3b <- InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name = "in3b")
  pool4 <- mx.symbol.Pooling(in3b, kernel = c(3, 3), stride = c(2, 2), pool_type = "max")
  in4a <- InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name = "in4a")
  in4b <- InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name = "in4b")
  in4c <- InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name = "in4c")
  in4d <- InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name = "in4d")
  in4e <- InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name = "in4e")
  pool5 <- mx.symbol.Pooling(in4e, kernel = c(3, 3), stride = c(2, 2), pool_type = "max")
  in5a <- InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name = "in5a")
  in5b <- InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name = "in5b")
  pool6 <- mx.symbol.Pooling(in5b, kernel = c(7, 7), stride = c(1, 1), pool_type = "avg" )
  flatten <- mx.symbol.Flatten(data = pool6, name = 'flatten0')
  fc1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = num_classes)
  softmax <- mx.symbol.SoftmaxOutput(data = fc1, name = 'softmax')
  return(softmax)
}
