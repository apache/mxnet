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

get_conv <- function(name, data, num_filter, kernel, stride,
                     pad, with_relu, bn_momentum) {
  conv = mx.symbol.Convolution(name = name, data = data, num_filter = num_filter,
                               kernel = kernel, stride = stride, pad = pad, no_bias = TRUE)
  bn = mx.symbol.BatchNorm(name = paste(name, '_bn', sep = ''), data = conv,
                           fix_gamma = FALSE, momentum = bn_momentum, eps = 2e-5)
  if (with_relu) {
    return(mx.symbol.Activation(name = paste(name, '_relu', sep = ''),
                                data = bn, act_type = 'relu'))
  } else {
    return(bn)
  }
}

make_block <- function(name, data, num_filter, dim_match, bn_momentum) {
  if (dim_match) {
    conv1 = get_conv(name = paste(name, '_conv1', sep = ''), data = data,
                     num_filter = num_filter, kernel = c(3, 3), stride = c(1, 1),
                     pad = c(1, 1), with_relu = TRUE, bn_momentum = bn_momentum)
  } else {
    conv1 = get_conv(name = paste(name, '_conv1', sep = ''), data = data,
                     num_filter = num_filter, kernel = c(3, 3), stride = c(2, 2),
                     pad = c(1, 1), with_relu = TRUE, bn_momentum = bn_momentum)
  }
  
  conv2 = get_conv(name = paste(name, '_conv2', sep = ''), data = conv1,
                   num_filter = num_filter, kernel = c(3, 3), stride = c(1, 1),
                   pad = c(1, 1), with_relu = FALSE, bn_momentum = bn_momentum)
  if (dim_match) {
    shortcut = data
  } else {
    shortcut = mx.symbol.Convolution(name = paste(name, '_proj', sep = ''),
                                     data = data, num_filter = num_filter, kernel = c(2, 2), 
                                     stride = c(2, 2), pad = c(0, 0), no_bias = TRUE)
  }
  fused = shortcut + conv2
  return(mx.symbol.Activation(name = paste(name, '_relu', sep = ''), data = fused, act_type = 'relu'))
}

get_body <- function(data, num_level, num_block, num_filter, bn_momentum) {
  for (level in 1:num_level) {
    for (block in 1:num_block) {
      data = make_block(
        name = paste('level', level, '_block', block, sep = ''),
        data = data,
        num_filter = num_filter * 2 ^ (level - 1),
        dim_match = (level == 1 || block > 1),
        bn_momentum = bn_momentum
      )
    }
  }
  return(data)
}

get_symbol <- function(num_class, num_level = 3, num_block = 9,
                       num_filter = 16, bn_momentum = 0.9, pool_kernel = c(8, 8)) {
  data = mx.symbol.Variable(name = 'data')
  zscore = mx.symbol.BatchNorm(name = 'zscore', data = data, 
                               fix_gamma = TRUE, momentum = bn_momentum)
  conv = get_conv(name = 'conv0', data = zscore, num_filter = num_filter,
                  kernel = c(3, 3), stride = c(1, 1), pad = c(1, 1),
                  with_relu = TRUE, bn_momentum = bn_momentum)
  body = get_body(conv, num_level, num_block, num_filter, bn_momentum)
  pool = mx.symbol.Pooling(data = body, kernel = pool_kernel, pool_type = 'avg')
  flat = mx.symbol.Flatten(data = pool)
  fc = mx.symbol.FullyConnected(data = flat, num_hidden = num_class, name = 'fc')
  return(mx.symbol.SoftmaxOutput(data = fc, name = 'softmax'))
}
