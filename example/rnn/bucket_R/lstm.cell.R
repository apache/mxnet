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

# LSTM cell symbol
lstm.cell <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0,
  data_masking) {
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$i2h.weight, bias = param$i2h.bias,
    num.hidden = num.hidden * 4, name = paste0("t", seqidx, ".l", layeridx, ".i2h"))

  if (dropout > 0)
    i2h <- mx.symbol.Dropout(data = i2h, p = dropout)

  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$h2h.weight,
      bias = param$h2h.bias, num.hidden = num.hidden * 4, name = paste0("t",
        seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }

  split.gates <- mx.symbol.split(gates, num.outputs = 4, axis = 1, squeeze.axis = F,
    name = paste0("t", seqidx, ".l", layeridx, ".slice"))

  in.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  in.transform <- mx.symbol.Activation(split.gates[[2]], act.type = "tanh")
  forget.gate <- mx.symbol.Activation(split.gates[[3]], act.type = "sigmoid")
  out.gate <- mx.symbol.Activation(split.gates[[4]], act.type = "sigmoid")

  if (is.null(prev.state)) {
    next.c <- in.gate * in.transform
  } else {
    next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
  }

  next.h <- out.gate * mx.symbol.Activation(next.c, act.type = "tanh")

  ### Add a mask - using the mask_array approach
  data_mask_expand <- mx.symbol.Reshape(data = data_masking, shape = c(1, -2))
  next.c <- mx.symbol.broadcast_mul(lhs = next.c, rhs = data_mask_expand)
  next.h <- mx.symbol.broadcast_mul(lhs = next.h, rhs = data_mask_expand)

  return(list(c = next.c, h = next.h))
}
