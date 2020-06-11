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

require(mxnet)

source("get_data.R")

context("models")

if (Sys.getenv("R_GPU_ENABLE") != "" & as.integer(Sys.getenv("R_GPU_ENABLE")) == 
  1) {
  mx.ctx.default(new = mx.gpu())
  message("Using GPU for testing.")
}

test_that("Classification", {
  data(Sonar, package = "mlbench")
  Sonar[, 61] <- as.numeric(Sonar[, 61]) - 1
  train.ind <- c(1:50, 100:150)
  train.x <- data.matrix(Sonar[train.ind, 1:60])
  train.y <- Sonar[train.ind, 61]
  test.x <- data.matrix(Sonar[-train.ind, 1:60])
  test.y <- Sonar[-train.ind, 61]
  mx.set.seed(0)
  model <- mx.mlp(train.x, train.y, hidden_node = 10, out_node = 2, out_activation = "softmax", 
    num.round = 5, array.batch.size = 15, learning.rate = 0.07, momentum = 0.9, 
    eval.metric = mx.metric.accuracy)
})
