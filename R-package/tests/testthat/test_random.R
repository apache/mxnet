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

context("random")

test_that("mx.runif", {
  X <- mx.runif(shape = 50000, min = 0, max = 1, ctx = mx.ctx.default())
  expect_equal(X >= 0, mx.nd.ones(50000))
  expect_equal(X <= 1, mx.nd.ones(50000))
  sample_mean <- mean(as.array(X))
  expect_equal(sample_mean, 0.5, tolerance = 0.01)
})

test_that("mx.rnorm", {
  X <- mx.rnorm(shape = 50000, mean = 5, sd = 0.1, ctx = mx.ctx.default())
  sample_mean <- mean(as.array(X))
  sample_sd <- sd(as.array(X))
  expect_equal(sample_mean, 5, tolerance = 0.01)
  expect_equal(sample_sd, 0.1, tolerance = 0.01)
})
