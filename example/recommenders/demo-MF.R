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
DF <- read.table("./ml-100k/u.data", header = F, sep = "\t")
names(DF) <- c("user", "item", "score", "time")
max_user <- max(DF$user)
max_item <- max(DF$item)
DF_mat_x <- data.matrix(t(DF[, 1:2]))
DF_y <- DF[, 3]
k <- 64
user <- mx.symbol.Variable("user")
item <- mx.symbol.Variable("item")
score <- mx.symbol.Variable("label")
user1 <-mx.symbol.Embedding(data = mx.symbol.BlockGrad(user), input_dim = max_user,
                            output_dim = k, name = "user1")
item1 <- mx.symbol.Embedding(data = mx.symbol.BlockGrad(item), input_dim = max_item,
                             output_dim = k, name = "item1")
pred <- user1 * item1
pred1 <- mx.symbol.sum_axis(pred, axis = 1, name = "pred1")
pred2 <- mx.symbol.Flatten(pred1, name = "pred2")
pred3 <- mx.symbol.LinearRegressionOutput(data = pred2, label = score, name = "pred3")
devices <- mx.cpu()
mx.set.seed(123)

CustomIter <- setRefClass("CustomIter", fields = c("iter1", "iter2"),
  contains = "Rcpp_MXArrayDataIter",
  methods = list(
    initialize = function(iter1, iter2) {
      .self$iter1 <- iter1
      .self$iter2 <- iter2
      .self
    },
    value = function() {
      user <- .self$iter1$value()$data
      item <- .self$iter2$value()$data
      label <- .self$iter1$value()$label
      list(user = user,
           item = item,
           label = label)
    },
    iter.next = function() {
      .self$iter1$iter.next()
      .self$iter2$iter.next()
    },
    reset = function() {
      .self$iter1$reset()
      .self$iter2$reset()
    },
    num.pad = function() {
      .self$iter1$num.pad()
    },
    finalize = function() {
      .self$iter1$finalize()
      .self$iter2$finalize()
    }
  )
)

user_iter = mx.io.arrayiter(data = DF[, 1], label = DF[, 3], batch.size = k)

item_iter = mx.io.arrayiter(data = DF[, 2], label = DF[, 3], batch.size = k)

train_iter <- CustomIter$new(user_iter, item_iter)

model <- mx.model.FeedForward.create(pred3, X = train_iter, ctx = devices,
                                     num.round = 10, initializer = mx.init.uniform(0.07),
                                     learning.rate = 0.07, eval.metric = mx.metric.rmse,
                                     momentum = 0.9, epoch.end.callback = mx.callback.log.train.metric(1),
                                     input.names = c("user", "item"), output.names = "label")
