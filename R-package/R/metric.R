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

#' Helper function to create a customized metric
#'
#' @export
mx.metric.custom <- function(name, feval) {
  init <- function() {
    c(0, 0)
  }
  update <- function(label, pred, state) {
    m <- feval(label, pred)
    state <- c(state[[1]] + 1, state[[2]] + m)
    return(state)
  }
  get <- function(state) {
    list(name=name, value=(state[[2]]/state[[1]]))
  }
  ret <- (list(init=init, update=update, get=get))
  class(ret) <- "mx.metric"
  return(ret)
}

#' Accuracy metric for classification
#'
#' @export
mx.metric.accuracy <- mx.metric.custom("accuracy", function(label, pred) {
  pred <- mx.nd.argmax(data = pred, axis = 1, keepdims = F)
  res <- mx.nd.mean(label == pred)
  return(as.array(res))
})

#' Top-k accuracy metric for classification
#'
#' @export
mx.metric.top_k_accuracy <- mx.metric.custom("top_k_accuracy", function(label, pred, top_k = 5) {
  label <- mx.nd.reshape(data = label, shape = c(1,0))
  pred <- mx.nd.topk(data = pred, axis = 1, k = top_k, ret_typ = "indices")
  pred <- mx.nd.broadcast.equal(lhs = pred, rhs = label)
  res <- mx.nd.mean(mx.nd.sum(data = pred, axis = 1, keepdims = F))
  return(as.array(res))
})

#' MSE (Mean Squared Error) metric for regression
#'
#' @export
mx.metric.mse <- mx.metric.custom("mse", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  pred <- mx.nd.reshape(pred, shape = -1)
  res <- mx.nd.mean(mx.nd.square(label-pred))
  return(as.array(res))
})

#' RMSE (Root Mean Squared Error) metric for regression
#'
#' @export
mx.metric.rmse <- mx.metric.custom("rmse", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  pred <- mx.nd.reshape(pred, shape = -1)
  res <- mx.nd.sqrt(mx.nd.mean(mx.nd.square(label-pred)))
  return(as.array(res))
})

#' MAE (Mean Absolute Error) metric for regression
#'
#' @export
mx.metric.mae <- mx.metric.custom("mae", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  pred <- mx.nd.reshape(pred, shape = -1)
  res <- mx.nd.mean(mx.nd.abs(label-pred))
  return(as.array(res))
})

#' RMSLE (Root Mean Squared Logarithmic Error) metric for regression
#'
#' @export
mx.metric.rmsle <- mx.metric.custom("rmsle", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  pred <- mx.nd.reshape(pred, shape = -1)
  res <- mx.nd.sqrt(mx.nd.mean(mx.nd.square(mx.nd.log1p(pred) - mx.nd.log1p(label))))
  return(as.array(res))
})

#' Perplexity metric for language model
#'
#' @export
mx.metric.Perplexity <- mx.metric.custom("Perplexity", function(label, pred, mask_element = -1) {

  label <- mx.nd.reshape(label, shape = -1)
  pred_probs <- mx.nd.pick(data = pred, index = label, axis = 1)

  mask <- label != mask_element
  mask_length <- mx.nd.sum(mask)

  NLL <- -mx.nd.sum(mx.nd.log(pred_probs) * mask) / mask_length
  res <- mx.nd.exp(NLL)
  return(as.array(res))
})

#' LogLoss metric for logistic regression
#'
#' @export
mx.metric.logloss <- mx.metric.custom("logloss", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  pred <- mx.nd.reshape(pred, shape = -1)
  pred <- mx.nd.clip(pred, a_min = 1e-15, a_max = 1-1e-15)
  res <- -mx.nd.mean(label * mx.nd.log(pred) + (1-label) * mx.nd.log(1-pred))
  return(as.array(res))
})

#' Accuracy metric for logistic regression
#'
#' @export
mx.metric.logistic_acc <- mx.metric.custom("accuracy", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  pred <- mx.nd.reshape(pred, shape = -1) > 0.5
  res <- mx.nd.mean(label == pred)
  return(as.array(res))
})

