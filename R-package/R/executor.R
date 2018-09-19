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

#' Simple bind the symbol to executor,
#' with information from input shapes.
#'
#' @export
mx.simple.bind <- function(symbol, ctx, grad.req = "null", fixed.param = NULL, ...) {
  if (!is.MXSymbol(symbol)) stop("symbol need to be MXSymbol")
  slist <- symbol$infer.shape(list(...))

  if (is.null(slist)) {
    stop("Need more shape information to decide the shapes of arguments")
  }
  arg.arrays <- sapply(slist$arg.shapes, function(shape) {
    mx.nd.zeros(shape, ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  aux.arrays <- sapply(slist$aux.shapes, function(shape) {
    mx.nd.zeros(shape, ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  grad.reqs <- lapply(names(slist$arg.shapes), function(nm) {
    if (nm %in% fixed.param) {
      "null"
    } else if (!endsWith(nm, "label") && !endsWith(nm, "data")) {
      grad.req
    } else {
      "null"
    }
  })
  mx.symbol.bind(symbol, ctx,
                 arg.arrays=arg.arrays,
                 aux.arrays=aux.arrays,
                 grad.reqs = grad.reqs)
}

#' Update the executors with new arrays
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.update.arg.arrays <- function(exec, arg.arrays, match.name=FALSE, skip.null=FALSE) {
  exec$update.arg.arrays(arg.arrays, match.name, skip.null)
}

#' Update the executors with new arrays
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.update.aux.arrays <- function(exec, arg.arrays, match.name=FALSE, skip.null=FALSE) {
  exec$update.aux.arrays(arg.arrays, match.name, skip.null)
}

#' Update the executors with new arrays
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.update.grad.arrays <- function(exec, arg.arrays, match.name=FALSE, skip.null=FALSE) {
  exec$update.grad.arrays(arg.arrays, match.name, skip.null)
}


#' Peform an forward on the executors
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.forward <- function(exec, is.train=TRUE) {
  exec$forward(is.train, list())
}

#' Peform an backward on the executors
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.backward <- function(exec, ...) {
  exec$backward(list(...))
}
