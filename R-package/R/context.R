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

# Initialize the global context
init.context.default <- function() {
  .MXNetEnv[["mx.ctx.internal.default.value"]] <- mx.cpu()
}

#' Set/Get default context for array creation.
#'
#' @param new optional takes \code{mx.cpu()} or \code{mx.gpu(id)}, new default ctx.
#' @return The default context.
#'
#' @export
mx.ctx.default <- function(new = NULL) {
  if (!is.null(new)) {
  	.MXNetEnv[["mx.ctx.internal.default.value"]] <- new
  }
  return (.MXNetEnv$mx.ctx.internal.default.value)
}

#' Check if the type is mxnet context.
#'
#' @return Logical indicator
#'
#' @export
is.mx.context <- function(x) {
  class(x) == "MXContext"
}


#' Create a mxnet CPU context.
#'
#' @param dev.id optional, default=0
#'     The device ID, this is meaningless for CPU, included for interface compatiblity.
#' @return The CPU context.
#' @name mx.cpu
#'
#' @export
NULL

#' Create a mxnet GPU context.
#'
#' @param dev.id optional, default=0
#'     The GPU device ID, starts from 0.
#' @return The GPU context.
#' @name mx.gpu
#'
#' @export
NULL
