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

#' MXNet: Flexible and Efficient GPU computing and Deep Learning.
#'
#' MXNet is a flexible and efficient GPU computing and deep learning framework.
#'
#' It enables you to write seamless tensor/matrix computation with multiple GPUs in R.
#'
#' It also enables you construct and customize the state-of-art deep learning models in R,
#' and apply them to tasks such as image classification and data science challenges.
#'
#' @docType package
#' @name mxnet
#' @import methods Rcpp
NULL

.MXNetEnv <- new.env()

.onLoad <- function(libname, pkgname) {
  # Require methods for older versions of R
  require(methods)
  tryCatch(library.dynam("libmxnet", pkgname, libname, local=FALSE), error = function(e) { print('Loading local: inst/libs/libmxnet.so'); dyn.load("R-package/inst/libs/libmxnet.so", local=FALSE) })
  tryCatch(library.dynam("mxnet", pkgname, libname), error = function(e) { print('Loading local: src/mxnet.so'); dyn.load("R-package/src/mxnet.so") })
  loadModule("mxnet", TRUE)
  init.symbol.methods()
  init.context.default()
}

.onUnload <- function(libpath) {
  message("Start unload")
  mx.internal.notify.shutdown()
  library.dynam.unload("mxnet", libpath)
  library.dynam.unload("libmxnet", libpath)
  message("MXNet shutdown")
}

.onAttach <- function(...) {
  if (!interactive() || stats::runif(1) > 0.1) return()

  tips <- c(
    "Need help? Feel free to open an issue on https://github.com/dmlc/mxnet/issues",
    "For more documents, please visit http://mxnet.io",
    "Use suppressPackageStartupMessages() to eliminate package startup messages."
  )

  tip <- sample(tips, 1)
  packageStartupMessage(paste(strwrap(tip), collapse = "\n"))
}
