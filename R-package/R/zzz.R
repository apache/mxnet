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
  library.dynam("libmxnet", pkgname, libname, local=FALSE)
  library.dynam("mxnet", pkgname, libname)
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
