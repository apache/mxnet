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
  print("Start unload")
  mx.internal.notify.shutdown()
  unloadModule("mxnet")
  library.dynam.unload("mxnet", libpath)
  library.dynam.unload("libmxnet", libpath)
}
