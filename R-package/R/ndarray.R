#' NDArray
#'
#' Additional NDArray related operations
init.ndarray.methods <- function() {
  require(methods)
  setMethod("+", signature(e1 = "Rcpp_MXNDArray", e2 = "numeric"), function(e1, e2) {
    mx.nd.internal.plus.scalar(e1, e2)
  })
  setMethod("+", signature(e1 = "Rcpp_MXNDArray", e2 = "Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.plus(e1, e2)
  })
  setMethod("+", signature(e1 = "numeric", e2 = "Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.plus.scalar(e2, e1)
  })
  setMethod("-", signature(e1 = "Rcpp_MXNDArray", e2 = "numeric"), function(e1, e2) {
    mx.nd.internal.minus.scalar(e1, e2)
  })
  setMethod("-", signature(e1 = "Rcpp_MXNDArray", e2 = "Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.minus(e1, e2)
  })
  setMethod("-", signature(e1 = "numeric", e2 = "Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.rminus.scalar(e2, e1)
  })
  setMethod("*", signature(e1 = "Rcpp_MXNDArray", e2 = "numeric"), function(e1, e2) {
    mx.nd.internal.mul.scalar(e1, e2)
  })
  setMethod("*", signature(e1 = "Rcpp_MXNDArray", e2 = "Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.mul(e1, e2)
  })
  setMethod("*", signature(e1 = "numeric", e2 = "Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.mul.scalar(e2, e1)
  })
  setMethod("/", signature(e1 = "Rcpp_MXNDArray", e2 = "numeric"), function(e1, e2) {
    mx.nd.internal.div.scalar(e1, e2)
  })
  setMethod("/", signature(e1 = "Rcpp_MXNDArray", e2 = "Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.div(e1, e2)
  })
  setMethod("/", signature(e1 = "numeric", e2 = "Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.rdiv.scalar(e2, e1)
  })
}

