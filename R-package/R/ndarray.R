#' NDArray
#'
#' Additional NDArray related operations
init.ndarray.methods <-function() {
  require(methods)
  setMethod("+", signature(e1="Rcpp_MXNDArray", e2="numeric"), function(e1, e2) {
    mx.nd.internal.plus.scalar(e1, e2)
  })
  setMethod("+", signature(e1="Rcpp_MXNDArray", e2="Rcpp_MXNDArray"), function(e1, e2) {
    mx.nd.internal.plus(e1, e2)
  })
}
