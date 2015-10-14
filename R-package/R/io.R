is.MXDataIter <- function(x) {
  inherits(x, "Rcpp_MXNativeDataIter") ||
  inherits(x, "Rcpp_MXArrayDataIter")
}

