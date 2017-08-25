is.MXKVStore <- function(x) {
  inherits(x, "Rcpp_MXKVStore")
}

#' Create a mxnet KVStore.
#'
#' @param type string(default="local") The type of kvstore.
#' @return The kvstore.
#'
#' @name mx.kv.create
#' @export
NULL
