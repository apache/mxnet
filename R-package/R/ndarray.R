
mx.nd.load <- function(filename) {
  filename <- path.expand(filename)
  mx.nd.internal.load(filename)
}

mx.nd.save <- function(ndarray, filename) {
  filename <- path.expand(filename)
  mx.nd.internal.save(ndarray, filename)
}

mx.nd.zeros <- function(shape, ctx) {
  ret <- mx.nd.internal.empty(shape, ctx)
  return (mx.nd.internal.set.value(0.0, out=ret))
}

mx.nd.ones <- function(shape, ctx) {
  ret <- mx.nd.internal.empty(shape, ctx)
  return (mx.nd.internal.set.value(1.0, out=ret))
}

is.MXNDArray <- function(x) {
  inherits(x, "Rcpp_MXNDArray")
}

init.ndarray.methods <- function() {
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
  setMethod("as.array", signature(x = "Rcpp_MXNDArray"), function(x) {
    x$as.array()
  })
}
