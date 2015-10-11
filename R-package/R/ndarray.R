
mx.nd.load <- function(filename) {
  filename <- path.expand(filename)
  mx.nd.internal.load(filename)
}

mx.nd.save <- function(ndarray, filename) {
  filename <- path.expand(filename)
  mx.nd.internal.save(ndarray, filename)
}

mx.nd.internal.empty <- function(shape, ctx=NULL) {
  if (is.null(ctx)) ctx <- mx.ctx.default()
  return (mx.nd.internal.empty.array(shape, ctx))
}

mx.nd.zeros <- function(shape, ctx=NULL) {
  ret <- mx.nd.internal.empty(shape, ctx)
  return (mx.nd.internal.set.value(0.0, out=ret))
}

mx.nd.ones <- function(shape, ctx=NULL) {
  ret <- mx.nd.internal.empty(shape, ctx)
  return (mx.nd.internal.set.value(1.0, out=ret))
}

# TODO(tong) improve this, add doc

#'
#' Create a new \code{mx.ndarray} that copies the content from src on ctx.
#'
#' @param src.array, Source array data of class \code{array}, \code{vector} or \code{matrix}.
#' @param ctx, optional The context device of the array. mx.ctx.default() will be used in default.
#'
#' @export
mx.nd.array <- function(src.array, ctx=NULL) {
  if (is.null(ctx)) ctx <- mx.ctx.default()
  if (!is.array(src.array)) {
    if (!is.vector(src.array) && !is.matrix(src.array)) {
      stop("mx.nd.array takes an object of class array, vector or matrix only.")
    } else {
      src.array <- as.array(src.array)
    }
  }
  return (mx.nd.internal.array(src.array, ctx))
}

is.MXNDArray <- function(x) {
  inherits(x, "Rcpp_MXNDArray")
}

is.mx.ndarray <- is.MXNDArray

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
  setMethod("as.matrix", signature(x = "Rcpp_MXNDArray"), function(x) {
    if (length(dim(x)) != 2) {
      stop("The input argument is not two dimensional matrix.")
    }
    as.matrix(x$as.array())
  })
  setMethod("print", signature(x = "Rcpp_MXNDArray"), function(x) {
    print(x$as.array())
  })
  setMethod("dim", signature(x = "Rcpp_MXNDArray"), function(x) {
    x$dim()
  })
}
