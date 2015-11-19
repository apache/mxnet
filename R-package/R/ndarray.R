#' Load an mx.nd.array object on disk
#'
#' @param filename the filename (including the path)
#'
#' @examples
#' mat = mx.nd.array(1:3)
#' mx.nd.save(mat, 'temp.mat')
#' mat2 = mx.nd.load('temp.mat')
#' as.array(mat)
#' as.array(mat2)
#'
#' @export
mx.nd.load <- function(filename) {
  filename <- path.expand(filename)
  return(mx.nd.internal.load(filename))
}

#' Save an mx.nd.array object
#'
#' @param ndarray the \code{mx.nd.array} object
#' @param filename the filename (including the path)
#'
#' @examples
#' mat = mx.nd.array(1:3)
#' mx.nd.save(mat, 'temp.mat')
#' mat2 = mx.nd.load('temp.mat')
#' as.array(mat)
#' as.array(mat2[[1]])
#'
#' @export
mx.nd.save <- function(ndarray, filename) {
  filename <- path.expand(filename)
  if (!is.list(ndarray)) {
    mx.nd.internal.save(list(ndarray), filename)
  } else {
    mx.nd.internal.save(ndarray, filename)
  }
}

mx.nd.internal.empty <- function(shape, ctx=NULL) {
  if (is.null(ctx)) ctx <- mx.ctx.default()
  if (!is.mx.context(ctx)) stop("wrong mx.context object, please specify with mx.cpu() or mx.gpu()")
  return (mx.nd.internal.empty.array(shape, ctx))
}

#' Generate an mx.nd.array object with zeros
#'
#' @param shape the dimension of the \code{mx.nd.array}
#' @param ctx optional The context device of the array. mx.ctx.default() will be used in default.
#'
#' @examples
#' mat = mx.nd.zeros(10)
#' as.array(mat)
#' mat2 = mx.nd.zeros(c(5,5))
#' as.array(mat)
#' mat3 = mx.nd.zeroes(c(3,3,3))
#' as.array(mat3)
#'
#' @export
mx.nd.zeros <- function(shape, ctx=NULL) {
  ret <- mx.nd.internal.empty(shape, ctx)
  return (mx.nd.internal.set.value(0.0, out=ret))
}

#' Generate an mx.ndarray object with ones
#'
#' @param shape the dimension of the \code{mx.ndarray}
#' @param ctx optional The context device of the array. mx.ctx.default() will be used in default.
#'
#' @examples
#' mat = mx.nd.ones(10)
#' as.array(mat)
#' mat2 = mx.nd.ones(c(5,5))
#' as.array(mat)
#' mat3 = mx.nd.ones(c(3,3,3))
#' as.array(mat3)
#'
#' @export
mx.nd.ones <- function(shape, ctx=NULL) {
  ret <- mx.nd.internal.empty(shape, ctx)
  return (mx.nd.internal.set.value(1.0, out=ret))
}

#' Generate an mx.ndarray object on ctx, with data copied from src
#'
#' @param src The source mx.ndarray object.
#' @param ctx The target context.
#'
#' @export
mx.nd.copyto <- function(src, ctx) {
  ret <- mx.nd.internal.empty(dim(src), ctx)
  return (mx.nd.internal.copyto(src, out=ret))
}

#' Create a new \code{mx.ndarray} that copies the content from src on ctx.
#'
#' @param src.array Source array data of class \code{array}, \code{vector} or \code{matrix}.
#' @param ctx optional The context device of the array. mx.ctx.default() will be used in default.
#'
#' @return An \code{mx.ndarray}
#'
#' @rdname mx.nd.array
#' 
#' @return An Rcpp_MXNDArray object
#' 
#' @examples
#' mat = mx.nd.array(x)
#' mat = 1 - mat + (2 * mat)/(mat + 0.5)
#' as.array(mat)
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

is.MXNDArray <- function(nd) {
  class(nd) == "MXNDArray"
}

#' Check if src.array is mx.ndarray
#'
#' @return Logical indicator
#'
#' @examples
#' mat = mx.nd.array(1:10)
#' is.mx.ndarray(mat)
#' mat2 = 1:10
#' is.mx.ndarray(mat2)
#'
#' @export
is.mx.ndarray <- function(src.array) {
  is.MXNDArray(src.array)
}

#' Binary operator overloading of mx.ndarray
#' @param e1 The first operand
#' @param e1 The second operand
#' @export
Ops.MXNDArray <- function(e1, e2) {
  mx.nd.internal.dispatch.Ops(.Generic, e1, e2)
}

#' Dimension operator overload of mx.ndarray
#' @param nd The mx.ndarray
#' @export
dim.MXNDArray <- function(nd) {
  mx.nd.internal.dim(nd)
}

#' Length operator overload of mx.ndarray
#' @param nd The mx.ndarray
#' @export
length.MXNDArray <- function(nd) {
  mx.nd.internal.length(nd)
}

#' as.array operator overload of mx.ndarray
#' @param nd The mx.ndarray
#' @export
as.array.MXNDArray <- function(nd) {
  mx.nd.internal.as.array(nd)
}

#' as.matrix operator overload of mx.ndarray
#' @param nd The mx.ndarray
#' @export
as.matrix.MXNDArray <- function(nd) {
  if (length(dim(nd)) != 2) {
    stop("The input argument is not two dimensional matrix.")
  }
  as.matrix(as.array(x))
}

#' print operator overload of mx.ndarray
#' @param nd The mx.ndarray
#' @export
print.MXNDArray <- function(nd) {
  print(as.array(nd))
}

# TODO(KK) use generics?

#' Get the context of mx.ndarray
#' @param nd The mx.ndarray
#' @export
ctx <-function(nd) {
  mx.nd.internal.ctx(nd)
}
