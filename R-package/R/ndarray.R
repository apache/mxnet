mx.nd.load <- function(filename) {
  filename <- path.expand(filename)
  mx.nd.internal.load(filename)
}

mx.nd.save <- function(ndarray, filename) {
  filename <- path.expand(filename)
  mx.nd.internal.save(ndarray, filename)
}

is.MXNDArray <- function(x)
  inherits(x, "Rcpp_MXNDArray")

as.array.Rcpp_MXNDArray <- function(x){
      return(x$as.array())
}

`+.Rcpp_MXNDArray` <- function(e1, e2) {
  if(is.MXNDArray(e1)&&is.MXNDArray(e2)) {
    mx.nd.internal.plus(e1, e2)
  } else if (is.MXNDArray(e1)&&is.numeric(e2)) {
    mx.nd.internal.plus.scalar(e1, e2)
  } else if (is.MXNDArray(e2)&&is.numeric(e1)) {
    mx.nd.internal.plus.scalar(e2, e1)
  } else {
    stop("unsupport type found.")
  }
}