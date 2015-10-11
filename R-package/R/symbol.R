
mx.symbol.save <-function(symbol, filename) {
  filename <- path.expand(filename)
  symbol$save(filename)
}

mx.symbol.load <-function(filename) {
  filename <- path.expand(filename)
  mx.symbol.load(filename)
}

mx.symbol.Activation <- function(...) {
  mx.varg.symbol.Activation(list(...))
}

mx.symbol.FullyConnected <- function(...) {
  mx.varg.symbol.FullyConnected(list(...))  
}

mx.symbol.LRN <- function(...) {
  mx.varg.symbol.LRN(list(...))
}

mx.symbol.BatchNorm <- function(...) {
  mx.varg.symbol.BatchNorm(list(...))
}

mx.symbol.Group <- function(...) {
  mx.varg.symbol.Group(list(...))
}

mx.symbol.Pooling <- function(...) {
  mx.varg.symbol.Pooling(list(...))
}

mx.symbol.Concat <- function(...){
  mx.varg.symbol.Concat(list(...))
}

mx.symbol.Reshape <- function(...) {
  mx.varg.symbol.Reshape(list(...))
}

mx.symbol.Convolution <- function(...) {
  mx.varg.symbol.Convolution(list(...))
}

mx.symbol.SliceChannel <- function(...) {
  mx.varg.symbol.SliceChannel(list(...))
}

mx.symbol.Dropout <- function(...) {
  mx.varg.symbol.Dropout(list(...))
}

mx.symbol.Softmax <- function(...) {
  mx.varg.symbol.Softmax(list(...))
}

mx.symbol.ElementWiseSum <- function(...) {
  mx.varg.symbol.ElementWiseSum(list(...))
}

mx.symbol.Flatten <- function(...) {
  mx.varg.symbol.Flatten(list(...))
}

mx.symbol.LeakyReLU <- function(...) {
  mx.varg.symbol.LeakyReLU(list(...))
}

is.MXSymbol <- function(x) {
  inherits(x, "Rcpp_MXSymbol")
}

arguments <- function(x) {
  if (!is.MXSymbol(x))
    stop("only for MXSymbol type")
  x$arguments()
}

mx.apply <- function(x, ...) {
  if (!is.MXSymbol(x))
    stop("only for MXSymbol type")
  x$apply(list(...))
}

outputs <- function(x, ...) {
  if (!is.MXSymbol(x))
    stop("only for MXSymbol type")
  x$outputs()
}

init.symbol.methods <- function() {
  # Think of what is the best naming
  setMethod("+", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Plus(list(e1, e2))
  })

  setMethod("-", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Minus(list(e1, e2))
  })
  
  setMethod("*", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Mul(list(e1, e2))
  })
  
  setMethod("/", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Div(list(e1, e2))
  })

}
