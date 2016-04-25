#' Create a symbolic variable with specified name.
#'
#' @param name string
#'     The name of the result symbol.
#' @return The result symbol
#' @name mx.symbol.Variable
#'
#' @export
NULL

#' Create a symbol that groups symbols together.
#'
#' @param kwarg
#'     Variable length of symbols or list of symbol.
#' @return The result symbol
#'
#' @export
mx.symbol.Group <- function(...) {
  mx.varg.symbol.internal.Group(list(...))
}

#' Perform an feature concat on channel dim (dim 1) over all the inputs.
#' 
#' @param data  list, required
#'     List of tensors to concatenate
#' @param num.args  int, required
#'     Number of inputs to be concated.
#' @param dim  int, optional, default='1'
#'     the dimension to be concated.
#' @param name  string, optional
#'     Name of the resulting symbol.
#' @return out The result mx.symbol
#' 
#' @export
mx.symbol.Concat <- function(data, num.args, dim = NULL, name = NULL) {
  data[['num.args']] <- num.args
  
  if(!is.null(dim)) data[['dim']] <- dim
  
  if(!is.null(name)) data[['name']] <- name
  
  mx.varg.symbol.Concat(data)
}

#' Save an mx.symbol object
#'
#' @param symbol the \code{mx.symbol} object
#' @param filename the filename (including the path)
#'
#' @examples
#' data = mx.symbol.Variable('data')
#' mx.symbol.save(data, 'temp.symbol')
#' data2 = mx.symbol.load('temp.symbol')
#'
#' @export
mx.symbol.save <-function(symbol, filename) {
  filename <- path.expand(filename)
  symbol$save(filename)
}

#' Load an mx.symbol object
#'
#' @param filename the filename (including the path)
#'
#' @examples
#' data = mx.symbol.Variable('data')
#' mx.symbol.save(data, 'temp.symbol')
#' data2 = mx.symbol.load('temp.symbol')
#'
#' @export
mx.symbol.load <-function(filename) {
  filename <- path.expand(filename)
  mx.symbol.load(filename)
}

#' Load an mx.symbol object from a json string
#'
#' @param str the json str represent a mx.symbol
#'
#' @export
#' @name mx.symbol.load.json
NULL


#' Inference the shape of arguments, outputs, and auxiliary states.
#'
#' @param symbol The \code{mx.symbol} object
#'
#' @export
mx.symbol.infer.shape <- function(symbol, ...) {
  symbol$infer.shape(list(...))
}

is.MXSymbol <- function(x) {
  inherits(x, "Rcpp_MXSymbol")
}

#' Judge if an object is mx.symbol
#'
#' @return Logical indicator
#'
#' @export
is.mx.symbol <- is.MXSymbol


#' Get the arguments of symbol.
#' @param x The input symbol
#'
#' @export
arguments <- function(x) {
  if (!is.MXSymbol(x))
    stop("only for MXSymbol type")
  x$arguments
}

#' Apply symbol to the inputs.
#' @param x The symbol to be applied
#' @param kwargs The keyword arguments to the symbol
#'
#' @export
mx.apply <- function(x, ...) {
  if (!is.MXSymbol(x)) stop("only for MXSymbol type")
  x$apply(list(...))
}

#' Get the outputs of a symbol.
#' @param x The input symbol
#'
#' @export
outputs <- function(x) {
  if (!is.MXSymbol(x)) stop("only for MXSymbol type")
  x$outputs
}

init.symbol.methods <- function() {
  # Think of what is the best naming
  setMethod("+", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Plus(list(e1, e2))
  })
  setMethod("+", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.PlusScalar(list(e1, scalar = e2))
  })
  setMethod("-", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Minus(list(e1, e2))
  })
  setMethod("-", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.MinusScalar(list(e1, scalar = e2))
  })
  setMethod("*", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Mul(list(e1, e2))
  })
  setMethod("*", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.MulScalar(list(e1, scalar = e2))
  })
  setMethod("/", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Div(list(e1, e2))
  })
  setMethod("/", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.DivScalar(list(e1, scalar = e2))
  })
}
