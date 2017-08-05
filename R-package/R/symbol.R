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
mx.symbol.concat <- function(data, num.args, dim = NULL, name = NULL) {
  data[['num.args']] <- num.args
  
  if(!is.null(dim)) data[['dim']] <- dim
  
  if(!is.null(name)) data[['name']] <- name
  
  mx.varg.symbol.concat(data)
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
  warning("mx.symbol.Concat is deprecated. Use mx.symbol.concat instead.")
  mx.symbol.concat(data, num.args, dim, name)
}

#' @export
mx.symbol.min <- function(e1, e2) {
  if (is.mx.symbol(e1) && is.mx.symbol(e2)) {
    mx.varg.symbol.internal.minimum(list(e1, e2))
  } else if (is.mx.symbol(e1)) {
    mx.varg.symbol.internal.minimum_scalar(list(e1, scalar = e2))
  } else if (is.mx.symbol(e2)) {
    mx.varg.symbol.internal.minimum_scalar(list(e2, scalar = e1))
  }
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

#' Get a symbol that contains all the internals
#' @param x The input symbol
#'
#' @export
internals <- function(x) {
  if (!is.MXSymbol(x)) stop("only for MXSymbol type")
  x$get.internals()
}

#' Gets a new grouped symbol whose output contains inputs to output nodes of the original symbol.
#' @param x The input symbol
#'
#' @export
children <- function(x) {
  if (!is.MXSymbol(x)) stop("only for MXSymbol type")
  x$get.children()
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
  setMethod("+", signature(e1 = "numeric", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.PlusScalar(list(e2, scalar = e1))
  })  
  setMethod("-", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Minus(list(e1, e2))
  })
  setMethod("-", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.MinusScalar(list(e1, scalar = e2))
  })
  setMethod("-", signature(e1 = "numeric", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.rminus_scalar(list(e2, scalar = e1))
  })  
  setMethod("*", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Mul(list(e1, e2))
  })
  setMethod("*", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.MulScalar(list(e1, scalar = e2))
  })
  setMethod("*", signature(e1 = "numeric", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.MulScalar(list(e2, scalar = e1))
  })  
  setMethod("/", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Div(list(e1, e2))
  })
  setMethod("/", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.DivScalar(list(e1, scalar = e2))
  })
  setMethod("/", signature(e1 = "numeric", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.rdiv_scalar(list(e2, scalar = e1))
  })  
  setMethod("%%", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Mod(list(e1, e2))
  })
  setMethod("%%", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.ModScalar(list(e1, scalar = e2))
  })
  setMethod("%%", signature(e1 = "numeric", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.RModScalar(list(e2, scalar = e1))
  })  
  setMethod("%/%", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Mod(list(e1, e2))
  })
  setMethod("%/%", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.ModScalar(list(e1, scalar = e2))
  })
  setMethod("%/%", signature(e1 = "numeric", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.RModScalar(list(e2, scalar = e1))
  })
  setMethod("^", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.power(list(e1, e2))
  })
  setMethod("^", signature(e1 = "Rcpp_MXSymbol", e2 = "numeric"), function(e1, e2) {
    mx.varg.symbol.internal.power_scalar(list(e1, scalar = e2))
  })
  setMethod("^", signature(e1 = "numeric", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.rpower_scalar(list(e2, scalar = e1))
  })
}
