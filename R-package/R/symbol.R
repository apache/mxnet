#' Symbol Interface of MXNet
#'
# TODO(KK, tong) expose more member functions
mx.symbol.save <-function(symbol, filename) {
  filename <- path.expand(filename)
  symbol$save(filename)
}
init.symbol.methods <- function() {
  # Think of what is the best naming
  setMethod("+", signature(e1 = "Rcpp_MXSymbol", e2 = "Rcpp_MXSymbol"), function(e1, e2) {
    mx.varg.symbol.internal.Plus(list(e1, e2))
  })
}
