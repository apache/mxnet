# TODO(KK) expose executor functions

# Simple bind the symbol to executor,
# with information from input shapes.
mx.simple.bind <- function(symbol, ctx, grad.req=FALSE, ...) {
  if (!is.MXSymbol(symbol)) stop("symbol need to be MXSymbol")
  slist <- symbol$infer.shape(list(...))

  if (is.null(slist)) {
    stop("Need more shape information to decide the shapes of arguments")
  }
  arg.arrays <- sapply(slist$arg.shapes, function(shape) {
    mx.nd.zeros(shape, ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  aux.arrays <- sapply(slist$aux.shapes, function(shape) {
    mx.nd.zeros(shape, ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  grad.reqs <- lapply(names(slist$arg.shapes), function(nm) {
    grad.req &&
    !mx.util.str.endswith(nm, "label") &&
    !mx.util.str.endswith(nm, "data")
  })
  mx.symbol.bind(symbol, ctx,
                 arg.arrays=arg.arrays,
                 aux.arrays=aux.arrays,
                 grad.reqs = grad.reqs)
}
