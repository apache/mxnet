#' Simple bind the symbol to executor,
#' with information from input shapes.
#'
#' @export
mx.simple.bind <- function(symbol, ctx, grad.req = "null", ...) {
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
    if (!mx.util.str.endswith(nm, "label") && !mx.util.str.endswith(nm, "data")) {
      grad.req
    } else {
      "null"
    }
  })
  mx.symbol.bind(symbol, ctx,
                 arg.arrays=arg.arrays,
                 aux.arrays=aux.arrays,
                 grad.reqs = grad.reqs)
}

#' Update the executors with new arrays
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.update.arg.arrays <- function(exec, arg.arrays, match.name=FALSE, skip.null=FALSE) {
  exec$update.arg.arrays(arg.arrays, match.name, skip.null)
}

#' Update the executors with new arrays
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.update.aux.arrays <- function(exec, arg.arrays, match.name=FALSE, skip.null=FALSE) {
  exec$update.aux.arrays(arg.arrays, match.name, skip.null)
}

#' Update the executors with new arrays
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.update.grad.arrays <- function(exec, arg.arrays, match.name=FALSE, skip.null=FALSE) {
  exec$update.grad.arrays(arg.arrays, match.name, skip.null)
}


#' Peform an forward on the executors
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.forward <- function(exec, is.train=TRUE) {
  exec$forward(is.train, list())
}

#' Peform an backward on the executors
#' This function will MUTATE the state of exec
#'
#' @export
mx.exec.backward <- function(exec, ...) {
  exec$backward(list(...))
}
