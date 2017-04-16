# Initialize the global context
init.context.default <- function() {
  assign("mx.ctx.internal.default.value", mx.cpu(), envir = .MXNetEnv)
}

#' Set/Get default context for array creation.
#'
#' @param new, optional takes \code{mx.cpu()} or \code{mx.gpu(id)}, new default ctx.
#' @return The default context.
#'
#' @export
mx.ctx.default <- function(new = NULL) {
  if (!is.null(new)) {
    assign("mx.ctx.internal.default.value", new, envir = .MXNetEnv)
  }
  return (.MXNetEnv$mx.ctx.internal.default.value)
}

#' Check if the type is mxnet context.
#'
#' @return Logical indicator
#'
#' @export
is.mx.context <- function(x) {
  class(x) == "MXContext"
}


#' Create a mxnet CPU context.
#'
#' @param dev.id optional, default=0
#'     The device ID, this is meaningless for CPU, included for interface compatiblity.
#' @return The CPU context.
#' @name mx.cpu
#'
#' @export
NULL

#' Create a mxnet GPU context.
#'
#' @param dev.id optional, default=0
#'     The GPU device ID, starts from 0.
#' @return The GPU context.
#' @name mx.gpu
#'
#' @export
NULL
