# Initialize the global context
init.context.default <- function() {
  .GlobalEnv$mx.ctx.internal.default.value <- mx.cpu()
}

# TODO(KK, tong) check if roxygen style is correct.

#'
#' Set/Get default context for array creation.
#'
#' @rdname mx.ctx
#' 
#' @param new, optional takes \code{mx.cpu()} or \code{mx.gpu(id)}, new default ctx.
#'
#' @export
mx.ctx.default <- function(new = NULL) {
  if (!is.null(new)) {
    mx.ctx.internal.default.value <<- new
  }
  return (mx.ctx.internal.default.value)
}

# TODO need examples

#' @rdname mx.ctx
#' 
#' @return Logical indicator 
#' 
#' @export
is.mx.context <- function(x) {
  class(x) == "MXContext"
}
