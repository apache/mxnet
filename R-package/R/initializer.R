mx.util.str.endswith <- function(name, suffix) {
#  slen <- nchar(suffix)
#  nlen <- nchar(name)
#  if (slen > nlen) return (FALSE)
#  nsuf <- substr(name, nlen - slen + 1, nlen)
#  return (nsuf == suffix)
  ptrn = paste0(suffix, "\\b")
  return(grepl(ptrn, name))
}

#' Internal default value initialization scheme.
#' @param name the name of the variable.
#' @param shape the shape of the array to be generated.
mx.init.internal.default <- function(name, shape, ctx, allow.unknown=FALSE) {
  if (mx.util.str.endswith(name, "bias")) return (mx.nd.zeros(shape, ctx))
  if (mx.util.str.endswith(name, "gamma")) return (mx.nd.ones(shape, ctx))
  if (mx.util.str.endswith(name, "beta")) return (mx.nd.zeros(shape, ctx))
  if (mx.util.str.endswith(name, "moving_mean")) return (mx.nd.zeros(shape, ctx))
  if (mx.util.str.endswith(name, "moving_var")) return (mx.nd.ones(shape, ctx))
  if (allow.unknown) return(NULL)
  stop(paste("Unkown initialization pattern for ", name))
}

#' Create a initializer that initialize the weight with uniform [-scale, scale]
#'
#' @param scale The scale of uniform distribution
#'
#' @export
mx.init.uniform <- function(scale) {
  function(name, shape, ctx, allow.unknown=FALSE) {
    if (!mx.util.str.endswith(name, "weight")) {
      return (mx.init.internal.default(name, shape, ctx, allow.unknown))
    }
    return (mx.runif(shape, -scale, scale, ctx))
  }
}

#' Create a initializer that initialize the weight with normal(0, sd)
#'
#' @param scale The scale of uniform distribution
#' 
#' @export
mx.init.normal <- function(sd) {
  function(name, shape, ctx, allow.unknown=FALSE) {
    if (!mx.util.str.endswith(name, "weight")) {
      return (mx.init.internal.default(name, shape, ctx, allow.unknown))
    }
    return (mx.rnorm(shape, 0, sd, ctx))
  }
}


# Create initialization of argument  like arg.array
mx.init.create <- function(initializer, arg.array, allow.unknown=TRUE) {
  names = names(arg.array)
  sapply(1 : length(names), function(i) {
    initializer(names[[i]], dim(arg.array[[i]]),
                arg.array[[i]]$ctx, allow.unknown=allow.unknown)
  }, simplify = FALSE, USE.NAMES = TRUE)
}
