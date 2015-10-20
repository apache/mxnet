#' Internal default value initialization scheme.
#'
#' @param name the name of the variable.
#' @param shape the shape of the array to be generated.
#'
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
#' @param sd The standard deviation of normal distribution
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

#' Create initialization of argument  like arg.array
#'
#' @param initializer The initializer.
#' @param shape.array named-list The shape of the weights
#' @param ctx mx.context The context of the weights
#' @param skip.unknown Whether skip the unknown weight types
#' @export
mx.init.create <- function(initializer, shape.array, ctx, skip.unknown=TRUE) {
  if (length(shape.array) == 0) return(list())
  names = names(shape.array)
  ret <- lapply(1 : length(names), function(i) {
    initializer(names[[i]], shape.array[[i]], ctx, allow.unknown=skip.unknown)
  })
  names(ret) <- names
  if (skip.unknown) {
    ret <- mx.util.filter.null(ret)
  }
  return(ret)
}
