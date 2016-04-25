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

#' @title Xavier initializer
#'
#' @description Create a initializer which initialize weight with Xavier or
#' similar initialization scheme.
#'
#' @param rnd_type A string of \code{character} indicating the type of
#' distribution from which the weights are initialized.
#' @param factor_type A string of \code{character}.
#' @param magnitude A \code{numeric} number indicating the scale of random
#' number range.
#' @export
mx.init.Xavier <- function(rnd_type = "uniform", factor_type = "avg",
                           magnitude = 3){
  function(name, shape, ctx, allow.unknown = FALSE){
    if (!mx.util.str.endswith(name, "weight")) {
      return (mx.init.internal.default(name, shape, ctx, allow.unknown))
    }

    fan_out = shape[length(shape)]
    fan_in  = prod(shape[-length(shape)])
    factor_val  = 1
    if (factor_type == "avg") {
      factor_val = (fan_in + fan_out) / 2
    } else if (factor_type == "in"){
      factor_val = fan_in
    } else if (factor_type == "out"){
      factor_val = fan_out
    } else {
      stop("Not supported factor type. See usage of function mx.init.Xavier")
    }

    scale = sqrt(magnitude / factor_val)

    if (rnd_type == "uniform"){
      return(mx.runif(shape, -scale, scale, ctx))
    } else if (rnd_type == "gaussian"){
      return(mx.rnorm(shape, 0, scale, ctx))
    } else {
      stop("Not supported random type. See usage of function mx.init.Xavier")
    }
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
