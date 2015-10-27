# TODO(Tong, KK) check style to make it more like R..

#' Set the seed used by mxnet device-specific random number generators.
#'
#' @details
#' We have a specific reason why \code{mx.set.seed} is introduced,
#' instead of simply use \code{set.seed}.
#'
#' The reason that is that most of mxnet random number generator
#' can run on different devices, such as GPU.
#' We need to use massively parallel PRNG on GPU to get fast
#' random number generations. It can also be quite costly to seed these PRNGs.
#' So we introduced \code{mx.set.seed} for mxnet specific device random numbers.
#'
#' @param seed the seed value to the device random number generators.
#'
#' @examples
#'
#' mx.set.seed(0)
#' as.array(mx.runif(2))
#' # 0.5488135 0.5928446
#' mx.set.seed(0)
#' as.array(mx.rnorm(2))
#' # 2.212206 1.163079
#'
#' @export
mx.set.seed <- function(seed) {
  mx.internal.set.seed(seed)
}

#' Generate uniform distribution in [low, high) with specified shape.
#'
#' @param shape Dimension, The shape(dimension) of the result.
#' @param min numeric, The lower bound of distribution.
#' @param max numeric, The upper bound of distribution.
#' @param ctx, optional The context device of the array. mx.ctx.default() will be used in default.
#'
#' @examples
#'
#' mx.set.seed(0)
#' as.array(mx.runif(2))
#' # 0.5488135 0.5928446
#' mx.set.seed(0)
#' as.array(mx.rnorm(2))
#' # 2.212206 1.163079
#'
#' @export
mx.runif <- function(shape, min=0, max=1, ctx=NULL) {
  if (!is.numeric(min)) stop("mx.rnorm only accept numeric min")
  if (!is.numeric(max)) stop("mx.rnorm only accept numeric max")
  ret <- mx.nd.internal.empty(shape, ctx)
  return (mx.nd.internal.random.uniform(min, max, out=ret))
}

#' Generate nomal distribution with mean and sd.
#'
#' @param shape Dimension, The shape(dimension) of the result.
#' @param mean numeric, The mean of distribution.
#' @param sd numeric, The standard deviations.
#' @param ctx, optional The context device of the array. mx.ctx.default() will be used in default.
#'
#' @examples
#'
#' mx.set.seed(0)
#' as.array(mx.runif(2))
#' # 0.5488135 0.5928446
#' mx.set.seed(0)
#' as.array(mx.rnorm(2))
#' # 2.212206 1.163079
#'
#' @export
mx.rnorm <- function(shape, mean=0, sd=1, ctx=NULL) {
  if (!is.numeric(mean)) stop("mx.rnorm only accept numeric mean")
  if (!is.numeric(sd)) stop("mx.rnorm only accept numeric sd")
  ret <- mx.nd.internal.empty(shape, ctx)
  return (mx.nd.internal.random.gaussian(mean, sd, out=ret))
}
