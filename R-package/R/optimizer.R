#' Create an SGD optimizer with respective parameters.
#' Perform SGD with momentum update
#'
mx.opt.sgd <- function(learning.rate,
                       momentum=0,
                       wd=0,
                       rescale.grad=1,
                       clip_gradient = NULL, 
                       lr_scheduler = NULL) {
  # use lr as short for learing rate.
  lr <- learning.rate
  count       <- 0
  num_update  <- 0

  sgd <- new.env()
  sgd$lr <- lr
  sgd$count <- 0
  sgd$num_update <- 0

  create.state <- function(index, weight) {
    if (momentum == 0) {
      return(NULL)
    } else {
      ret <- (mx.nd.zeros(dim(weight), ctx(weight)))
      return(ret)
    }
  }
  update <- function(index, weight, grad, state) {

    if (!is.null(lr_scheduler)){
      lr_scheduler(sgd) ## changing lr
      lr <- sgd$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = sgd, x = indexKey)){
        assign(x = indexKey, value = 0, envir = sgd)
      } else {
        indexValue <- get(envir = sgd, x = indexKey)
        assign(x = indexKey, value = indexValue + 1, envir = sgd)
        sgd$num_update <- max(sgd$num_update, get(envir = sgd, x = indexKey))
      }
    }
    grad <- grad * rescale.grad
    if (!is.null(clip_gradient)){
      if(clip_gradient >= 0){
          grad_ctx <- ctx(grad)
          grad <- as.array(grad)
          grad <- pmax(grad, -1 * clip_gradient)
          grad <- pmin(grad, clip_gradient)
          grad <- mx.nd.array(grad, grad_ctx)
      } else {
        stop("Error: clip_gradient should be positive number.")
      }
    }
    if (is.null(state)) {
      weight <- weight - lr * (grad + wd * weight)
    } else {
      mom <- state
      mom <- mom * momentum
      mom <- mom - lr * (grad + wd * weight)
      weight <- weight + mom
      state <- mom
    }
    return(list(weight=weight, state=state))
  }
  return(list(create.state=create.state, update=update))
}

#' Create an RMSProp optimizer with respective parameters.
#' Reference: Tieleman T, Hinton G. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude[J]. COURSERA: Neural Networks for Machine Learning, 2012, 4(2).
#' The code follows: http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
#' 
#' @param learning.rate float, default=0.002
#'      Step size.
#' @param gamma1 float, default=0.95
#'      decay factor of moving average for gradient, gradient^2.
#' @param gamm2 float, default=0.9
#'      "momentum" factor.
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
#'
mx.opt.rmsprop <- function(learning.rate=0.002,
                           gamma1=0.95,
                           gamma2=0.9,
                           wd=0,
                           rescale.grad=1,
                           clip_gradient = NULL, 
                           lr_scheduler = NULL) {
  # use lr as short for learing rate.
  lr <- learning.rate
  count       <- 0
  num_update  <- 0

  rmsprop <- new.env()
  rmsprop$lr <- lr
  rmsprop$count <- 0
  rmsprop$num_update <- 0

  create.state <- function(index, weight) {
      return (list(n=mx.nd.zeros(dim(weight), ctx(weight)),
                   g=mx.nd.zeros(dim(weight), ctx(weight)),
                   delta=mx.nd.zeros(dim(weight), ctx(weight))))
  }

  update <- function(index, weight, grad, state) {
    if (!is.null(lr_scheduler)){
      lr_scheduler(rmsprop) ## changing lr
      lr <- rmsprop$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = rmsprop, x = indexKey)){
        assign(x = indexKey, value = 0, envir = rmsprop)
      } else {
        indexValue <- get(envir = rmsprop, x = indexKey)
        assign(x = indexKey, value = indexValue + 1, envir = rmsprop)
        rmsprop$num_update <- max(rmsprop$num_update, get(envir = rmsprop, x = indexKey))
      }
    }
    grad <- grad * rescale.grad
    if (!is.null(clip_gradient)){
      if(clip_gradient >= 0){
          grad_ctx <- ctx(grad)
          grad <- as.array(grad)
          grad <- pmax(grad, -1 * clip_gradient)
          grad <- pmin(grad, clip_gradient)
          grad <- mx.nd.array(grad, grad_ctx)
      } else {
        stop("Error: clip_gradient should be positive number.")
      }
    }

    n <- state$n
    g <- state$g
    delta <- state$delta
    n <- gamma1 * n + (1 - gamma1) * (grad * grad)
    g <- gamma1 * g + (1 - gamma1) * grad
    delta <- gamma2 * delta - lr * (grad / mx.nd.sqrt(n - g*g + 1e-4) + wd * weight)
    weight <- weight + delta
    state <- list(n=n, g=g, delta=delta)

    return(list(weight=weight, state=state))
  }
  return(list(create.state=create.state, update=update))
}

#' Create an optimizer by name and parameters
#'
#' @param name The name of the optimizer
#' @param ... Additional arguments
#'
#' @export
mx.opt.create <- function(name, ...) {
  if (name == "sgd") {
    return(mx.opt.sgd(...))
  }
  else if (name == "rmsprop") {
    return (mx.opt.rmsprop(...))
  }
  stop(paste("Unknown optimizer ", name))
}

#' Get an updater closure that can take list of weight and gradient
#' and return updated list of weight.
#'
#' @param optimizer The optimizer
#' @param weights The weights to be optimized
#'
#' @export
mx.opt.get.updater <- function(optimizer, weights) {
  n <- length(weights)
  # This is the list to keep track of internal states of optimzer
  state.list <- lapply(1:n, function(i) {
    if (is.null(weights[[i]])) return(NULL)
    optimizer$create.state(i, weights[[i]])
  })
  update <- optimizer$update

  update.closure <- function(weight, grad) {
    ulist <- lapply(1:n, function(i) {
      if (!is.null(grad[[i]])) {
        update(i, weight[[i]], grad[[i]], state.list[[i]])
      } else {
        return(NULL)
      }
    })
    # update state list, use mutate assignment
    state.list <<- lapply(ulist, function(x) {
      x$state
    })
    # return updated weight list
    weight.list <- lapply(ulist, function(x) {
      x$weight
    })
    return(weight.list)
  }
  return(update.closure)
}
