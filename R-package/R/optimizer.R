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

#' Create an Adam optimizer with respective parameters.
#' Adam optimizer as described in [King2014].
#'
#' [King2014] Diederik Kingma, Jimmy Ba,
#' Adam: A Method for Stochastic Optimization,
#' http://arxiv.org/abs/1412.6980
#'
#' @param learning.rate float, default=0.001
#'      Step size.
#' @param beta1 float, default=0.9
#'      Exponential decay rate for the first moment estimates.
#' @param beta2 float, default=0.999
#'      Exponential decay rate for the second moment estimates.
#' @param epsilon float, default=1e-8
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
#'
mx.opt.adam <- function(learning.rate=0.001,
                        beta1=0.9,
                        beta2=0.999,
                        epsilon=1e-8,
                        wd=0,
                        rescale.grad=1,
                        clip_gradient = NULL,
                        lr_scheduler = NULL) {
  # use lr as short for learing rate.
  lr <- learning.rate
  count       <- 0
  num_update  <- 0

  adam <- new.env()
  adam$lr <- lr
  adam$count <- 0
  adam$num_update <- 0

  create.state <- function(index, weight) {
      return (list(mean=mx.nd.zeros(dim(weight), ctx(weight)),
                   variance=mx.nd.zeros(dim(weight), ctx(weight))))
  }

  update <- function(index, weight, grad, state) {
    if (!is.null(lr_scheduler)){
      lr_scheduler(adam) ## changing lr
      lr <- adam$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = adam, x = indexKey)){
        assign(x = indexKey, value = 0, envir = adam)
      } else {
        indexValue <- get(envir = adam, x = indexKey)
        assign(x = indexKey, value = indexValue + 1, envir = adam)
        adam$num_update <- max(adam$num_update, get(envir = adam, x = indexKey))
      }
    }

    # increment time
    time.key <- paste0('t', index)
    if (!exists(envir = adam, x = time.key)){
      assign(x = time.key, value = 0, envir = adam)
    }
    t <- get(envir = adam, x = time.key)
    t <- t + 1
    assign(x = time.key, value = t, envir = adam)

    mean <- state$mean
    variance <- state$variance

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

    mean <- beta1 * mean + (1 - beta1) * grad
    variance <- beta2 * variance + (1 - beta2) * (grad * grad)

    coef1 <- 1 - beta1^t
    coef2 <- 1 - beta2^t
    lr <- lr * sqrt(coef2)/coef1

    weight <- weight - lr * mean / (mx.nd.sqrt(variance) + epsilon)
    weight <- weight - lr * wd * weight

    state <- list(mean=mean, variance=variance)

    return(list(weight=weight, state=state))
  }
  return(list(create.state=create.state, update=update))
}

#' Create an AdaGrad optimizer with respective parameters.
#' AdaGrad optimizer of Duchi et al., 2011,
#'
#' This code follows the version in http://arxiv.org/pdf/1212.5701v1.pdf  Eq(5)
#' by Matthew D. Zeiler, 2012. AdaGrad will help the network to converge faster
#' in some cases.
#'
#' @param learning.rate float, default=0.05
#'      Step size.
#' @param epsilon float, default=1e-8
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
#'
mx.opt.adagrad <- function(learning.rate=0.05,
                           epsilon=1e-8,
                           wd=0,
                           rescale.grad=1,
                           clip_gradient = NULL,
                           lr_scheduler = NULL) {
  # use lr as short for learing rate.
  lr <- learning.rate
  count       <- 0
  num_update  <- 0

  adagrad <- new.env()
  adagrad$lr <- lr
  adagrad$count <- 0
  adagrad$num_update <- 0

  create.state <- function(index, weight) {
      return (mx.nd.zeros(dim(weight), ctx(weight))) #history
  }

  update <- function(index, weight, grad, state) {
    if (!is.null(lr_scheduler)){
      lr_scheduler(adagrad) ## changing lr
      lr <- adagrad$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = adagrad, x = indexKey)){
        assign(x = indexKey, value = 0, envir = adagrad)
      } else {
        indexValue <- get(envir = adagrad, x = indexKey)
        assign(x = indexKey, value = indexValue + 1, envir = adagrad)
        adagrad$num_update <- max(adagrad$num_update, get(envir = adagrad, x = indexKey))
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

    history <- state
    history <- history + (grad * grad)
    weight <- weight - lr * (grad / mx.nd.sqrt(history + epsilon) + wd * weight)
    state <- history

    return(list(weight=weight, state=state))
  }
  return(list(create.state=create.state, update=update))
}

#' Create an AdaDelta optimizer with respective parameters.
#'
#' AdaDelta optimizer as described in Zeiler, M. D. (2012).
#' *ADADELTA: An adaptive learning rate method.*
#' http://arxiv.org/abs/1212.5701
#'
#' @param rho float, default=0.90
#'      Decay rate for both squared gradients and delta x.
#' @param epsilon float, default=1e-5
#'      The constant as described in the thesis.
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional
#'      clip gradient in range [-clip_gradient, clip_gradient].
#'
mx.opt.adadelta <- function(rho=0.90,
                            epsilon=1e-5,
                            wd=0,
                            rescale.grad=1,
                            clip_gradient = NULL) {
  adadelta <- new.env()

  create.state <- function(index, weight) {
    return (list(acc.g=mx.nd.zeros(dim(weight), ctx(weight)),       # accumulated g
                 acc.delta=mx.nd.zeros(dim(weight), ctx(weight))))  # accumulated delta
  }

  update <- function(index, weight, grad, state) {
    # preprocess grad
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

    # accumulated g and delta initlization
    acc.g <- state$acc.g
    acc.delta <- state$acc.delta

    # update g, delta
    acc.g <- rho * acc.g + (1 - rho) * (grad * grad)
    current.delta <- mx.nd.sqrt(acc.delta + epsilon) / mx.nd.sqrt(acc.g + epsilon) * grad
    acc.delta <- rho * acc.delta + (1 - rho) * (current.delta * current.delta)
    weight <- weight - current.delta - wd * weight
    state <- list(acc.g=acc.g, acc.delta=acc.delta)

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
  else if (name == "adam") {
    return (mx.opt.adam(...))
  }
  else if (name == "adagrad") {
    return (mx.opt.adagrad(...))
  }
  else if (name == "adadelta") {
    return (mx.opt.adadelta(...))
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
