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
