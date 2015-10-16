#' Create an SGD optimizer with respective parameters.
#' Perform SGD with momentum update
#'
mx.opt.sgd <- function(learning.rate,
                       momentum=0,
                       wd=0,
                       rescale.grad=1) {
  # use lr as short for learing rate.
  lr <- learning.rate
  create.state <- function(index, weight) {
    if (momentum == 0) {
      return(NULL)
    } else {
      ret <- (mx.nd.zeros(dim(weight), ctx(weight)))
      return(ret)
    }
  }
  update <- function(index, weight, grad, state) {
    if (is.null(state)) {
      weight <- weight - lr * (grad * rescale.grad + wd * weight)
    } else {
      mom <- state
      mom <- mom * momentum
      mom <- mom - lr * (grad * rescale.grad + wd * weight)
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
