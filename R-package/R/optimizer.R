#' Create an SGD optimizer with respective parameters.
#' Perform SGD with momentum update
#'
#' @param learning.rate float, default=1e-3
#'      The initial learning rate.
#' @param momentum float, default=0
#'      The momentumvalue
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional, default=-1
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
mx.opt.sgd <- function(learning.rate = 0.01,
                       momentum = 0,
                       wd = 0,
                       rescale.grad = 1,
                       clip_gradient = -1,
                       lr_scheduler = NULL) {
  
  lr <- learning.rate
  count <- 0
  num_update <- 0
  
  sgd <- new.env()
  sgd$lr <- lr
  sgd$count <- 0
  sgd$num_update <- 0
  
  create_exec <- function(index, weight_dim, ctx) {
    
    if (momentum == 0) {
      
      weight <- mx.symbol.Variable("weight")
      grad <- mx.symbol.Variable("grad")
      
      sym <- mx.symbol.sgd_update(weight,
                                  grad,
                                  lr = lr,
                                  wd = wd,
                                  rescale_grad = rescale.grad,
                                  clip_gradient = clip_gradient,
                                  name = "w")
    } else {
      
      weight <- mx.symbol.Variable("weight")
      grad <- mx.symbol.Variable("grad")
      mom <- mx.symbol.Variable("mom")
      
      sym <- mx.symbol.sgd_mom_update(weight,
                                      grad,
                                      mom,
                                      lr = lr,
                                      wd = wd,
                                      momentum= momentum,
                                      rescale_grad = rescale.grad,
                                      clip_gradient = clip_gradient,
                                      name = "w")
    }
    exec <- mx.simple.bind(symbol = sym, weight = weight_dim, ctx = ctx, grad.req = "null")
    return(exec)
  }
  
  update <- function(index, exec_w, weight, grad) {
    
    if (!is.null(lr_scheduler)){
      lr_scheduler(sgd) ## changing lr
      lr <- sgd$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = sgd, x = indexKey, inherits = FALSE)){
        sgd[[indexKey]] <- 0
      } else {
        indexValue <- sgd[[indexKey]]
        sgd[[indexKey]] <- indexValue + 1
        sgd$num_update <- max(sgd$num_update, sgd[[indexKey]])
      }
    }
    
    mx.exec.update.arg.arrays(exec_w, arg.arrays = list(weight = weight,grad = grad), match.name = T)
    mx.exec.forward(exec_w, is.train = F)
    return(exec_w$ref.outputs$w_output)
  }
  return(list(create_exec = create_exec, update = update))
}

#' Create an RMSProp optimizer with respective parameters.
#' Reference: Tieleman T, Hinton G. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude[J]. COURSERA: Neural Networks for Machine Learning, 2012, 4(2).
#' The code follows: http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
#'
#' @param learning.rate float, default=1e-3
#'      The initial learning rate.
#' @param gamma1 float, default=0.9
#'      decay factor of moving average for gradient, gradient^2.
#' @param gamm2 float, default=0.9
#'      "momentum" factor.
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional, default=-1
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
#'
mx.opt.rmsprop <- function(learning.rate = 1e-3,
                           centered = TRUE,
                           gamma1 = 0.9,
                           gamma2 = 0.9,
                           epsilon = 1e-8,
                           wd = 0,
                           rescale.grad = 1,
                           clip_gradient = -1,
                           lr_scheduler = NULL) {
  
  lr <- learning.rate
  count <- 0
  num_update <- 0
  
  rmsprop <- new.env()
  rmsprop$lr <- lr
  rmsprop$count <- 0
  rmsprop$num_update <- 0
  
  create_exec <- function(index, weight_dim, ctx) {
    
    if (centered) {
      
      weight <- mx.symbol.Variable("weight")
      grad <- mx.symbol.Variable("grad")
      n <- mx.symbol.Variable("n")
      g <- mx.symbol.Variable("g")
      delta <- mx.symbol.Variable("delta")
      
      sym <- mx.symbol.rmspropalex_update(weight,
                                          grad,
                                          n,
                                          g,
                                          delta,
                                          lr = lr,
                                          gamma1 = gamma1,
                                          gamma2 = gamma2,
                                          epsilon = epsilon,
                                          wd = wd,
                                          rescale_grad = rescale.grad,
                                          clip_gradient = clip_gradient,
                                          name = "w")
    } else {
      weight <- mx.symbol.Variable("weight")
      grad <- mx.symbol.Variable("grad")
      n <- mx.symbol.Variable("n")
      
      sym <- mx.symbol.rmsprop_update(weight,
                                      grad,
                                      n,
                                      lr = lr,
                                      gamma1 = gamma1,
                                      epsilon = epsilon,
                                      wd = wd,
                                      rescale_grad = rescale.grad,
                                      clip_gradient = clip_gradient,
                                      name = "w")
    }
    
    exec <- mx.simple.bind(symbol = sym, weight = weight_dim, ctx = ctx, grad.req = "null")
    return(exec)
  }
  
  update <- function(index, exec_w, weight, grad) {
    if (!is.null(lr_scheduler)){
      lr_scheduler(rmsprop) ## changing lr
      lr <- rmsprop$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = rmsprop, x = indexKey, inherits = FALSE)) {
        rmsprop[[indexKey]] <- 0
      } else {
        indexValue <- rmsprop[[indexKey]]
        rmsprop[[indexKey]] <- indexValue + 1
        rmsprop$num_update <- max(rmsprop$num_update, rmsprop[[indexKey]])
      }
    }
    
    mx.exec.update.arg.arrays(exec_w, arg.arrays = list(weight = weight,grad = grad), match.name = T)
    mx.exec.forward(exec_w, is.train = F)
    return(exec_w$ref.outputs$w_output)
  }
  return(list(create_exec = create_exec, update = update))
}

#' Create an Adam optimizer with respective parameters.
#' Adam optimizer as described in [King2014].
#'
#' [King2014] Diederik Kingma, Jimmy Ba,
#' Adam: A Method for Stochastic Optimization,
#' http://arxiv.org/abs/1412.6980
#'
#' @param learning.rate float, default=1e-3
#'      The initial learning rate.
#' @param beta1 float, default=0.9
#'      Exponential decay rate for the first moment estimates.
#' @param beta2 float, default=0.999
#'      Exponential decay rate for the second moment estimates.
#' @param epsilon float, default=1e-8
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional, default=-1
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
#'
mx.opt.adam <- function(learning.rate = 1e-3,
                        beta1 = 0.9,
                        beta2 = 0.999,
                        epsilon = 1e-8,
                        wd = 0,
                        rescale.grad = 1,
                        clip_gradient = -1,
                        lr_scheduler = NULL) {
  
  lr <- learning.rate
  count <- 0
  num_update <- 0
  
  adam <- new.env()
  adam$lr <- lr
  adam$count <- 0
  adam$num_update <- 0
  
  create_exec <- function(index, weight_dim, ctx) {
    
    weight <- mx.symbol.Variable("weight")
    grad <- mx.symbol.Variable("grad")
    mean <- mx.symbol.Variable("mean")
    var <- mx.symbol.Variable("var")
    
    sym <- mx.symbol.adam_update(weight,
                                 grad,
                                 mean,
                                 var,
                                 lr = lr,
                                 beta1 = beta1,
                                 beta2 = beta2,
                                 epsilon = epsilon,
                                 wd = wd,
                                 rescale_grad = rescale.grad,
                                 clip_gradient = clip_gradient,
                                 name = "w")
    
    exec <- mx.simple.bind(symbol = sym, weight = weight_dim, ctx = ctx, grad.req = "null")
    return(exec)
  }
  
  update <- function(index, exec_w, weight, grad) {
    if (!is.null(lr_scheduler)){
      lr_scheduler(adam) ## changing lr
      lr <- adam$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = adam, x = indexKey, inherits = FALSE)){
        adam[[indexKey]] <- 0
      } else {
        indexValue <- adam[[indexKey]]
        adam[[indexKey]] <- indexValue + 1
        adam$num_update <- max(adam$num_update, adam[[indexKey]])
      }
    }
    
    mx.exec.update.arg.arrays(exec_w, arg.arrays = list(weight = weight,grad = grad), match.name = T)
    mx.exec.forward(exec_w, is.train = F)
    return(exec_w$ref.outputs$w_output)
  }
  return(list(create_exec = create_exec, update = update))
}


#' Create an optimizer by name and parameters
#'
#' @param name The name of the optimizer
#' @param ... Additional arguments
#'
#' @export
mx.opt.create <- function(name, ...) {
  switch(name,
         "sgd" = mx.opt.sgd(...),
         "rmsprop" = mx.opt.rmsprop(...),
         "adam" = mx.opt.adam(...),
         # "adagrad" = mx.opt.adagrad(...),
         # "adadelta" = mx.opt.adadelta(...),
         stop("Unknown optimizer ", name))
}

#' Get an updater closure that can take list of weight and gradient
#' and return updated list of weight.
#'
#' @param optimizer The optimizer
#' @param weights The weights to be optimized
#'
#' @export
mx.opt.get.updater <- function(optimizer, weights, ctx) {
  
  exec_list <- lapply(seq_along(weights), function(i) {
    if (is.null(weights[[i]])) return(NULL) else
      optimizer$create_exec(index = i, weight_dim = dim(weights[[i]]), ctx = ctx)
  })
  
  update <- optimizer$update
  
  update.closure <- function(weight, grad) {
    
    weight_list <- lapply(seq_along(weight), function(i) {
      if (!is.null(grad[[i]])) return(update(i, exec_list[[i]], weight[[i]], grad[[i]])) else
        return(NULL)
    })
    return(weight_list)
  }
  return(update.closure)
}
