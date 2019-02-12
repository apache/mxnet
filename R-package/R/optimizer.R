# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#' Create an SGD optimizer with respective parameters.
#' Perform SGD with momentum update
#'
#' @param learning.rate float, default=0.01
#'      The initial learning rate.
#' @param momentum float, default=0
#'      The momentum value
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional, default=-1 (no clipping if < 0)
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
#' @param learning.rate float, default=0.002
#'      The initial learning rate.
#' @param gamma1 float, default=0.95
#'      decay factor of moving average for gradient, gradient^2.
#' @param gamma2 float, default=0.9
#'      "momentum" factor.
#' @param epsilon float, default=1e-4
#' @param wd float, default=0.0
#'      L2 regularization coefficient add to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional, default=-1 (no clipping if < 0)
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
#'
mx.opt.rmsprop <- function(learning.rate = 0.002,
                           centered = TRUE,
                           gamma1 = 0.95,
                           gamma2 = 0.9,
                           epsilon = 1e-4,
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
#' @param clip_gradient float, optional, default=-1 (no clipping if < 0)
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
      if (!exists(envir = adam, x = indexKey, inherits = FALSE)) {
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
#' @param clip_gradient float, default=-1.0 (no clipping if < 0)
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
#'
mx.opt.adagrad <- function(learning.rate = 0.05,
                           epsilon = 1e-8,
                           wd = 0,
                           rescale.grad = 1,
                           clip_gradient = -1,
                           lr_scheduler = NULL) {
  # use lr as short for learing rate.
  lr <- learning.rate
  count <- 0
  num_update <- 0

  adagrad <- new.env()
  adagrad$lr <- lr
  adagrad$count <- 0
  adagrad$num_update <- 0

  create_exec <- function(index, weight_dim, ctx) {

    weight <- mx.symbol.Variable("weight")
    grad <- mx.symbol.Variable("grad")
    history <- mx.symbol.Variable("history")

    grad <- grad * rescale.grad
    if (!is.null(clip_gradient)) {
      if (clip_gradient >= 0) {
        grad <- mx.symbol.clip(data = grad, a.min = -clip_gradient, a.max = clip_gradient)
      }
    }

    history <- history + (grad * grad)
    weight <- weight - lr * (grad / mx.symbol.sqrt(history + epsilon) + wd * weight)

    w <- mx.symbol.identity(weight, name = "w")
    h <- mx.symbol.identity(history, name = "h")
    sym <- mx.symbol.Group(c(w, h))

    exec <- mx.simple.bind(symbol = sym, weight = weight_dim, ctx = ctx, grad.req = "null")
    return(exec)
  }

  update <- function(index, exec_w, weight, grad) {
    if (!is.null(lr_scheduler)) {
      lr_scheduler(adagrad) ## changing lr
      lr <- adagrad$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = adagrad, x = indexKey, inherits = FALSE)) {
        adagrad[[indexKey]] <- 0
      } else {
        indexValue <- adagrad[[indexKey]]
        adagrad[[indexKey]] <- indexValue + 1
        adagrad$num_update <- max(adagrad$num_update, adagrad[[indexKey]])
      }
    }

    mx.exec.update.arg.arrays(exec_w, arg.arrays = list(weight = weight,grad = grad), match.name = T)
    mx.exec.forward(exec_w, is.train = F)

    # update state
    mx.exec.update.arg.arrays(exec_w, arg.arrays = list(history = exec_w$ref.outputs$h_output), match.name = T)

    return(exec_w$ref.outputs$w_output)
  }
  return(list(create_exec = create_exec, update = update))
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
#' @param rescale.grad float, default=1
#'      rescaling factor of gradient.
#' @param clip_gradient float, default=-1 (no clipping if < 0)
#'      clip gradient in range [-clip_gradient, clip_gradient].
#'
mx.opt.adadelta <- function(rho = 0.90,
                            epsilon = 1e-5,
                            wd = 0,
                            rescale.grad = 1,
                            clip_gradient = -1) {
  adadelta <- new.env()

  create_exec <- function(index, weight_dim, ctx) {
    weight <- mx.symbol.Variable("weight")
    grad <- mx.symbol.Variable("grad")
    acc.g <- mx.symbol.Variable("acc.g")
    acc.delta <- mx.symbol.Variable("acc.delta")

    grad <- grad * rescale.grad
    if (!is.null(clip_gradient)) {
      if (clip_gradient >= 0) {
        grad <- mx.symbol.clip(data = grad, a.min = -clip_gradient, a.max = clip_gradient)
      }
    }

    # update state (acc.g, acc.delta)
    acc.g <- rho * acc.g + (1 - rho) * (grad * grad)
    current.delta <- mx.symbol.sqrt(acc.delta + epsilon) / mx.symbol.sqrt(acc.g + epsilon) * grad
    acc.delta <- rho * acc.delta + (1 - rho) * (current.delta * current.delta)
    weight <- weight - current.delta - wd * weight

    w <- mx.symbol.identity(weight, name = "w")
    g <- mx.symbol.identity(acc.g, name = "g")
    delta <- mx.symbol.identity(acc.delta, name = "delta")
    sym <- mx.symbol.Group(c(w, g, delta))

    exec <- mx.simple.bind(symbol = sym, weight = weight_dim, ctx = ctx, grad.req = "null")
    return(exec)
  }

  update <- function(index, exec_w, weight, grad) {

    mx.exec.update.arg.arrays(exec_w, arg.arrays = list(weight = weight,grad = grad), match.name = T)
    mx.exec.forward(exec_w, is.train = F)

    # update state
    mx.exec.update.arg.arrays(exec_w,
                              arg.arrays = list(
                                acc.g = exec_w$ref.outputs$g_output,
                                acc.delta = exec_w$ref.outputs$delta_output),
                              match.name = T)

    return(exec_w$ref.outputs$w_output)
  }
  return(list(create_exec = create_exec, update = update))
}


#' Create a Nesterov Accelerated SGD( NAG) optimizer.
#'
#' NAG optimizer is described in Aleksandar Botev. et al (2016).
#' *NAG: A Nesterov accelerated SGD.*
#' https://arxiv.org/pdf/1607.01981.pdf
#'
#' @param learning.rate float, default=0.01
#'      The initial learning rate.
#' @param momentum float, default=0
#'      The momentum value
#' @param wd float, default=0.0
#'      L2 regularization coefficient added to all the weights.
#' @param rescale.grad float, default=1.0
#'      rescaling factor of gradient.
#' @param clip_gradient float, optional, default=-1 (no clipping if < 0)
#'      clip gradient in range [-clip_gradient, clip_gradient].
#' @param lr_scheduler function, optional
#'      The learning rate scheduler.
#'
mx.opt.nag <- function(learning.rate = 0.01,
                       momentum = 0,
                       wd = 0,
                       rescale.grad = 1,
                       clip_gradient = -1,
                       lr_scheduler = NULL) {

  lr <- learning.rate
  count <- 0
  num_update <- 0

  nag <- new.env()
  nag$lr <- learning.rate
  nag$count <- 0
  nag$num_update <- 0

  create_exec <- function(index, weight_dim, ctx) {

    weight <- mx.symbol.Variable("weight")
    grad <- mx.symbol.Variable("grad")
    mom <- mx.symbol.Variable("mom")
    grad <- grad * rescale.grad

    if (!is.null(clip_gradient)) {
      if (clip_gradient >= 0) {
        grad <- mx.symbol.clip(data = grad, a.min = -clip_gradient, a.max = clip_gradient)
      }
    }

    if (momentum == 0) {

      weight <- weight - lr * (grad + (wd * weight))
      w <- mx.symbol.identity(weight, name = "w")
      sym <- mx.symbol.Group(c(w))

    } else {

      mom <- momentum * mom + grad + wd * weight
      grad <- momentum * mom + grad
      weight <- weight - lr * grad

      w <- mx.symbol.identity(weight, name = "w")
      m <- mx.symbol.identity(mom, name = "m")
      sym <- mx.symbol.Group(c(w, m))

    }

    exec <- mx.simple.bind(symbol = sym, weight = weight_dim, ctx = ctx, grad.req = "null")
    return(exec)
  }

  update <- function(index, exec_w, weight, grad) {

    if (!is.null(lr_scheduler)){
      lr_scheduler(nag) ## changing lr
      lr <- nag$lr
      ## update count
      indexKey <- paste0('ik', index)
      if (!exists(envir = nag, x = indexKey, inherits = FALSE)){
        nag[[indexKey]] <- 0
      } else {
        indexValue <- nag[[indexKey]]
        nag[[indexKey]] <- indexValue + 1
        nag$num_update <- max(nag$num_update, nag[[indexKey]])
      }
    }

    mx.exec.update.arg.arrays(exec_w,
                              arg.arrays = list(weight = weight,grad = grad),
                              match.name = T)
    mx.exec.forward(exec_w, is.train = F)

    # update state
    if (!is.null(exec_w$ref.outputs$m_output)){
      mx.exec.update.arg.arrays(exec_w,
                                arg.arrays = list(mom = exec_w$ref.outputs$m_output),
                                match.name = T) 
    }

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
         "adagrad" = mx.opt.adagrad(...),
         "adadelta" = mx.opt.adadelta(...),
         "nag" = mx.opt.nag(...),
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
    if (is.null(weights[[i]])) {
      return(NULL)
    } else {
      optimizer$create_exec(index = i, weight_dim = dim(weights[[i]]), ctx = ctx)
    }
  })

  update <- optimizer$update

  update.closure <- function(weight, grad) {

    weight_list <- lapply(seq_along(weight), function(i) {
      if (!is.null(grad[[i]])) {
        return(update(i, exec_list[[i]], weight[[i]], grad[[i]]))
      } else {
        return(NULL)
      }
    })
    return(weight_list)
  }
  return(update.closure)
}
