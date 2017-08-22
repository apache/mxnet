# Internal function to do multiple device training on RNN
mx.model.train.rnn.buckets <- function(ctx, symbol, 
                                       arg.params, aux.params, 
                                       dlist, arg.params.fix, 
                                       begin.round, end.round, optimizer, train.data, eval.data, metric, 
                                       epoch.end.callback, batch.end.callback, kvstore, verbose = TRUE) {
  
  arguments <- symbol$arguments
  arg.params.fix.names <- names(arg.params.fix)
  arg.params.names <- names(arg.params)
  input.names <- names(dlist)
  
  ndevice <- length(ctx)
  if (verbose) 
    message(paste0("Start training with ", ndevice, " devices"))
  
  # Grad request
  grad_req <- rep("null", length(arguments))
  grad.req.write <- arguments %in% names(arg.params)
  grad_req[grad.req.write] <- "write"
  
  # Arg array order
  update_names <- c(input.names, arg.params.fix.names, arg.params.names)
  arg_update_idx <- match(arguments, update_names)
  
  # Initial binding
  # dlist <- lapply(input.shape, function(shape) {
  #   mx.nd.zeros(shape = shape, ctx = mx.cpu()) 
  # })
  
  slices <- lapply(1:ndevice, function(i) {
    sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = F))
  })
  
  train.execs <- lapply(1:ndevice, function(i) {
    s <- slices[[i]]
    mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, arg.params.fix, arg.params)[arg_update_idx], 
                           aux.arrays = aux.params, ctx = ctx[[i]], grad.req = grad_req)
  })
  
  # KVStore related stuffs
  params.index <- as.integer(
    mxnet:::mx.util.filter.null(
      lapply(1:length(train.execs[[1]]$ref.grad.arrays), function(k) {
        if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL}
      )))
  
  update.on.kvstore <- FALSE
  if (!is.null(kvstore) && kvstore$update.on.kvstore) {
    update.on.kvstore <- TRUE
    kvstore$set.optimizer(optimizer)
  } else {
    updaters <- lapply(1:ndevice, function(i) {
      mx.opt.get.updater(optimizer, train.execs[[i]]$ref.arg.arrays)
    })
  }
  
  if (!is.null(kvstore)) {
    kvstore$init(params.index, train.execs[[1]]$ref.arg.arrays[params.index])
  }
  
  # train over specified number of epochs
  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    train.data$reset()
    while (train.data$iter.next()) {
      
      # Get iterator data
      dlist <- train.data$value()
      
      # Slice inputs for multi-devices
      slices <- lapply(1:ndevice, function(i) {
        sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = F))
      })
      
      # Assign input to each executor - bug on inference if using BatchNorm
      train.execs <- lapply(1:ndevice, function(i) {
        s <- slices[[i]]
        mxnet:::mx.symbol.bind(symbol = symbol, 
                               arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.params.fix.names], train.execs[[i]]$arg.arrays[arg.params.names])[arg_update_idx],
                               aux.arrays = train.execs[[i]]$aux.arrays, ctx = ctx[[i]], grad.req = grad_req)
      })
      
      for (texec in train.execs) {
        mx.exec.forward(texec, is.train = TRUE)
      }
      
      out.preds <- lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
      })
      
      for (texec in train.execs) {
        mx.exec.backward(texec)
      }
      
      if (!is.null(kvstore)) {
        # push the gradient
        kvstore$push(params.index, lapply(train.execs, function(texec) {
          texec$ref.grad.arrays[params.index]
        }), -params.index)
      }
      if (update.on.kvstore) {
        # pull back weight
        kvstore$pull(params.index, lapply(train.execs, function(texec) {
          texec$ref.arg.arrays[params.index]
        }), -params.index)
      } else {
        # pull back gradient sums
        if (!is.null(kvstore)) {
          kvstore$pull(params.index, lapply(train.execs, function(texec) {
            texec$ref.grad.arrays[params.index]
          }), -params.index)
        }
        arg.blocks <- lapply(1:ndevice, function(i) {
          updaters[[i]](train.execs[[i]]$ref.arg.arrays, train.execs[[i]]$ref.grad.arrays)
        })
        for (i in 1:ndevice) {
          mx.exec.update.arg.arrays(train.execs[[i]], arg.blocks[[i]], skip.null = TRUE)
        }
      }
      
      # Update the evaluation metrics
      if (!is.null(metric)) {
        for (i in 1:ndevice) {
          train.metric <- metric$update(label = mx.nd.reshape(slices[[i]][[length(slices[[i]])]], shape=-1), 
                                        pred = out.preds[[i]], state = train.metric)
        }
      }
      
      nbatch <- nbatch + 1
      
      if (!is.null(batch.end.callback)) {
        batch.end.callback(iteration, nbatch, environment())
      }
    }
    
    if (!is.null(metric)) {
      result <- metric$get(train.metric)
      if (verbose) 
        message(paste0("[", iteration, "] Train-", result$name, "=", result$value))
    }
    
    if (!is.null(eval.data)) {
      if (!is.null(metric)) {
        eval.metric <- metric$init()
      }
      eval.data$reset()
      while (eval.data$iter.next()) {
        
        # Get iterator data
        dlist <- eval.data$value()
        
        # Slice input to multiple devices
        slices <- lapply(1:ndevice, function(i) {
          sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = F))
        })
        
        # Assign input to each executor
        train.execs <- lapply(1:ndevice, function(i) {
          s <- slices[[i]]
          mxnet:::mx.symbol.bind(symbol = symbol, 
                                 arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.params.fix.names], train.execs[[i]]$arg.arrays[arg.params.names])[arg_update_idx],
                                 aux.arrays = train.execs[[i]]$aux.arrays, ctx = ctx[[i]], grad.req = grad_req)
        })
        
        for (texec in train.execs) {
          mx.exec.forward(texec, is.train = FALSE)
        }
        
        # copy outputs to CPU
        out.preds <- lapply(train.execs, function(texec) {
          mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
        })
        
        if (!is.null(metric)) {
          for (i in 1:ndevice) {
            eval.metric <- metric$update(mx.nd.reshape(slices[[i]][[length(slices[[i]])]], shape=-1), 
                                         out.preds[[i]], eval.metric)
          }
        }
      }
      
      if (!is.null(metric)) {
        result <- metric$get(eval.metric)
        if (verbose) {
          message(paste0("[", iteration, "] Validation-", result$name, "=", 
                         result$value))
        }
      }
    } else {
      eval.metric <- NULL
    }
    # get the model out
    model <- mxnet:::mx.model.extract.model(symbol, train.execs)
    
    epoch_continue <- TRUE
    if (!is.null(epoch.end.callback)) {
      epoch_continue <- epoch.end.callback(iteration, 0, environment(), verbose = verbose)
    }
    
    if (!epoch_continue) {
      break
    }
  }
  return(model)
}


# 
#' Train RNN with bucket support
#'
#' @param symbol Symbolic representation of the model
#' @param train.data Training data created by mx.io.bucket.iter
#' @param eval.data Evaluation data created by mx.io.bucket.iter
#' @param num.round int, number of epoch
#' @param initializer
#' @param optimizer
#' @param batch.end.callback
#' @param epoch.end.callback
#' @param begin.round
#' @param metric
#' @param ctx
#' @param kvstore
#' @param verbose
#'
#' @export
mx.rnn.buckets <- function(symbol, train.data, eval.data = NULL, init.state = NULL, 
                           ctx = NULL, num.round = 1, begin.round = 1, 
                           initializer = mx.init.uniform(0.01), optimizer = "sgd", metric = mx.metric.rmse, 
                           batch.end.callback = NULL, epoch.end.callback = NULL, 
                           kvstore = "local", verbose = TRUE,
                           input.params = NULL) {
  
  if (!train.data$iter.next()) {
    train.data$reset()
    if (!train.data$iter.next()) 
      stop("Empty train.data")
  }
  
  if (!is.null(eval.data)) {
    if (!eval.data$iter.next()) {
      eval.data$reset()
      if (!eval.data$iter.next()) 
        stop("Empty eval.data")
    }
  }
  
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) 
    stop("ctx must be mx.context or list of mx.context")
  if (is.character(optimizer)) {
    if (is.numeric(input.shape)) {
      ndim <- length(input.shape)
      batchsize <- input.shape[[ndim]]
    } else {
      ndim <- length(input.shape[[1]])
      batchsize <- input.shape[[1]][[ndim]]
    }
    optimizer <- mx.opt.create(optimizer, rescale.grad = (1/batchsize), ...)
  }
  
  arguments <- symbol$arguments
  input.names <- names(train.data$value())
  
  input.shape <- sapply(input.names, function(n) {
    dim(train.data$value()[[n]])
  }, simplify = FALSE)
  
  shapes <- symbol$infer.shape(input.shape)
  
  # initialize all arguments with zeros
  arguments.ini <- lapply(shapes$arg.shapes, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
    
  arg.params <- mx.init.create(initializer = initializer, shape.array = shapes$arg.shapes, ctx = mx.cpu(), skip.unknown = TRUE)
  
  dlist <- arguments.ini[input.names]
  
  # Assign fixed parameters to their value and keep non initialized arguments to zero
  arg.params.fix.names <- unique(c(names(input.params), setdiff(arguments, c(names(arg.params), input.names))))
  
  # Assign zeros to non initialized arg parameters
  arg.params.fix <- arguments.ini[arg.params.fix.names]
  # Assign weights to arguments specifies by input.params
  arg.params.fix[names(input.params)] <- input.params
  
  # aux parameters setup
  aux.params <- lapply(shapes$aux.shapes, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  aux.params.ini <- mx.init.create(initializer, shapes$aux.shapes, ctx = mx.cpu(), skip.unknown = FALSE)
  if (length(aux.params) > 0) {
    aux.params[names(aux.params.ini)] <- aux.params.ini
  } else aux.params <- NULL
  
  # arg.arrays <- sapply(slist$arg.shapes, function(shape) {
  #   mx.nd.zeros(shape, ctx)
  # }, simplify = FALSE, USE.NAMES = TRUE)
  # 
  # arg.params <- mx.init.create(initializer, shapes$arg.shapes, ctx = mx.cpu(), skip.unknown = TRUE)
  # 
  # params <- mx.model.init.params(symbol = symbol, input.shape = input.shape, output.shape = NULL, initializer, mx.cpu())
  
  kvstore <- mxnet:::mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), 
                                             verbose = verbose)
  
  ### Execute training
  model <- mx.model.train.rnn.buckets(symbol = symbol, 
                                      arg.params = arg.params, aux.params = aux.params, 
                                      dlist = dlist, arg.params.fix = arg.params.fix,   
                                      optimizer = optimizer, train.data = train.data, eval.data = eval.data, 
                                      begin.round = begin.round, end.round = num.round, metric = metric, ctx = ctx, 
                                      batch.end.callback = batch.end.callback, epoch.end.callback = epoch.end.callback, 
                                      kvstore = kvstore, verbose = verbose)
  
  return(model)
}
