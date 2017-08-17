# Internal function to do multiple device training on RNN
mx.model.train.rnn.buckets <- function(ctx, symbol, arg.params, aux.params, input.shape, 
                                       output.shape, begin.round, end.round, optimizer, train.data, eval.data, metric, 
                                       epoch.end.callback, batch.end.callback, kvstore, verbose = TRUE) {
  
  # symbol <- sym_list[[names(train.data$bucketID)]]
  
  input.names <- names(input.shape)
  output.names <- names(output.shape)
  arg.params.names <- names(arg.params)
  
  ndevice <- length(ctx)
  if (verbose) 
    message(paste0("Start training with ", ndevice, " devices"))
  
  # Grad request
  grad_req <- rep("write", length(symbol$arguments))
  grad_null_idx <- match(c(input.names, output.names), symbol$arguments)
  grad_req[grad_null_idx] <- "null"
  
  # Arg array order
  update_names <- c(input.names, output.names, arg.params.names)
  arg_update_idx <- match(symbol$arguments, update_names)
  
  # Initial binding
  dlist <- lapply(c(input.shape, output.shape), function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu()) 
  })
  
  slices <- lapply(1:ndevice, function(i) {
    sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = F))
  })
  
  train.execs <- lapply(1:ndevice, function(i) {
    s <- slices[[i]]
    mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, arg.params)[arg_update_idx], 
                           aux.arrays = aux.params, ctx = ctx[[i]], grad.req = grad_req)
  })
  
  # KVStore related stuffs
  params.index <- as.integer(
    mxnet:::mx.util.filter.null(
      lapply(1:length(train.execs[[1]]$ref.grad.arrays), function(k) {
        if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL
      }
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
  
  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    train.data$reset()
    while (train.data$iter.next()) {
      dlist <- train.data$value()
      # symbol <- sym_list[[names(train.data$bucketID)]]
      
      # Slice inputs for multi-devices
      slices <- lapply(1:ndevice, function(i) {
        sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = F))
      })
      
      train.execs <- lapply(1:ndevice, function(i) {
        s <- slices[[i]]
        mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.params.names])[arg_update_idx], 
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
          train.metric <- metric$update(slices[[i]][[length(slices[[i]])]], 
                                        out.preds[[i]], train.metric)
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
        # Get input data slice
        dlist <- eval.data$value()
        # symbol <- sym_list[[names(eval.data$bucketID)]]
        
        slices <- lapply(1:ndevice, function(i) {
          sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = F))
        })
        
        train.execs <- lapply(1:ndevice, function(i) {
          s <- slices[[i]]
          mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.params.names])[arg_update_idx], 
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
            eval.metric <- metric$update(slices[[i]][[length(slices[[i]])]], 
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
mx.rnn.buckets <- function(symbol, train.data, eval.data = NULL, 
                           ctx = NULL, num.round = 1, begin.round = 1, 
                           initializer = mx.init.uniform(0.01), optimizer = "sgd", metric = mx.metric.rmse, 
                           batch.end.callback = NULL, epoch.end.callback = NULL, 
                           kvstore = "local", verbose = TRUE) {
  
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
  
  # get list of bucketed symbol - no longer needed with mx.symbol.RNN
  # sym_list <- sapply(train.data$bucket.names, function(x) {
  #   rnn.graph(num.rnn.layer = num.rnn.layer, num.hidden = num.hidden, 
  #             input.size = input.size, num.embed = num.embed, num.label = num.label, 
  #             dropout = dropout, cell.type = cell.type, config = config)
  # }, simplify = F, USE.NAMES = T)
  # symbol <- sym_list[[names(train.data$bucketID)]]
  
  input.names <- c("data", "seq.mask")
  input.shape <- sapply(input.names, function(n) {
    dim(train.data$value()[[n]])
  }, simplify = FALSE)
  output.names <- "label"
  output.shape <- sapply(output.names, function(n) {
    dim(train.data$value()[[n]])
  }, simplify = FALSE)
  
  params <- mx.model.init.params(symbol, input.shape, output.shape, initializer, 
                                 mx.cpu())
  
  kvstore <- mxnet:::mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), 
                                             verbose = verbose)
  
  ### Execute training
  model <- mx.model.train.rnn.buckets(symbol = symbol, input.shape = input.shape, output.shape = output.shape, 
                                      arg.params = params$arg.params, aux.params = params$aux.params, 
                                      optimizer = optimizer, train.data = train.data, eval.data = eval.data, verbose = verbose, 
                                      begin.round = begin.round, end.round = num.round, metric = metric, ctx = ctx, 
                                      batch.end.callback = batch.end.callback, epoch.end.callback = epoch.end.callback, 
                                      kvstore = kvstore)
  
  return(model)
}
