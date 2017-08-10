library(mxnet)

source("rnn.R")

# Internal function to do multiple device training on RNN
mx.model.train.rnn.buckets <- function(ctx, sym_list, arg.params, aux.params, input.shape, 
  output.shape, begin.round, end.round, optimizer, train.data, eval.data, metric, 
  epoch.end.callback, batch.end.callback, kvstore, verbose = TRUE) {
  symbol <- sym_list[[names(train.data$bucketID)]]
  
  input.names <- names(input.shape)
  output.names <- names(output.shape)
  arg.names <- names(arg.params)
  
  ndevice <- length(ctx)
  if (verbose) 
    message(paste0("Start training with ", ndevice, " devices"))
  input_slice <- mxnet:::mx.model.slice.shape(input.shape, ndevice)
  output_slice <- mxnet:::mx.model.slice.shape(output.shape, ndevice)
  
  
  # Grad request
  grad_req <- rep("write", length(symbol$arguments))
  # grad_null_idx <- match(c(input.names, output.names), symbol$arguments)
  grad_null_idx <- match(input.names, symbol$arguments)
  grad_req[grad_null_idx] <- "null"
  
  # Arg array order
  update_names <- c(input.names, output.names, arg.names)
  arg_update_idx <- match(symbol$arguments, update_names)
  
  train.execs <- lapply(1:ndevice, function(i) {
    s <- sapply(append(input_slice[[i]]$shape, output_slice[[i]]$shape), function(shape) {
      mx.nd.zeros(shape = shape, ctx = mx.cpu())
    })
    mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, arg.params)[arg_update_idx], 
      aux.arrays = aux.params, ctx = mx.cpu(), grad.req = grad_req)
  })
  
  # KVStore related stuffs
  params.index <- as.integer(mxnet:::mx.util.filter.null(lapply(1:length(train.execs[[1]]$ref.grad.arrays), 
    function(k) {
      if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL
    })))
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
      dlist <- train.data$value()  #[input.names]
      symbol <- sym_list[[names(train.data$bucketID)]]
      slices <- lapply(1:ndevice, function(i) {
        s <- input_slice[[i]]
        ret <- sapply(names(dlist), function(n) {
          mxnet:::mx.nd.slice(dlist[[n]], s$begin, s$end)
        })
        return(ret)
      })
      
      train.execs <- lapply(1:ndevice, function(i) {
        s <- slices[[i]]
        mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.names])[arg_update_idx], 
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
        # train.metric <- metric$update(dlist$label, out.preds, train.metric)
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
        dlist <- eval.data$value()  #[input.names]
        symbol <- sym_list[[names(eval.data$bucketID)]]
        slices <- lapply(1:ndevice, function(i) {
          s <- input_slice[[i]]
          ret <- sapply(names(dlist), function(n) {
          mxnet:::mx.nd.slice(dlist[[n]], s$begin, s$end)
          })
          return(ret)
        })
        
        
        train.execs <- lapply(1:ndevice, function(i) {
          s <- slices[[i]]
          mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.names])[arg_update_idx], 
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
