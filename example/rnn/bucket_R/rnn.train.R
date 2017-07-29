library(mxnet)

source("rnn.R")

# Internal function to do multiple device training on RNN
mx.model.train.rnn.buckets <- function(ctx, sym_list, arg.params, aux.params, input.shape, 
  begin.round, end.round, optimizer, train.data, eval.data, metric, epoch.end.callback, 
  batch.end.callback, verbose = TRUE) {
  symbol <- sym_list[[names(train.data$bucketID)]]
  
  input.names <- names(input.shape)
  arg.names <- names(arg.params)
  
  # Grad request
  grad_req <- rep("write", length(symbol$arguments))
  grad_null_idx <- match(input.names, symbol$arguments)
  grad_req[grad_null_idx] <- "null"
  
  # Arg array order
  update_names <- c(input.names, arg.names)
  arg_update_idx <- match(symbol$arguments, update_names)
  
  s <- sapply(input.shape, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  train.exec <- mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, arg.params)[arg_update_idx], 
    aux.arrays = aux.params, ctx = ctx, grad.req = grad_req)
  
  updaters <- mx.opt.get.updater(optimizer, train.exec$ref.arg.arrays)
  
  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    train.data$reset()
    while (train.data$iter.next()) {
      dlist <- train.data$value()[input.names]
      symbol <- sym_list[[names(train.data$bucketID)]]
      
      train.exec <- mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(dlist, 
        train.exec$arg.arrays[arg.names])[arg_update_idx], aux.arrays = train.exec$aux.arrays, 
        ctx = ctx, grad.req = grad_req)
      
      mx.exec.forward(train.exec, is.train = TRUE)
      
      # copy outputs to CPU
      out.preds <- mx.nd.copyto(train.exec$ref.outputs[[1]], mx.cpu())
      
      mx.exec.backward(train.exec)
      
      arg.blocks <- updaters(train.exec$ref.arg.arrays, train.exec$ref.grad.arrays)
      mx.exec.update.arg.arrays(train.exec, arg.blocks, skip.null = TRUE)
      
      # Update the evaluation metrics
      if (!is.null(metric)) {
        train.metric <- metric$update(dlist$label, out.preds, train.metric)
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
        dlist <- eval.data$value()[input.names]
        symbol <- sym_list[[names(eval.data$bucketID)]]
        train.exec <- mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(dlist, 
          train.exec$arg.arrays[arg.names])[arg_update_idx], aux.arrays = train.exec$aux.arrays, 
          ctx = ctx, grad.req = grad_req)
        
        mx.exec.forward(train.exec, is.train = FALSE)
        
        # copy outputs to CPU
        out.preds <- mx.nd.copyto(train.exec$ref.outputs[[1]], mx.cpu())
        
        if (!is.null(metric)) {
          eval.metric <- metric$update(dlist$label, out.preds, eval.metric)
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
    model <- mxnet:::mx.model.extract.model(symbol, list(train.exec))
    
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
