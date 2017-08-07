library(mxnet)

source("lstm.cell.R")
source("gru.cell.R")

# unrolled RNN network
rnn.unroll <- function(num.rnn.layer, seq.len, input.size, num.embed, num.hidden, 
  num.label, dropout = 0, ignore_label = 0, init.state = NULL, config, cell.type = "lstm", 
  output_last_state = F) {
  embed.weight <- mx.symbol.Variable("embed.weight")
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  param.cells <- lapply(1:num.rnn.layer, function(i) {
    if (cell.type == "lstm") {
      cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")), 
        i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")), h2h.weight = mx.symbol.Variable(paste0("l", 
          i, ".h2h.weight")), h2h.bias = mx.symbol.Variable(paste0("l", i, 
          ".h2h.bias")))
    } else if (cell.type == "gru") {
      cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")), 
        gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")), 
        gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")), 
        gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")), 
        trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")), 
        trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")), 
        trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")), 
        trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
    }
    return(cell)
  })
  
  # embeding layer
  label <- mx.symbol.Variable("label")
  data <- mx.symbol.Variable("data")
  data_mask_array <- mx.symbol.Variable("data.mask.array")
  data_mask_array <- mx.symbol.stop_gradient(data_mask_array, name = "data.mask.array")
  
  embed <- mx.symbol.Embedding(data = data, input_dim = input.size, weight = embed.weight, 
    output_dim = num.embed, name = "embed")
  
  wordvec <- mx.symbol.split(data = embed, axis = 1, num.outputs = seq.len, squeeze_axis = T)
  data_mask_split <- mx.symbol.split(data = data_mask_array, axis = 1, num.outputs = seq.len, 
    squeeze_axis = T)
  
  last.hidden <- list()
  last.states <- list()
  decode <- list()
  softmax <- list()
  fc <- list()
  
  for (seqidx in 1:seq.len) {
    hidden <- wordvec[[seqidx]]
    
    for (i in 1:num.rnn.layer) {
      if (seqidx == 1) {
        prev.state <- init.state[[i]]
      } else {
        prev.state <- last.states[[i]]
      }
      
      if (cell.type == "lstm") {
        cell.symbol <- lstm.cell
      } else if (cell.type == "gru") {
        cell.symbol <- gru.cell
      }
      
      next.state <- cell.symbol(num.hidden = num.hidden, indata = hidden, prev.state = prev.state, 
        param = param.cells[[i]], seqidx = seqidx, layeridx = i, dropout = dropout, 
        data_masking = data_mask_split[[seqidx]])
      hidden <- next.state$h
      # if (dropout > 0) hidden <- mx.symbol.Dropout(data=hidden, p=dropout)
      last.states[[i]] <- next.state
    }
    
    # Decoding
    if (config == "one-to-one") {
      last.hidden <- c(last.hidden, hidden)
    }
  }
  
  if (config == "seq-to-one") {
    fc <- mx.symbol.FullyConnected(data = hidden, weight = cls.weight, bias = cls.bias, 
      num.hidden = num.label)
    
    loss <- mx.symbol.SoftmaxOutput(data = fc, name = "sm", label = label, ignore_label = ignore_label)
    
  } else if (config == "one-to-one") {
    last.hidden_expand <- lapply(last.hidden, function(i) mx.symbol.expand_dims(i, 
      axis = 1))
    concat <- mx.symbol.concat(last.hidden_expand, num.args = seq.len, dim = 1)
    reshape <- mx.symbol.Reshape(concat, shape = c(num.hidden, -1))
    
    fc <- mx.symbol.FullyConnected(data = reshape, weight = cls.weight, bias = cls.bias, 
      num.hidden = num.label)
    
    label <- mx.symbol.reshape(data = label, shape = c(-1))
    loss <- mx.symbol.SoftmaxOutput(data = fc, name = "sm", label = label, ignore_label = ignore_label)
    
  }
  
  if (output_last_state) {
    group <- mx.symbol.Group(c(unlist(last.states), loss))
    return(group)
  } else {
    return(loss)
  }
}

########################################### mx.rnn.buckets
mx.rnn.buckets <- function(train.data, eval.data = NULL, num.rnn.layer, num.hidden, 
  num.embed, num.label, input.size, ctx = NULL, num.round = 1, initializer = mx.init.uniform(0.01), 
  dropout = 0, config = "one-to-one", optimizer = "sgd", batch.end.callback = NULL, 
  epoch.end.callback = NULL, begin.round = 1, metric = mx.metric.rmse, cell.type = "lstm", 
  kvstore = "local", verbose = FALSE) {
  
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
  
  # get unrolled lstm symbol
  sym_list <- sapply(train.data$bucket.names, function(x) {
    rnn.unroll(num.rnn.layer = num.rnn.layer, num.hidden = num.hidden, seq.len = as.integer(x), 
      input.size = input.size, num.embed = num.embed, num.label = num.label, 
      dropout = dropout, cell.type = cell.type, config = config)
  }, simplify = F, USE.NAMES = T)
  
  # setup lstm model
  symbol <- sym_list[[names(train.data$bucketID)]]
  
  arg.names <- symbol$arguments
  input.names <- c("data", "data.mask.array")
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
  
  ### Execute training - rnn.model.R
  model <- mx.model.train.rnn.buckets(sym_list = sym_list, input.shape = input.shape, 
    output.shape = output.shape, arg.params = params$arg.params, aux.params = params$aux.params, 
    optimizer = optimizer, train.data = train.data, eval.data = eval.data, verbose = verbose, 
    begin.round = begin.round, end.round = num.round, metric = metric, ctx = ctx, 
    batch.end.callback = batch.end.callback, epoch.end.callback = epoch.end.callback, 
    kvstore = kvstore)
  
  return(model)
}


# get the argument name of data and label
mx.model.check.arguments <- function(symbol) {
  data <- NULL
  label <- NULL
  for (nm in arguments(symbol)) {
    if (mx.util.str.endswith(nm, "data")) {
      if (!is.null(data)) {
        stop("Multiple fields contains suffix data")
      } else {
        data <- nm
      }
    }
    if (mx.util.str.endswith(nm, "label")) {
      if (!is.null(label)) {
        stop("Multiple fields contains suffix label")
      } else {
        label <- nm
      }
    }
  }
  return(c(data, label))
}
