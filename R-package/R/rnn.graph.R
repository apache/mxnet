# 
#' Generate a RNN symbolic model - requires CUDA
#' 
#' @param config Either seq-to-one or one-to-one
#' @param cell.type Type of RNN cell: either gru or lstm
#' @param num.rnn.layer int, number of stacked layers
#' @param num.hidden int, size of the state in each RNN layer
#' @param num.embed  int, default = NULL - no embedding. Dimension of the embedding vectors
#' @param num.decode int, number of output variables in the decoding layer
#' @param input.size int, number of levels in the data - only used for embedding
#' @param dropout
#' 
#' @export
rnn.graph <- function(num.rnn.layer, 
                      input.size = NULL,
                      num.embed = NULL, 
                      num.hidden,
                      num.decode,
                      dropout = 0,
                      ignore_label = -1,
                      loss_output = NULL, 
                      config,
                      cell.type,
                      masking = F,
                      output_last_state = F) {
  
  # define input arguments
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  seq.mask <- mx.symbol.Variable("seq.mask")
  
  if (!is.null(num.embed)) embed.weight <- mx.symbol.Variable("embed.weight")
  
  rnn.params.weight <- mx.symbol.Variable("rnn.params.weight")
  rnn.state <- mx.symbol.Variable("rnn.state")
  
  if (cell.type == "lstm") {
    rnn.state.cell <- mx.symbol.Variable("rnn.state.cell")
  }
  
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  if (!is.null(num.embed)){
    data <- mx.symbol.Embedding(data=data, input_dim=input.size,
                                weight=embed.weight, output_dim=num.embed, name="embed")
  }
  
  # RNN cells
  if (cell.type == "lstm") {
    rnn <- mx.symbol.RNN(data=data, state=rnn.state, state_cell = rnn.state.cell, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=output_last_state, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", sep="_"))
    
  } else {
    rnn <- mx.symbol.RNN(data=data, state=rnn.state, parameters=rnn.params.weight, state.size=num.hidden, num.layers=num.rnn.layer, bidirectional=F, mode=cell.type, state.outputs=output_last_state, p=dropout, name=paste(cell.type, num.rnn.layer, "layer", sep="_"))
  }
  
  # Decode
  if (config=="seq-to-one") {
    
    if (masking) mask <- mx.symbol.SequenceLast(data=rnn[[1]], use.sequence.length = T, sequence_length = seq.mask, name = "mask") else
      mask <- mx.symbol.SequenceLast(data=rnn[[1]], use.sequence.length = F, name = "mask")
    
    decode <- mx.symbol.FullyConnected(data=mask,
                                       weight=cls.weight,
                                       bias=cls.bias,
                                       num.hidden=num.decode,
                                       name = "decode")
    
    if (!is.null(loss_output)) {
      loss <- switch(loss_output,
                     softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = "loss"),
                     linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = "loss"),
                     logictic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = "loss"),
                     MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = "loss")
      )
    } else loss <- decode
    
  } else if (config=="one-to-one"){
    
    if (masking) mask <- mx.symbol.SequenceMask(data = rnn[[1]], use.sequence.length = T, sequence_length = seq.mask, value = 0, name = "mask") else
      mask <- mx.symbol.identity(data = rnn[[1]], name = "mask")
    
    mask = mx.symbol.reshape(mask, shape=c(num.hidden, -1))
    
    decode <- mx.symbol.FullyConnected(data=reshape,
                                       weight=cls.weight,
                                       bias=cls.bias,
                                       num.hidden=num.decode,
                                       name = "decode")
    
    label <- mx.symbol.reshape(data=label, shape=c(-1), name = "label_reshape")
    
    if (!is.null(loss_output)) {
      loss <- switch(loss_output,
                     softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = "loss"),
                     linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = "loss"),
                     logictic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = "loss"),
                     MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = "loss")
      )
    } else loss <- decode
  }
  return(loss)
}


# LSTM cell symbol
lstm.cell <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0) {
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$i2h.weight, bias = param$i2h.bias, 
                                  num.hidden = num.hidden * 4, name = paste0("t", seqidx, ".l", layeridx, ".i2h"))
  
  if (dropout > 0) 
    i2h <- mx.symbol.Dropout(data = i2h, p = dropout)
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$h2h.weight, 
                                    bias = param$h2h.bias, num.hidden = num.hidden * 4, 
                                    name = paste0("t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 4, axis = 1, squeeze.axis = F, 
                                 name = paste0("t", seqidx, ".l", layeridx, ".slice"))
  
  in.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  in.transform <- mx.symbol.Activation(split.gates[[2]], act.type = "tanh")
  forget.gate <- mx.symbol.Activation(split.gates[[3]], act.type = "sigmoid")
  out.gate <- mx.symbol.Activation(split.gates[[4]], act.type = "sigmoid")
  
  if (is.null(prev.state)) {
    next.c <- in.gate * in.transform
  } else {
    next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
  }
  
  next.h <- out.gate * mx.symbol.Activation(next.c, act.type = "tanh")
  
  return(list(c = next.c, h = next.h))
}

# GRU cell symbol
gru.cell <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0) {
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$gates.i2h.weight, 
                                  bias = param$gates.i2h.bias, num.hidden = num.hidden * 2, 
                                  name = paste0("t", seqidx, ".l", layeridx, ".gates.i2h"))
  
  if (dropout > 0) 
    i2h <- mx.symbol.Dropout(data = i2h, p = dropout)
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$gates.h2h.weight, 
                                    bias = param$gates.h2h.bias, num.hidden = num.hidden * 2, 
                                    name = paste0("t", seqidx, ".l", layeridx, ".gates.h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 2, axis = 1, squeeze.axis = F, 
                                 name = paste0("t", seqidx, ".l", layeridx, ".split"))
  
  update.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  reset.gate <- mx.symbol.Activation(split.gates[[2]], act.type = "sigmoid")
  
  htrans.i2h <- mx.symbol.FullyConnected(data = indata, weight = param$trans.i2h.weight, 
                                         bias = param$trans.i2h.bias, num.hidden = num.hidden, 
                                         name = paste0("t", seqidx, ".l", layeridx, ".trans.i2h"))
  
  if (is.null(prev.state)) {
    h.after.reset <- reset.gate * 0
  } else {
    h.after.reset <- prev.state$h * reset.gate
  }
  
  htrans.h2h <- mx.symbol.FullyConnected(data = h.after.reset, weight = param$trans.h2h.weight, 
                                         bias = param$trans.h2h.bias, num.hidden = num.hidden, 
                                         name = paste0("t", seqidx, ".l", layeridx, ".trans.h2h"))
  
  h.trans <- htrans.i2h + htrans.h2h
  h.trans.active <- mx.symbol.Activation(h.trans, act.type = "tanh")
  
  if (is.null(prev.state)) {
    next.h <- update.gate * h.trans.active
  } else {
    next.h <- prev.state$h + update.gate * (h.trans.active - prev.state$h)
  }
  
  return(list(h = next.h))
}

# 
#' unroll representation of RNN running on non CUDA device - under development
#' 
#' @export
rnn.graph.unroll <- function(num.rnn.layer, 
                             seq.len, 
                             input.size = NULL,
                             num.embed = NULL, 
                             num.hidden,
                             num.decode,
                             dropout = 0,
                             ignore_label = -1,
                             loss_output = NULL, 
                             init.state = NULL,
                             config,
                             cell.type = "lstm", 
                             masking = F, 
                             output_last_state = F) {
  
  
  if (!is.null(num.embed)) embed.weight <- mx.symbol.Variable("embed.weight")
  
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  
  param.cells <- lapply(1:num.rnn.layer, function(i) {
    
    if (cell.type=="lstm"){
      cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                   i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                   h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                   h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
    } else if (cell.type=="gru"){
      cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")),
                   gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")),
                   gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")),
                   gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")),
                   trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")),
                   trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")),
                   trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")),
                   trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
    }
    return (cell)
  })
  
  # embeding layer
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  seq.mask <- mx.symbol.Variable("seq.mask")
  
  if (!is.null(num.embed)) {
    data <- mx.symbol.Embedding(data = data, input_dim = input.size,
                                weight=embed.weight, output_dim = num.embed, name = "embed")
  }
  
  data <- mx.symbol.split(data = data, axis = 0, num.outputs = seq.len, squeeze_axis = T)
  
  last.hidden <- list()
  last.states <- list()
  
  for (seqidx in 1:seq.len) {
    hidden <- data[[seqidx]]
    
    for (i in 1:num.rnn.layer) {
      
      if (seqidx==1) prev.state<- init.state[[i]] else prev.state <- last.states[[i]]
      
      if (cell.type=="lstm") {
        cell.symbol <- lstm.cell
      } else if (cell.type=="gru"){
        cell.symbol <- gru.cell
      }
      
      next.state <- cell.symbol(num.hidden = num.hidden, 
                                indata = hidden,
                                prev.state = prev.state,
                                param = param.cells[[i]],
                                seqidx = seqidx, 
                                layeridx = i,
                                dropout = dropout)
      hidden <- next.state$h
      last.states[[i]] <- next.state
    }
    
    # Aggregate outputs from each timestep
    last.hidden <- c(last.hidden, hidden)
  }
  
  # concat hidden units - concat seq.len blocks of dimension num.hidden x batch.size
  concat <- mx.symbol.concat(data = last.hidden, num.args = seq.len, dim = 0, name = "concat")
  concat <- mx.symbol.reshape(data = concat, shape = c(num.hidden, -1, seq.len), name = "rnn_reshape")
  
  if (config=="seq-to-one"){
    
    if (masking) mask <- mx.symbol.SequenceLast(data=concat, use.sequence.length = T, sequence_length = seq.mask, name = "mask") else
      mask <- mx.symbol.SequenceLast(data=concat, use.sequence.length = F, name = "mask")
    
    decode <- mx.symbol.FullyConnected(data = mask,
                                       weight = cls.weight,
                                       bias = cls.bias,
                                       num.hidden = num.decode,
                                       name = "decode")
    
    if (!is.null(loss_output)) {
      loss <- switch(loss_output,
                     softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = "loss"),
                     linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = "loss"),
                     logictic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = "loss"),
                     MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = "loss")
      )
    } else loss <- decode
    
  } else if (config=="one-to-one"){
    
    if (masking) mask <- mx.symbol.SequenceMask(data = concat, use.sequence.length = T, sequence_length = seq.mask, value = 0, name = "mask") else
      mask <- mx.symbol.identity(data = concat, name = "mask")
    
    mask = mx.symbol.reshape(mask, shape=c(num.hidden, -1))
    
    decode <- mx.symbol.FullyConnected(data = mask,
                                       weight = cls.weight,
                                       bias = cls.bias,
                                       num.hidden = num.decode,
                                       name = "decode")
    
    label <- mx.symbol.reshape(data = label, shape = -1, name = "label_reshape")
    
    if (!is.null(loss_output)) {
      loss <- switch(loss_output,
                     softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = "loss"),
                     linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = "loss"),
                     logictic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = "loss"),
                     MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = "loss")
      )
    } else loss <- decode
  }
  return(loss)
}
