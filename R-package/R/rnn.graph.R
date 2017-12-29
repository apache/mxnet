
#' Generate a RNN symbolic model - requires CUDA
#' 
#' @param config Either seq-to-one or one-to-one
#' @param cell_type Type of RNN cell: either gru or lstm
#' @param num_rnn_layer int, number of stacked layers
#' @param num_hidden int, size of the state in each RNN layer
#' @param num_embed  int, default = NULL - no embedding. Dimension of the embedding vectors
#' @param num_decode int, number of output variables in the decoding layer
#' @param input_size int, number of levels in the data - only used for embedding
#' @param dropout
#' 
#' @export
rnn.graph <- function (num_rnn_layer, input_size = NULL, num_embed = NULL, 
                       num_hidden, num_decode, dropout = 0, ignore_label = -1, bidirectional = F, 
                       loss_output = NULL, config, cell_type, masking = F, output_last_state = F,
                       rnn.state = NULL, rnn.state.cell = NULL, prefix = "") {
  
  data <- mx.symbol.Variable("data")
  label <- mx.symbol.Variable("label")
  seq.mask <- mx.symbol.Variable("seq.mask")
  if (!is.null(num_embed)) 
    embed.weight <- mx.symbol.Variable("embed.weight")
  rnn.params.weight <- mx.symbol.Variable("rnn.params.weight")
  
  if (is.null(rnn.state)) rnn.state <- mx.symbol.Variable("rnn.state")
  if (cell_type == "lstm" & is.null(rnn.state.cell)) {
    rnn.state.cell <- mx.symbol.Variable("rnn.state.cell")
  }
  
  cls.weight <- mx.symbol.Variable("cls.weight")
  cls.bias <- mx.symbol.Variable("cls.bias")
  if (!is.null(num_embed)) {
    data <- mx.symbol.Embedding(data = data, input_dim = input_size, 
                                weight = embed.weight, output_dim = num_embed, name = "embed")
  }
  
  data = mx.symbol.swapaxes(data = data, dim1 = 0, dim2 = 1, name = paste0(prefix, "swap_pre"))
  
  if (cell_type == "lstm") {
    rnn <- mx.symbol.RNN(data = data, state = rnn.state, 
                         state_cell = rnn.state.cell, parameters = rnn.params.weight, 
                         state.size = num_hidden, num.layers = num_rnn_layer, 
                         bidirectional = bidirectional, mode = cell_type, state.outputs = output_last_state, 
                         p = dropout, name = paste0(prefix, "RNN"))
  } else {
    rnn <- mx.symbol.RNN(data = data, state = rnn.state, 
                         parameters = rnn.params.weight, state.size = num_hidden, 
                         num.layers = num_rnn_layer, bidirectional = bidirectional, mode = cell_type, 
                         state.outputs = output_last_state, p = dropout, 
                         name = paste0(prefix, "RNN"))
  }
  
  if (config == "seq-to-one") {
    if (masking) mask <- mx.symbol.SequenceLast(data = rnn[[1]], use.sequence.length = T, sequence_length = seq.mask, name = "mask") else
      mask <- mx.symbol.SequenceLast(data = rnn[[1]], use.sequence.length = F, name = "mask")
    
    if (!is.null(loss_output)) {
      decode <- mx.symbol.FullyConnected(data = mask, weight = cls.weight, bias = cls.bias, num_hidden = num_decode, name = "decode")
      out <- switch(loss_output, softmax = mx.symbol.SoftmaxOutput(data = decode, label = label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = "loss"), 
                    linear = mx.symbol.LinearRegressionOutput(data = decode, label = label, name = "loss"), 
                    logistic = mx.symbol.LogisticRegressionOutput(data = decode, label = label, name = "loss"), 
                    MAE = mx.symbol.MAERegressionOutput(data = decode, label = label, name = "loss"))
    }
    else out <- mask
  }
  
  else if (config == "one-to-one") {
    
    if (masking) mask <- mx.symbol.SequenceMask(data = rnn[[1]], use.sequence.length = T, sequence_length = seq.mask, value = 0, name = "mask") else
      mask <- mx.symbol.identity(data = rnn[[1]], name = "mask")
    
    mask = mx.symbol.swapaxes(data = mask, dim1 = 0, dim2 = 1, name = paste0(prefix, "swap_post"))
    
    if (!is.null(loss_output)) {
      
      mask <- mx.symbol.reshape(data = mask, shape = c(0, -1), reverse = TRUE)
      label <- mx.symbol.reshape(data = label, shape = c(-1))
      
      decode <- mx.symbol.FullyConnected(data = mask, weight = cls.weight, bias = cls.bias, num_hidden = num_decode, 
                                         flatten = TRUE, name = paste0(prefix, "decode"))
      
      out <- switch(loss_output, softmax = mx.symbol.SoftmaxOutput(data = decode, label = label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = "loss"), 
                    linear = mx.symbol.LinearRegressionOutput(data = decode, label = label, name = "loss"), 
                    logistic = mx.symbol.LogisticRegressionOutput(data = decode, label = label, name = "loss"), 
                    MAE = mx.symbol.MAERegressionOutput(data = decode, label = label, name = "loss"))
    } else out <- mask
  }
  return(out)
}

# LSTM cell symbol
lstm.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0, prefix = "") {
  
  if (dropout > 0 && layeridx > 1) 
    indata <- mx.symbol.Dropout(data = indata, p = dropout)
  
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$i2h.weight, bias = param$i2h.bias, 
                                  num_hidden = num_hidden * 4, name = paste0(prefix, "t", seqidx, ".l", layeridx, ".i2h"))
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$h2h.weight, 
                                    bias = param$h2h.bias, num_hidden = num_hidden * 4, 
                                    name = paste0(prefix, "t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 4, axis = 1, squeeze.axis = F, 
                                 name = paste0(prefix, "t", seqidx, ".l", layeridx, ".slice"))
  
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
  
  return(list(h = next.h, c = next.c))
}


# GRU cell symbol
gru.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0, prefix)
{
  if (dropout > 0 && layeridx > 1) 
    indata <- mx.symbol.Dropout(data = indata, p = dropout)
  
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$gates.i2h.weight, 
                                  bias = param$gates.i2h.bias, num_hidden = num_hidden * 2, 
                                  name = paste0(prefix, "t", seqidx, ".l", layeridx, ".gates.i2h"))
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$gates.h2h.weight, 
                                    bias = param$gates.h2h.bias, num_hidden = num_hidden * 2, 
                                    name = paste0(prefix, "t", seqidx, ".l", layeridx, ".gates.h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }

  split.gates <- mx.symbol.split(gates, num.outputs = 2, axis = 1, squeeze.axis = F, 
                                 name = paste0(prefix, "t", seqidx, ".l", layeridx, ".split"))
  
  update.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  reset.gate <- mx.symbol.Activation(split.gates[[2]], act.type = "sigmoid")
  
  htrans.i2h <- mx.symbol.FullyConnected(data = indata, weight = param$trans.i2h.weight, 
                                         bias = param$trans.i2h.bias, num_hidden = num_hidden, 
                                         name = paste0(prefix, "t", seqidx, ".l", layeridx, ".trans.i2h"))
  
  if (is.null(prev.state)) {
    h.after.reset <- reset.gate * 0
  } else {
    h.after.reset <- prev.state$h * reset.gate
  }
  
  htrans.h2h <- mx.symbol.FullyConnected(data = h.after.reset, weight = param$trans.h2h.weight, 
                                         bias = param$trans.h2h.bias, num_hidden = num_hidden, 
                                         name = paste0(prefix, "t", seqidx, ".l", layeridx, ".trans.h2h"))
  
  h.trans <- htrans.i2h + htrans.h2h
  h.trans.active <- mx.symbol.Activation(h.trans, act.type = "tanh")
  
  if (is.null(prev.state)) {
    next.h <- update.gate * h.trans.active
  } else {
    next.h <- prev.state$h + update.gate * (h.trans.active - prev.state$h)
  }
  
  return(list(h = next.h))
}


#' unroll representation of RNN running on non CUDA device
#' 
#' @param config Either seq-to-one or one-to-one
#' @param cell_type Type of RNN cell: either gru or lstm
#' @param num_rnn_layer int, number of stacked layers
#' @param seq_len int, number of time steps to unroll
#' @param num_hidden int, size of the state in each RNN layer
#' @param num_embed  int, default = NULL - no embedding. Dimension of the embedding vectors
#' @param num_decode int, number of output variables in the decoding layer
#' @param input_size int, number of levels in the data - only used for embedding
#' @param dropout 
#' 
#' @export
rnn.graph.unroll <- function(num_rnn_layer, 
                             seq_len, 
                             input_size = NULL,
                             num_embed = NULL, 
                             num_hidden,
                             num_decode,
                             dropout = 0,
                             ignore_label = -1,
                             loss_output = NULL, 
                             init.state = NULL,
                             config,
                             cell_type = "lstm", 
                             masking = F, 
                             output_last_state = F,
                             prefix = "",
                             data_name = "data",
                             label_name = "label") {
  
  if (!is.null(num_embed)) embed.weight <- mx.symbol.Variable(paste0(prefix, "embed.weight"))
  
  # Initial state
  if (is.null(init.state) & output_last_state) {
    init.state <- lapply(1:num_rnn_layer, function(i) {
      if (cell_type=="lstm") {
        state <- list(h = mx.symbol.Variable(paste0("init_", prefix, i, "_h")),
                      c = mx.symbol.Variable(paste0("init_", prefix, i, "_c")))
      } else if (cell_type=="gru") {
        state <- list(h = mx.symbol.Variable(paste0("init_", prefix, i, "_h")))
      }
      return (state)
    })
  }
  
  cls.weight <- mx.symbol.Variable(paste0(prefix, "cls.weight"))
  cls.bias <- mx.symbol.Variable(paste0(prefix, "cls.bias"))
  
  param.cells <- lapply(1:num_rnn_layer, function(i) {
    
    if (cell_type=="lstm") {
      cell <- list(i2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".i2h.weight")),
                   i2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".i2h.bias")),
                   h2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".h2h.weight")),
                   h2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".h2h.bias")))
    } else if (cell_type=="gru") {
      cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".gates.i2h.weight")),
                   gates.i2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".gates.i2h.bias")),
                   gates.h2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".gates.h2h.weight")),
                   gates.h2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".gates.h2h.bias")),
                   trans.i2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".trans.i2h.weight")),
                   trans.i2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".trans.i2h.bias")),
                   trans.h2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".trans.h2h.weight")),
                   trans.h2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".trans.h2h.bias")))
    }
    return (cell)
  })
  
  # embeding layer
  data <- mx.symbol.Variable(data_name)
  label <- mx.symbol.Variable(label_name)
  seq.mask <- mx.symbol.Variable(paste0(prefix, "seq.mask"))
  
  data = mx.symbol.swapaxes(data = data, dim1 = 0, dim2 = 1, name = paste0(prefix, "swap_pre"))
  
  if (!is.null(num_embed)) {
    data <- mx.symbol.Embedding(data = data, input_dim = input_size,
                                weight=embed.weight, output_dim = num_embed, name = paste0(prefix, "embed"))
  }
  
  data <- mx.symbol.split(data = data, axis = 0, num.outputs = seq_len, squeeze_axis = T)
  
  last.hidden <- list()
  last.states <- list()
  
  for (seqidx in 1:seq_len) {
    hidden <- data[[seqidx]]
    
    for (i in 1:num_rnn_layer) {
      
      if (seqidx==1) prev.state <- init.state[[i]] else 
        prev.state <- last.states[[i]]
      
      if (cell_type=="lstm") {
        cell.symbol <- lstm.cell
      } else if (cell_type=="gru"){
        cell.symbol <- gru.cell
      }
      
      next.state <- cell.symbol(num_hidden = num_hidden, 
                                indata = hidden,
                                prev.state = prev.state,
                                param = param.cells[[i]],
                                seqidx = seqidx, 
                                layeridx = i,
                                dropout = dropout,
                                prefix = prefix)
      
      hidden <- next.state$h
      last.states[[i]] <- next.state
    }
    
    # Aggregate outputs from each timestep
    last.hidden <- c(last.hidden, hidden)
  }
  
  if (output_last_state) {
    out.states = mx.symbol.Group(unlist(last.states))
  }
  
  # concat hidden units - concat seq_len blocks of dimension num_hidden x batch.size
  concat <- mx.symbol.concat(data = last.hidden, num.args = seq_len, dim = 0, name = paste0(prefix, "concat"))
  concat <- mx.symbol.reshape(data = concat, shape = c(num_hidden, -1, seq_len), name = paste0(prefix, "rnn_reshape"))
  
  if (config=="seq-to-one") {
    
    if (masking) mask <- mx.symbol.SequenceLast(data=concat, use.sequence.length = T, sequence_length = seq.mask, name = paste0(prefix, "mask")) else
      mask <- mx.symbol.SequenceLast(data=concat, use.sequence.length = F, name = paste0(prefix, "mask"))
    
    if (!is.null(loss_output)) {
      
      decode <- mx.symbol.FullyConnected(data = mask,
                                         weight = cls.weight,
                                         bias = cls.bias,
                                         num_hidden = num_decode,
                                         name = paste0(prefix, "decode"))
      
      out <- switch(loss_output,
                    softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = paste0(prefix, "loss")),
                    linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    logistic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, paste0(prefix, name = "loss")),
                    MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, paste0(prefix, name = "loss"))
      )
    } else out <- mask
    
  } else if (config=="one-to-one"){
    
    if (masking) mask <- mx.symbol.SequenceMask(data = concat, use.sequence.length = T, sequence_length = seq.mask, value = 0, name = paste0(prefix, "mask")) else
      mask <- mx.symbol.identity(data = concat, name = paste0(prefix, "mask"))
    
    mask = mx.symbol.swapaxes(data = mask, dim1 = 0, dim2 = 1, name = paste0(prefix, "swap_post"))
    
    if (!is.null(loss_output)) {
      
      mask <- mx.symbol.reshape(data = mask, shape = c(0, -1), reverse = TRUE)
      label <- mx.symbol.reshape(data = label, shape = c(-1))
      
      decode <- mx.symbol.FullyConnected(data = mask, weight = cls.weight, bias = cls.bias, num_hidden = num_decode, 
                                         flatten = T, name = paste0(prefix, "decode"))
      
      out <- switch(loss_output,
                    softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, 
                                                      name = paste0(prefix, "loss")),
                    linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    logistic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = paste0(prefix, "loss"))
      )
    } else out <- mask
  }
  
  if (output_last_state) {
    return(mx.symbol.Group(c(out, out.states)))
  } else return(out)
}
