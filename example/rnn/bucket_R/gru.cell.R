# GRU cell symbol
gru.cell <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0, 
  data_masking) {
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$gates.i2h.weight, 
    bias = param$gates.i2h.bias, num.hidden = num.hidden * 2, name = paste0("t", 
      seqidx, ".l", layeridx, ".gates.i2h"))
  
  if (dropout > 0) 
    i2h <- mx.symbol.Dropout(data = i2h, p = dropout)
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$gates.h2h.weight, 
      bias = param$gates.h2h.bias, num.hidden = num.hidden * 2, name = paste0("t", 
        seqidx, ".l", layeridx, ".gates.h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 2, axis = 1, squeeze.axis = F, 
    name = paste0("t", seqidx, ".l", layeridx, ".split"))
  
  update.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  reset.gate <- mx.symbol.Activation(split.gates[[2]], act.type = "sigmoid")
  
  htrans.i2h <- mx.symbol.FullyConnected(data = indata, weight = param$trans.i2h.weight, 
    bias = param$trans.i2h.bias, num.hidden = num.hidden, name = paste0("t", 
      seqidx, ".l", layeridx, ".trans.i2h"))
  
  if (is.null(prev.state)) {
    h.after.reset <- reset.gate * 0
  } else {
    h.after.reset <- prev.state$h * reset.gate
  }
  
  htrans.h2h <- mx.symbol.FullyConnected(data = h.after.reset, weight = param$trans.h2h.weight, 
    bias = param$trans.h2h.bias, num.hidden = num.hidden, name = paste0("t", 
      seqidx, ".l", layeridx, ".trans.h2h"))
  
  h.trans <- htrans.i2h + htrans.h2h
  h.trans.active <- mx.symbol.Activation(h.trans, act.type = "tanh")
  
  if (is.null(prev.state)) {
    next.h <- update.gate * h.trans.active
  } else {
    next.h <- prev.state$h + update.gate * (h.trans.active - prev.state$h)
  }
  
  ### Add a mask - using the mask_array approach
  data_mask_expand <- mx.symbol.Reshape(data = data_masking, shape = c(1, -2))
  next.h <- mx.symbol.broadcast_mul(lhs = next.h, rhs = data_mask_expand)
  
  return(list(h = next.h))
}
