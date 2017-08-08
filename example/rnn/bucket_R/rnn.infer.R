library(mxnet)

source("rnn.R")

mx.rnn.infer.buckets <- function(infer_iter, model, config, ctx = mx.cpu(), output_last_state = FALSE, 
  init.state = NULL, cell.type = "lstm") {
  ### Infer parameters from model
  if (cell.type == "lstm") {
    num.rnn.layer <- round((length(model$arg.params) - 3)/4)
    num.hidden <- dim(model$arg.params$l1.h2h.weight)[1]
  } else if (cell.type == "gru") {
    num.rnn.layer <- round((length(model$arg.params) - 3)/8)
    num.hidden <- dim(model$arg.params$l1.gates.h2h.weight)[1]
  }
  
  input.size <- dim(model$arg.params$embed.weight)[2]
  num.embed <- dim(model$arg.params$embed.weight)[1]
  num.label <- dim(model$arg.params$cls.bias)
  
  ### Initialise the iterator
  infer_iter$reset()
  infer_iter$iter.next()
  batch_size <- infer_iter$batch.size
  
  # get unrolled lstm symbol
  sym_list <- sapply(infer_iter$bucket.names, function(x) {
    rnn.unroll(num.rnn.layer = num.rnn.layer, num.hidden = num.hidden, seq.len = as.integer(x), 
      input.size = input.size, num.embed = num.embed, num.label = num.label, 
      config = config, dropout = 0, init.state = init.state, cell.type = cell.type, 
      output_last_state = output_last_state)
  }, simplify = F, USE.NAMES = T)
  
  symbol <- sym_list[[names(infer_iter$bucketID)]]
  
  input.shape <- lapply(infer_iter$value(), dim)
  input.shape <- input.shape[names(input.shape) %in% arguments(symbol)]
  
  infer_shapes <- symbol$infer.shape(input.shape)
  arg.params <- model$arg.params
  aux.params <- model$aux.params
  
  input.names <- names(input.shape)
  arg.names <- names(arg.params)
  
  # Grad request
  grad_req <- rep("null", length(symbol$arguments))
  
  # Arg array order
  update_names <- c(input.names, arg.names)
  arg_update_idx <- match(symbol$arguments, update_names)
  
  # Initial input shapes - need to be adapted for multi-devices - divide highest
  # dimension by device nb
  s <- sapply(input.shape, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  train.execs <- mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(s, arg.params)[arg_update_idx], 
    aux.arrays = aux.params, ctx = ctx, grad.req = grad_req)
  
  packer <- mxnet:::mx.nd.arraypacker()
  infer_iter$reset()
  while (infer_iter$iter.next()) {
    # Get input data slice
    dlist <- infer_iter$value()[input.names]
    
    symbol <- sym_list[[names(infer_iter$bucketID)]]
    
    texec <- mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(dlist, train.execs$arg.arrays[arg.names])[arg_update_idx], 
      aux.arrays = train.execs$aux.arrays, ctx = ctx, grad.req = grad_req)
    
    mx.exec.forward(texec, is.train = FALSE)
    
    out.preds <- mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
    packer$push(out.preds)
  }
  infer_iter$reset()
  return(packer$get())
}
