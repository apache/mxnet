library(mxnet)

source("rnn.R")

mx.rnn.infer.buckets <- function(infer_iter, model, ctx = mx.cpu(), 
                                 verbose=T) {
  
  ### Initialise the iterator
  infer_iter$reset()
  infer_iter$iter.next()
  batch_size <- infer_iter$batch.size
  
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) 
    stop("ctx must be mx.context or list of mx.context")
  
  ndevice <- length(ctx)
  symbol <- model$symbol
  
  input.names <- c("data", "seq.mask")
  input.shape <- sapply(input.names, function(n) {
    dim(infer_iter$value()[[n]])
  }, simplify = FALSE)
  
  output.names <- "label"
  output.shape <- sapply(output.names, function(n) {
    dim(infer_iter$value()[[n]])
  }, simplify = FALSE)
  
  arg.params <- model$arg.params
  arg.names <- names(arg.params)
  aux.params <- model$aux.params
  
  # Grad request
  grad_req <- rep("null", length(symbol$arguments))
  
  # Arg array order
  update_names <- c(input.names, output.names, arg.names)
  arg_update_idx <- match(symbol$arguments, update_names)
  
  # Initial binding
  dlist <- lapply(c(input.shape, output.shape), function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu()) 
  })
  
  execs <- mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(dlist, arg.params)[arg_update_idx], 
                                  aux.arrays = aux.params, ctx = ctx[[1]], grad.req = grad_req)
  
  # Initial input shapes - need to be adapted for multi-devices - divide highest
  # dimension by device nb
  
  packer <- mxnet:::mx.nd.arraypacker()
  infer_iter$reset()
  while (infer_iter$iter.next()) {
    
    # Get input data slice
    dlist <- infer_iter$value()  #[input.names]
    
    execs <- mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(dlist, execs$arg.arrays[arg.names])[arg_update_idx], 
                                    aux.arrays = execs$aux.arrays, ctx = ctx[[1]], grad.req = grad_req)
    
    mx.exec.forward(execs, is.train = FALSE)
    
    out.pred <- mx.nd.copyto(execs$ref.outputs[[1]], mx.cpu())
    padded <- infer_iter$num.pad()
    oshape <- dim(out.pred)
    ndim <- length(oshape)
    packer$push(mx.nd.slice.axis(data = out.pred, axis = 0, begin = 0, end = oshape[[ndim]] - padded))
    
  }
  infer_iter$reset()
  return(packer$get())
}
