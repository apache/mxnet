# 
#' Inference of RNN model
#'
#' @param infer.data Data iterator created by mx.io.bucket.iter
#' @param model Model used for inference
#' @param ctx The element to mask
#'
#' @export
mx.rnn.infer.buckets <- function(infer.data, model, ctx = mx.cpu()) {
  
  ### Initialise the iterator
  infer.data$reset()
  infer.data$iter.next()
  batch_size <- infer.data$batch.size
  
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
    dim(infer.data$value()[[n]])
  }, simplify = FALSE)
  
  output.names <- "label"
  output.shape <- sapply(output.names, function(n) {
    dim(infer.data$value()[[n]])
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
  infer.data$reset()
  while (infer.data$iter.next()) {
    
    # Get input data slice
    dlist <- infer.data$value()  #[input.names]
    
    execs <- mxnet:::mx.symbol.bind(symbol = symbol, arg.arrays = c(dlist, execs$arg.arrays[arg.names])[arg_update_idx], 
                                    aux.arrays = execs$aux.arrays, ctx = ctx[[1]], grad.req = grad_req)
    
    mx.exec.forward(execs, is.train = FALSE)
    
    out.pred <- mx.nd.copyto(execs$ref.outputs[[1]], mx.cpu())
    padded <- infer.data$num.pad()
    oshape <- dim(out.pred)
    ndim <- length(oshape)
    packer$push(mx.nd.slice.axis(data = out.pred, axis = 0, begin = 0, end = oshape[[ndim]] - padded))
    
  }
  infer.data$reset()
  return(packer$get())
}
