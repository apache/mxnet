
#' Inference of RNN model
#'
#' @param infer.data Data iterator created by mx.io.bucket.iter
#' @param model Model used for inference
#' @param ctx
#'
#' @export
mx.infer.rnn <- function(infer.data, model, ctx = mx.cpu()) {
  
  ### Initialise the iterator
  infer.data$reset()
  infer.data$iter.next()
  
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) 
    stop("ctx must be mx.context or list of mx.context")
  
  ndevice <- length(ctx)
  symbol <- model$symbol
  if (is.list(symbol)) sym_ini <- symbol[[names(train.data$bucketID)]] else sym_ini <- symbol
  
  arguments <- sym_ini$arguments
  input.names <- intersect(names(infer.data$value()), arguments)
  
  input.shape <- sapply(input.names, function(n) {
    dim(infer.data$value()[[n]])
  }, simplify = FALSE)
  
  shapes <- sym_ini$infer.shape(input.shape)
  
  # initialize all arguments with zeros
  arguments.ini <- lapply(shapes$arg.shapes, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })

  arg.params <- model$arg.params
  arg.params.names <- names(arg.params)
  aux.params <- model$aux.params
  
  # Initial binding
  dlist <- arguments.ini[input.names]
  
  # Assign fixed parameters to their value and keep non initialized arguments to zero
  arg.params.fix.names <- setdiff(arguments, c(arg.params.names, input.names))
  
  # Assign zeros to non initialized arg parameters
  arg.params.fix <- arguments.ini[arg.params.fix.names]
  
  # Grad request
  grad.req <- rep("null", length(arguments))
  
  # Arg array order
  update_names <- c(input.names, arg.params.fix.names, arg.params.names)
  arg_update_idx <- match(arguments, update_names)
  
  execs <- mx.symbol.bind(symbol = symbol, arg.arrays = c(dlist, arg.params.fix, arg.params)[arg_update_idx], 
                                  aux.arrays = aux.params, ctx = ctx[[1]], grad.req = grad.req)
  
  # Initial input shapes - need to be adapted for multi-devices - divide highest
  # dimension by device nb
  
  packer <- mx.nd.arraypacker()
  infer.data$reset()
  while (infer.data$iter.next()) {
    
    # Get input data slice
    dlist <- infer.data$value()  #[input.names]
    
    execs <- mx.symbol.bind(symbol = symbol, arg.arrays = c(dlist, execs$arg.arrays[arg.params.fix.names], execs$arg.arrays[arg.params.names])[arg_update_idx], 
                                    aux.arrays = execs$aux.arrays, ctx = ctx[[1]], grad.req = grad.req)
    
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


#' Inference for one-to-one fusedRNN (CUDA) models
#'
#' @param infer.data Data iterator created by mx.io.bucket.iter
#' @param symbol Symbol used for inference
#' @param arg.params
#' @param aux.params
#' @param input.params
#' @param ctx
#'
#' @export
mx.infer.rnn.one <- function(infer.data, 
                             symbol, 
                             arg.params, 
                             aux.params, 
                             input.params = NULL, 
                             ctx = mx.cpu()) {
  
  ### Initialise the iterator
  infer.data$reset()
  infer.data$iter.next()
  
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) 
    stop("ctx must be mx.context or list of mx.context")
  
  ndevice <- length(ctx)
  
  arguments <- symbol$arguments
  input.names <- intersect(names(infer.data$value()), arguments)
  
  input.shape <- sapply(input.names, function(n) {
    dim(infer.data$value()[[n]])
  }, simplify = FALSE)
  
  shapes <- symbol$infer.shape(input.shape)
  
  # initialize all arguments with zeros
  arguments.ini <- lapply(shapes$arg.shapes, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  arg.params <- arg.params
  arg.params.names <- names(arg.params)
  
  dlist <- arguments.ini[input.names]
  
  # Assign fixed parameters to their value and keep non initialized arguments to zero
  arg.params.fix.names <- unique(c(names(input.params), setdiff(arguments, c(arg.params.names, input.names))))
  
  # Assign zeros to non initialized arg parameters
  arg.params.fix <- arguments.ini[arg.params.fix.names]
  # Assign weights to arguments specifies by input.params
  arg.params.fix[names(input.params)] <- input.params
  
  aux.params <- aux.params
  
  # Grad request
  grad.req <- rep("null", length(arguments))
  
  # Arg array order
  update_names <- c(input.names, arg.params.fix.names, arg.params.names)
  arg_update_idx <- match(arguments, update_names)
  
  # Initial binding
  execs <- mx.symbol.bind(symbol = symbol, 
                          arg.arrays = c(dlist, arg.params.fix, arg.params)[arg_update_idx], 
                          aux.arrays = aux.params, ctx = ctx[[1]], grad.req = grad.req)
  
  # Initial input shapes - need to be adapted for multi-devices - divide highest
  # dimension by device nb
  
  infer.data$reset()
  while (infer.data$iter.next()) {
    
    # Get input data slice
    dlist <- infer.data$value()[input.names]
    
    execs <- mx.symbol.bind(symbol = symbol, 
                            arg.arrays = c(dlist, execs$arg.arrays[arg.params.fix.names], execs$arg.arrays[arg.params.names])[arg_update_idx],
                            aux.arrays = execs$aux.arrays, ctx = ctx[[1]], grad.req = grad.req)
    
    mx.exec.forward(execs, is.train = FALSE)
    
    out.pred <- mx.nd.copyto(execs$ref.outputs[[1]], mx.cpu())
    state <- mx.nd.copyto(execs$ref.outputs[[2]], mx.cpu())
    state_cell <- mx.nd.copyto(execs$ref.outputs[[3]], mx.cpu())
    
    out <- lapply(execs$ref.outputs, function(out) {
      mx.nd.copyto(out, mx.cpu())
    })
  }
  infer.data$reset()
  return(out)
}


#' Inference for one-to-one unroll models
#'
#' @param infer.data NDArray
#' @param symbol Model used for inference
#' @param num_hidden 
#' @param arg.params
#' @param aux.params
#' @param init_states
#' @param ctx
#'
#' @export
mx.infer.rnn.one.unroll <- function(infer.data, 
                                    symbol,
                                    num_hidden, 
                                    arg.params, 
                                    aux.params, 
                                    init_states = NULL, 
                                    ctx = mx.cpu()) {
  
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  
  if (!is.list(ctx)) 
    stop("ctx must be mx.context or list of mx.context")
  
  ndevice <- length(ctx)
  
  arguments <- symbol$arguments
  input.names <- intersect(c("data", "label"), arguments)
  
  input.shape <- list("data" = dim(infer.data), "label" = dim(infer.data))
  
  # init_state_shapes
  init_states_names <- arguments[startsWith(arguments, "init_")]
  init_states_shapes = lapply(init_states_names, function(x) c(num_hidden, tail(input.shape[[1]], 1)))
  names(init_states_shapes) <- init_states_names
  
  shapes <- symbol$infer.shape(c(input.shape, init_states_shapes))
  
  # initialize all arguments with zeros
  arguments.ini <- lapply(shapes$arg.shapes, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  dlist <- list("data" = infer.data, "label" = infer.data)
  
  if (is.null(init_states)) {
    init_states <- arguments.ini[init_states_names]
  } else {
    names(init_states) <- init_states_names
  }
  
  # remove potential duplicates arguments - if inference on CUDA RNN symbol
  arg.params <- arg.params[setdiff(names(arg.params), c(input.names, init_states_names))]
  arg.params.names <- names(arg.params)
  
  # Aux params
  aux.params <- aux.params
  
  # Grad request
  grad.req <- rep("null", length(arguments))
  
  # Arg array order
  update_names <- c(input.names, init_states_names, arg.params.names)
  arg_update_idx <- match(arguments, update_names)
  
  # Bind to exec
  execs <- mxnet:::mx.symbol.bind(symbol = symbol,
                                  arg.arrays = c(dlist, init_states, arg.params)[arg_update_idx],
                                  aux.arrays = aux.params, ctx = ctx[[1]], grad.req = grad.req)
  
  mx.exec.forward(execs, is.train = FALSE)
  
  out <- lapply(execs$ref.outputs, function(out) mx.nd.copyto(out, mx.cpu()))
  
  return(out)
}
