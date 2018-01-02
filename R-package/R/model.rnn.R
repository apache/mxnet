# Internal function to do multiple device training on RNN
mx.model.train.buckets <- function(symbol, ctx, train.data, eval.data, 
                                   dlist, arg.params, aux.params, 
                                   grad.req, arg.update.idx, 
                                   begin.round, end.round, optimizer, metric, 
                                   epoch.end.callback, batch.end.callback, kvstore, verbose = TRUE) {
  
  ndevice <- length(ctx)
  if (verbose) 
    message("Start training with ", ndevice, " devices")
  
  input.names <- names(dlist)
  arg.params.names <- names(arg.params)
  
  if (is.list(symbol)) sym_ini <- symbol[[names(train.data$bucketID)]] else sym_ini <- symbol
  
  slices <- lapply(seq_len(ndevice), function(i) {
    sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = FALSE))
  })
  
  train.execs <- lapply(seq_len(ndevice), function(i) {
    s <- slices[[i]]
    mx.symbol.bind(symbol = sym_ini, arg.arrays = c(s, arg.params)[arg.update.idx], 
                           aux.arrays = aux.params, ctx = ctx[[i]], grad.req = grad.req)
  })
  
  # KVStore related stuffs
  params.index <- as.integer(
    mx.util.filter.null(
      lapply(seq_along(train.execs[[1]]$ref.grad.arrays), function(k) {
        if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL}
      )))
  
  update.on.kvstore <- FALSE
  if (!is.null(kvstore) && kvstore$update.on.kvstore) {
    update.on.kvstore <- TRUE
    kvstore$set.optimizer(optimizer)
  } else {
    updaters <- lapply(seq_len(ndevice), function(i) {
      mx.opt.get.updater(optimizer, train.execs[[i]]$ref.arg.arrays)
    })
  }
  
  if (!is.null(kvstore)) {
    kvstore$init(params.index, train.execs[[1]]$ref.arg.arrays[params.index])
  }
  
  # train over specified number of epochs
  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    train.data$reset()
    while (train.data$iter.next()) {
      
      # Get iterator data
      dlist <- train.data$value()[input.names]
      
      # Slice inputs for multi-devices
      slices <- lapply(seq_len(ndevice), function(i) {
        sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = F))
      })
      
      # Assign input to each executor - bug on inference if using BatchNorm
      if (is.list(symbol)) {
        train.execs <- lapply(seq_len(ndevice), function(i) {
          s <- slices[[i]]
          mx.symbol.bind(symbol = symbol[[names(train.data$bucketID)]], 
                         arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.params.names])[arg.update.idx],
                         aux.arrays = train.execs[[i]]$aux.arrays, ctx = ctx[[i]], grad.req = grad.req)
        })
      } else {
        for (i in seq_len(ndevice)) {
          s <- slices[[i]]
          mx.exec.update.arg.arrays(train.execs[[i]], s, match.name=TRUE)
        }
      }
      
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
        arg.blocks <- lapply(seq_len(ndevice), function(i) {
          updaters[[i]](train.execs[[i]]$ref.arg.arrays, train.execs[[i]]$ref.grad.arrays)
        })
        for (i in seq_len(ndevice)) {
          mx.exec.update.arg.arrays(train.execs[[i]], arg.blocks[[i]], skip.null = TRUE)
        }
      }
      
      # Update the evaluation metrics
      if (!is.null(metric)) {
        for (i in seq_len(ndevice)) {
          train.metric <- metric$update(label = slices[[i]][[length(slices[[i]])]], 
                                        pred = out.preds[[i]], state = train.metric)
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
        message("[", iteration, "] Train-", result$name, "=", result$value)
    }
    
    if (!is.null(eval.data)) {
      if (!is.null(metric)) {
        eval.metric <- metric$init()
      }
      eval.data$reset()
      while (eval.data$iter.next()) {
        
        # Get iterator data
        dlist <- eval.data$value()[input.names]
        
        # Slice input to multiple devices
        slices <- lapply(seq_len(ndevice), function(i) {
          sapply(names(dlist), function(n) mx.nd.split(data=dlist[[n]], num_outputs = ndevice, axis = 0, squeeze_axis = FALSE))
        })
        
        # Assign input to each executor - bug on inference if using BatchNorm
        if (is.list(symbol)) {
          train.execs <- lapply(seq_len(ndevice), function(i) {
            s <- slices[[i]]
            mx.symbol.bind(symbol = symbol[[names(eval.data$bucketID)]], 
                                   arg.arrays = c(s, train.execs[[i]]$arg.arrays[arg.params.names])[arg.update.idx],
                                   aux.arrays = train.execs[[i]]$aux.arrays, ctx = ctx[[i]], grad.req = grad.req)
          })
        } else {
          for (i in seq_len(ndevice)) {
            s <- slices[[i]]
            mx.exec.update.arg.arrays(train.execs[[i]], s, match.name=TRUE)
          }
        }
        
        for (texec in train.execs) {
          mx.exec.forward(texec, is.train = FALSE)
        }
        
        # copy outputs to CPU
        out.preds <- lapply(train.execs, function(texec) {
          mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
        })
        
        if (!is.null(metric)) {
          for (i in seq_len(ndevice)) {
            eval.metric <- metric$update(slices[[i]][[length(slices[[i]])]], 
                                         out.preds[[i]], eval.metric)
          }
        }
      }
      
      if (!is.null(metric)) {
        result <- metric$get(eval.metric)
        if (verbose) {
          message("[", iteration, "] Validation-", result$name, "=", 
                         result$value)
        }
      }
    } else {
      eval.metric <- NULL
    }
    # get the model out
    model <- mx.model.extract.model(sym_ini, train.execs)
    
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


# 
#' Train RNN with bucket support
#'
#' @param symbol Symbol or list of Symbols representing the model
#' @param train.data Training data created by mx.io.bucket.iter
#' @param eval.data Evaluation data created by mx.io.bucket.iter
#' @param num.round int, number of epoch
#' @param initializer
#' @param optimizer
#' @param batch.end.callback
#' @param epoch.end.callback
#' @param begin.round
#' @param metric
#' @param ctx
#' @param kvstore
#' @param verbose
#'
#' @export
mx.model.buckets <- function(symbol, train.data, eval.data = NULL, metric = NULL, 
                             arg.params = NULL, aux.params = NULL, fixed.params = NULL, 
                             num.round = 1, begin.round = 1, 
                             initializer = mx.init.uniform(0.01), optimizer = "sgd", ctx = NULL, 
                             batch.end.callback = NULL, epoch.end.callback = NULL, 
                             kvstore = "local", verbose = TRUE) {
  
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
  
  sym_ini <- if (is.list(symbol)) symbol[[names(train.data$bucketID)]] else symbol
  
  arguments <- sym_ini$arguments
  input.names <- intersect(names(train.data$value()), arguments)
  
  input.shape <- sapply(input.names, function(n) {
    dim(train.data$value()[[n]])
  }, simplify = FALSE)
  
  shapes <- sym_ini$infer.shape(input.shape)
  
  # assign arg.params and aux.params arguments to arg.params.input and aux.params.input
  arg.params.input <- arg.params
  aux.params.input <- aux.params
  
  # initialize all arguments with zeros
  arg.params <- lapply(shapes$arg.shapes, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  # initialize input parameters
  dlist <- arg.params[input.names]
  
  # initialize parameters - only argument ending with _weight and _bias are initialized
  arg.params.ini <- mx.init.create(initializer = initializer, shape.array = shapes$arg.shapes, ctx = mx.cpu(), skip.unknown = TRUE)
  
  # assign initilized parameters to arg.params
  arg.params[names(arg.params.ini)] <- arg.params.ini
  
  # assign input params to arg.params
  arg.params[names(arg.params.input)] <- arg.params.input
  
  # remove input params from arg.params
  arg.params[input.names] <- NULL
  
  # Grad request
  grad.req <- rep("null", length(arguments))
  grad.req.write <- arguments %in% setdiff(names(arg.params.ini), fixed.params)
  grad.req[grad.req.write] <- "write"
  
  # Arg array order
  update_names <- c(input.names, names(arg.params))
  arg.update.idx <- match(arguments, update_names)
  
  # aux parameters setup
  aux.params <- lapply(shapes$aux.shapes, function(shape) {
    mx.nd.zeros(shape = shape, ctx = mx.cpu())
  })
  
  aux.params.ini <- mx.init.create(initializer, shapes$aux.shapes, ctx = mx.cpu(), skip.unknown = FALSE)
  if (length(aux.params) > 0) {
    aux.params[names(aux.params.ini)] <- aux.params.ini
  } else aux.params <- NULL
  
  aux.params[names(aux.params.input)] <- aux.params.input
  
  # kvstore initialization
  kvstore <- mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), 
                                             verbose = verbose)
  
  ### Execute training
  model <- mx.model.train.buckets(symbol = symbol, ctx = ctx,  train.data = train.data, eval.data = eval.data, 
                                  dlist = dlist,  arg.params = arg.params, aux.params = aux.params, 
                                  grad.req = grad.req, arg.update.idx = arg.update.idx, 
                                  optimizer = optimizer, metric = metric, 
                                  begin.round = begin.round, end.round = num.round, 
                                  batch.end.callback = batch.end.callback, epoch.end.callback = epoch.end.callback, 
                                  kvstore = kvstore, verbose = verbose)
  
  return(model)
}
