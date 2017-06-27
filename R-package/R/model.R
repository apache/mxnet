# slice the shape on the highest dimension
mx.model.slice.shape <- function(shape, nsplit) {
  ndim <- length(shape)
  batchsize <- shape[[ndim]]
  step <- as.integer((batchsize + nsplit - 1) / nsplit)
  lapply(0:(nsplit - 1), function(k) {
    begin = min(k * step, batchsize)
    end = min((k + 1) * step, batchsize)
    s <- shape
    s[[ndim]] = end - begin
    return(list(begin=begin, end=end, shape=s))
  })
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


# Extract model from executors
mx.model.extract.model <- function(symbol, train.execs) {
  reduce.sum <- function(x) Reduce("+", x)
  # Get the parameters
  ndevice <- length(train.execs)
  narg <- length(train.execs[[1]]$ref.arg.arrays)
  arg.params <- lapply(1:narg, function(k) {
    if (is.null(train.execs[[1]]$ref.grad.arrays[[k]])) {
      result <- NULL
    } else {
      result <- reduce.sum(lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.arg.arrays[[k]], mx.cpu())
      })) / ndevice
    }
    return(result)
  })
  names(arg.params) <- names(train.execs[[1]]$ref.arg.arrays)
  arg.params <- mx.util.filter.null(arg.params)
  # Get the auxiliary
  naux <- length(train.execs[[1]]$ref.aux.arrays)
  if (naux != 0) {
    aux.params <- lapply(1:naux, function(k) {
      reduce.sum(lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.aux.arrays[[k]], mx.cpu())
      })) / ndevice
    })
    names(aux.params) <- names(train.execs[[1]]$ref.aux.arrays)
  } else {
    aux.params <- list()
  }
  # Get the model
  model <- list(symbol=symbol, arg.params=arg.params, aux.params=aux.params)
  return(structure(model, class="MXFeedForwardModel"))
}

# decide what type of kvstore to use
mx.model.create.kvstore <- function(kvstore, arg.params, ndevice, verbose=TRUE) {
  if (is.MXKVStore(kvstore)) return (kvstore)
  if (!is.character(kvstore)) {
    stop("kvstore msut be either MXKVStore or a string")
  }
  if (ndevice == 1) return (NULL)
  if (kvstore == "local") {
    max.size <- max(as.integer(lapply(arg.params, length)))
    if (max.size < 1024 * 1024 * 16) {
      kvstore <- 'local_update_cpu'
    } else {
      kvstore <- 'local_allreduce_cpu'
    }
    if(verbose) message(paste0("Auto-select kvstore type = ", kvstore))
  }
  return(mx.kv.create(kvstore))
}

# Internal function to do multiple device training.
mx.model.train <- function(symbol, ctx, input.shape, output.shape,
                           arg.params, aux.params,
                           begin.round, end.round, optimizer,
                           train.data, eval.data,
                           metric,
                           epoch.end.callback,
                           batch.end.callback,
                           kvstore,
                           verbose=TRUE) {
  ndevice <- length(ctx)
  if(verbose) message(paste0("Start training with ", ndevice, " devices"))
  # create the executors
  sliceinfo <- mx.model.slice.shape(input.shape, ndevice)
  sliceinfo2 <- mx.model.slice.shape(output.shape, ndevice)
  arg_names <- arguments(symbol)
  tmp <- unlist(lapply(arg_names, function(a) {
    mxnet:::mx.util.str.endswith(a, "label")
  }))
  label_name <- arg_names[tmp]
  train.execs <- lapply(1:ndevice, function(i) {
    arg_lst <- list(symbol = symbol, ctx = ctx[[i]], grad.req = "write",
                    data=sliceinfo[[i]]$shape)
    arg_lst[[label_name]] = sliceinfo2[[i]]$shape
    do.call(mx.simple.bind, arg_lst)
  })
  # set the parameters into executors
  for (texec in train.execs) {
    mx.exec.update.arg.arrays(texec, arg.params, match.name=TRUE)
    mx.exec.update.aux.arrays(texec, aux.params, match.name=TRUE)
  }
  # KVStore related stuffs
  params.index <-
    as.integer(mx.util.filter.null(
      lapply(1:length(train.execs[[1]]$ref.grad.arrays), function(k) {
        if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL
      })))
  update.on.kvstore <- FALSE
  if (!is.null(kvstore) && kvstore$update.on.kvstore) {
    update.on.kvstore <- TRUE
    kvstore$set.optimizer(optimizer)
  } else {
    updaters <- lapply(1:ndevice, function(i) {
      mx.opt.get.updater(optimizer, train.execs[[i]]$ref.arg.arrays)
    })
  }
  if (!is.null(kvstore)) {
    kvstore$init(params.index, train.execs[[1]]$ref.arg.arrays[params.index])
  }
  # Get the input names
  input.names <- mx.model.check.arguments(symbol)

  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    while (train.data$iter.next()) {
      # Get input data slice
      dlist <- train.data$value()
      slices <- lapply(1:ndevice, function(i) {
        s <- sliceinfo[[i]]
        ret <- list(data=mx.nd.slice(dlist$data, s$begin, s$end),
                    label=mx.nd.slice(dlist$label, s$begin, s$end))
        return(ret)
      })
      # copy data to executor
      for (i in 1:ndevice) {
        s <- slices[[i]]
        names(s) <- input.names
        mx.exec.update.arg.arrays(train.execs[[i]], s, match.name=TRUE)
      }
      for (texec in train.execs) {
        mx.exec.forward(texec, is.train=TRUE)
      }
      # copy outputs to CPU
      out.preds <- lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
      })
      # backward pass
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
        arg.blocks <- lapply(1:ndevice, function(i) {
          updaters[[i]](train.execs[[i]]$ref.arg.arrays, train.execs[[i]]$ref.grad.arrays)
        })
        for (i in 1:ndevice) {
          mx.exec.update.arg.arrays(train.execs[[i]], arg.blocks[[i]], skip.null=TRUE)
        }
      }
      # Update the evaluation metrics
      if (!is.null(metric)) {
        for (i in 1 : ndevice) {
          train.metric <- metric$update(slices[[i]]$label, out.preds[[i]], train.metric)
        }
      }
      nbatch <- nbatch + 1
      if (!is.null(batch.end.callback)) {
        batch.end.callback(iteration, nbatch, environment())
      }
    }
    # reset training data
    train.data$reset()
    if (!is.null(metric)) {
      result <- metric$get(train.metric)
      if(verbose) message(paste0("[", iteration, "] Train-", result$name, "=", result$value))
    }
    if (!is.null(eval.data)) {
      if (!is.null(metric)) {
        eval.metric <- metric$init()
      }
      while (eval.data$iter.next()) {
        dlist <- eval.data$value()
        slices <- lapply(1:ndevice, function(i) {
          s <- sliceinfo[[i]]
          ret <- list(data=mx.nd.slice(dlist$data, s$begin, s$end),
                    label=mx.nd.slice(dlist$label, s$begin, s$end))
          return(ret)
        })
        for (i in 1:ndevice) {
          s <- slices[[i]]
          names(s) <- input.names
          mx.exec.update.arg.arrays(train.execs[[i]], s, match.name=TRUE)
        }
        for (texec in train.execs) {
          mx.exec.forward(texec, is.train=FALSE)
        }
        out.preds <- lapply(train.execs, function(texec) {
          mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
        })
        if (!is.null(metric)) {
          for (i in 1 : ndevice) {
            eval.metric <- metric$update(slices[[i]]$label, out.preds[[i]], eval.metric)
          }
        }
      }
      eval.data$reset()
      if (!is.null(metric)) {
        result <- metric$get(eval.metric)
        if(verbose) message(paste0("[", iteration, "] Validation-", result$name, "=", result$value))
      }
    } else {
      eval.metric <- NULL
    }
    # get the model out
    model <- mx.model.extract.model(symbol, train.execs)

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

# Initialize parameters
mx.model.init.params <- function(symbol, input.shape, output.shape, initializer, ctx) {
  if (!is.MXSymbol(symbol)) stop("symbol need to be MXSymbol")
  arg_names <- arguments(symbol)
  tmp <- unlist(lapply(arg_names, function(a) {
    mxnet:::mx.util.str.endswith(a, "label")
  }))
  label_name <- arg_names[tmp]
  arg_lst <- list(symbol = symbol, data=input.shape)
  arg_lst[[label_name]] = output.shape

  slist <- do.call(mx.symbol.infer.shape, arg_lst)
  if (is.null(slist)) stop("Not enough information to get shapes")
  arg.params <- mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown=TRUE)
  aux.params <- mx.init.create(initializer, slist$aux.shapes, ctx, skip.unknown=FALSE)
  return(list(arg.params=arg.params, aux.params=aux.params))
}

# Initialize the data iter
mx.model.init.iter <- function(X, y, batch.size, is.train) {
  if (is.MXDataIter(X)) return(X)
  if (is.null(y)) {
    if (is.train) stop("Need to provide parameter y for training with R arrays.")
    shape <- dim(X)
    ndim <- length(shape)
    y <- c(1:shape[[ndim]]) * 0
  }
  batch.size <- min(length(y), batch.size)
  return(mx.io.arrayiter(X, y, batch.size=batch.size, shuffle=is.train))
}

# select layout by matching shape, report error if nothing matches up.
mx.model.select.layout.train <- function(X, y) {
  if (is.null(y)) stop("Need to provide y for training")
  y <- as.array(y)
  dimX <- dim(X)
  dimy <- dim(y)
  if (length(dimX) != 2) return("colmajor")
  rowmajor <- 0
  colmajor <- 0
  if (dimX[[1]] == dimy[[1]]) rowmajor <- 1
  if (dimX[[length(dimX)]] == dimy[[length(dimy)]]) colmajor <- 1
  if (rowmajor + colmajor != 1) {
    stop("Cannot auto select array.layout, please specify this parameter")
  }
  if (rowmajor == 1) {
    warning("Auto detect layout of input matrix, use rowmajor..\n")
    return("rowmajor")
  } else{
    warning("Auto detect layout input matrix, use colmajor..\n")
    return("colmajor")
  }
}

# select layout by matching shape, report error if nothing matches up.
mx.model.select.layout.predict <- function(X, model) {
  dimX <- dim(X)
  if (length(dimX) != 2) return("colmajor")
  rowmajor <- 1
  colmajor <- 1
  # try row major
  ret <- mx.symbol.infer.shape(model$symbol, data=c(dimX[[2]], 1))
  if (!is.null(ret)) {
    names = names(model$arg.params)
    for (i in 1:length(names)) {
      if (any(ret$arg.shapes[[names[i]]] != dim(model$arg.params[[i]]))) {
        rowmajor <- 0
      }
    }
  }
  # try col major
  ret <- mx.symbol.infer.shape(model$symbol, data=c(dimX[[1]], 1))
  if (!is.null(ret)) {
    names = names(model$arg.params)
    for (i in 1:length(names)) {
      if (any(ret$arg.shapes[[names[i]]] != dim(model$arg.params[[i]]))) {
        colmajor <- 0
      }
    }
  }
  if (rowmajor + colmajor != 1) {
    stop("Cannot auto select array.layout, please specify this parameter")
  }
  if (rowmajor == 1) {
    warning("Auto detect layout of input matrix, use rowmajor..\n")
    return("rowmajor")
  } else{
    warning("Auto detect layout input matrix, use colmajor..\n")
    return("colmajor")
  }
}


#' Create a MXNet Feedforward neural net model with the specified training.
#'
#' @param symbol The symbolic configuration of the neural network.
#' @param X mx.io.DataIter or R array/matrix
#'     The training data.
#' @param y R array, optional label of the data
#'     This is only used when X is R array.
#' @param ctx mx.context or list of mx.context, optional
#'     The devices used to perform training.
#' @param begin.round integer (default=1)
#'     The initial iteration over the training data to train the model.
#' @param num.round integer (default=10)
#'     The number of iterations over training data to train the model.
#' @param optimizer string, default="sgd"
#'     The optimization method.
#' @param initializer, initializer object. default=mx.init.uniform(0.01)
#'     The initialization scheme for parameters.
#' @param eval.data mx.io.DataIter or list(data=R.array, label=R.array), optional
#'     The validation set used for validation evaluation during the progress
#' @param eval.metric function, optional
#'     The evaluation function on the results.
#' @param epoch.end.callback function, optional
#'     The callback when iteration ends.
#' @param batch.end.callback function, optional
#'     The callback when one mini-batch iteration ends.
#' @param array.batch.size integer (default=128)
#'     The batch size used for R array training.
#' @param array.layout can be "auto", "colmajor", "rowmajor", (detault=auto)
#'     The layout of array. "rowmajor" is only supported for two dimensional array.
#'     For matrix, "rowmajor" means dim(X) = c(nexample, nfeatures),
#'     "colmajor" means dim(X) = c(nfeatures, nexample)
#'     "auto" will auto detect the layout by match the feature size,
#'      and will report error when X is a square matrix to ask user to explicitly specify layout.
#' @param kvstore string (default="local")
#'     The parameter synchronization scheme in multiple devices.
#' @param verbose logical (default=TRUE)
#'     Specifies whether to print information on the iterations during training.     
#' @param arg.params list, optional
#'     Model parameter, list of name to NDArray of net's weights.
#' @param aux.params list, optional
#'     Model parameter, list of name to NDArray of net's auxiliary states.
#' @return model A trained mxnet model.
#'
#' @export

mx.model.FeedForward.create <-
function(symbol, X, y=NULL, ctx=NULL, begin.round=1,
         num.round=10, optimizer="sgd",
         initializer=mx.init.uniform(0.01),
         eval.data=NULL, eval.metric=NULL,
         epoch.end.callback=NULL, batch.end.callback=NULL,
         array.batch.size=128, array.layout="auto",
         kvstore="local",
         verbose=TRUE,
         arg.params=NULL, aux.params=NULL,
         ...) {
  if (is.array(X) || is.matrix(X)) {
    if (array.layout == "auto") {
      array.layout <- mx.model.select.layout.train(X, y)
    }
    if (array.layout == "rowmajor") {
      X <- t(X)
    }
  }
  X <- mx.model.init.iter(X, y, batch.size=array.batch.size, is.train=TRUE)
  if (!X$iter.next()) {
    X$reset()
    if (!X$iter.next()) stop("Empty input")
  }
  input.shape <- dim((X$value())$data)
  output.shape <- dim((X$value())$label)
  params <- mx.model.init.params(symbol, input.shape, output.shape, initializer, mx.cpu())
  if (!is.null(arg.params)) params$arg.params <- arg.params
  if (!is.null(aux.params)) params$aux.params <- aux.params
  if (is.null(ctx)) ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) stop("ctx must be mx.context or list of mx.context")
  if (is.character(optimizer)) {
    ndim <- length(input.shape)
    batchsize = input.shape[[ndim]]
    optimizer <- mx.opt.create(optimizer, rescale.grad=(1/batchsize), ...)
  }
  if (!is.null(eval.data) && !is.list(eval.data) && !is.mx.dataiter(eval.data)) {
    stop("The validation set should be either a mx.io.DataIter or a R list")
  }
  if (is.list(eval.data)) {
    if (is.null(eval.data$data) || is.null(eval.data$label)){
      stop("Please provide the validation set as list(data=R.array, label=R.array)")
    }
    if (is.array(eval.data$data) || is.matrix(eval.data$data)) {
      if (array.layout == "auto") {
        array.layout <- mx.model.select.layout.train(eval.data$data, eval.data$label)
      }
      if (array.layout == "rowmajor") {
        eval.data$data <- t(eval.data$data)
      }
    }
    eval.data <- mx.model.init.iter(eval.data$data, eval.data$label, batch.size=array.batch.size, is.train = TRUE)
  }
  kvstore <- mx.model.create.kvstore(kvstore, params$arg.params, length(ctx), verbose=verbose)
  model <- mx.model.train(symbol, ctx, input.shape, output.shape,
                          params$arg.params, params$aux.params,
                          begin.round, num.round, optimizer=optimizer,
                          train.data=X, eval.data=eval.data,
                          metric=eval.metric,
                          epoch.end.callback=epoch.end.callback,
                          batch.end.callback=batch.end.callback,
                          kvstore=kvstore, 
                          verbose=verbose)
  return (model)
}

#' Predict the outputs given a model and dataset.
#'
#' @param model The MXNet Model.
#' @param X The dataset to predict.
#' @param ctx mx.cpu() or mx.gpu(i) The device used to generate the prediction.
#' @param array.batch.size The batch size used in batching. Only used when X is R's array.
#' @param array.layout can be "auto", "colmajor", "rowmajor", (detault=auto)
#'     The layout of array. "rowmajor" is only supported for two dimensional array.
#'     For matrix, "rowmajor" means dim(X) = c(nexample, nfeatures),
#'     "colmajor" means dim(X) = c(nfeatures, nexample)
#'     "auto" will auto detect the layout by match the feature size,
#'      and will report error when X is a square matrix to ask user to explicitly specify layout.
#'
#' @export
predict.MXFeedForwardModel <- function(model, X, ctx=NULL, array.batch.size=128, array.layout="auto") {
  if (is.serialized(model)) model <- mx.unserialize(model)
  if (is.null(ctx)) ctx <- mx.ctx.default()
  if (is.array(X) || is.matrix(X)) {
    if (array.layout == "auto") {
      array.layout <- mx.model.select.layout.predict(X, model)
    }
    if (array.layout == "rowmajor") {
      X <- t(X)
    }
  }
  X <- mx.model.init.iter(X, NULL, batch.size=array.batch.size, is.train=FALSE)
  X$reset()
  if (!X$iter.next()) stop("Cannot predict on empty iterator")
  dlist = X$value()
  arg_lst <- list(symbol = model$symbol, ctx = ctx, data = dim(dlist$data), grad.req="null")

  pexec <- do.call(mx.simple.bind, arg_lst)
  mx.exec.update.arg.arrays(pexec, model$arg.params, match.name=TRUE)
  mx.exec.update.aux.arrays(pexec, model$aux.params, match.name=TRUE)
  packer <- mx.nd.arraypacker()
  X$reset()
  while (X$iter.next()) {
    dlist = X$value()
    mx.exec.update.arg.arrays(pexec, list(data=dlist$data), match.name=TRUE)
    mx.exec.forward(pexec, is.train=FALSE)
    out.pred <- mx.nd.copyto(pexec$ref.outputs[[1]], mx.cpu())
    padded <- X$num.pad()
    oshape <- dim(out.pred)
    ndim <- length(oshape)
    packer$push(mx.nd.slice(out.pred, 0, oshape[[ndim]] - padded))
  }
  X$reset()
  return(packer$get())
}

#' Load model checkpoint from file.
#'
#' @param prefix string prefix of the model name
#' @param iteration integer Iteration number of model we would like to load.
#'
#' @export
mx.model.load <- function(prefix, iteration) {
  symbol <- mx.symbol.load(paste0(prefix, "-symbol.json"))
  save.dict <- mx.nd.load(sprintf("%s-%04d.params", prefix, iteration))
  names <- names(save.dict)
  arg.index <- as.integer(mx.util.filter.null(lapply(1:length(names), function(i) {
    if (mx.util.str.startswith(names[[i]], "arg:")) i else NULL
  })))
  aux.index <- as.integer(mx.util.filter.null(lapply(1:length(names), function(i) {
    if (mx.util.str.startswith(names[[i]], "aux:")) i else NULL
  })))

  if (length(arg.index) != 0) {
    arg.params <- save.dict[arg.index]
    names(arg.params) <- as.character(lapply(names[arg.index], function(nm) {
      substr(nm, 5, nchar(nm))
    }))
  } else {
    arg.params <- list()
  }
  if (length(aux.index) != 0) {
    aux.params <- save.dict[aux.index]
    names(aux.params) <- as.character(lapply(names[aux.index], function(nm) {
      substr(nm, 5, nchar(nm))
    }))
  } else {
    aux.params <- list()
  }
  model <- list(symbol=symbol, arg.params=arg.params, aux.params=aux.params)
  return(structure(model, class="MXFeedForwardModel"))
}

#' Save model checkpoint into file.
#'
#' @param model The feedforward model to be saved.
#' @param prefix string prefix of the model name
#' @param iteration integer Iteration number of model we would like to load.
#'
#' @export
mx.model.save <- function(model, prefix, iteration) {
  arg.params <- model$arg.params
  aux.params <- model$aux.params
  names(arg.params) <- as.character(lapply(names(arg.params), function(nm) {
    paste0("arg:", nm)
  }))
  names(aux.params) <- as.character(lapply(names(aux.params), function(nm) {
    paste0("aux:", nm)
  }))
  save.dict <- append(arg.params, aux.params)
  mx.symbol.save(model$symbol, paste0(prefix, "-symbol.json"))
  mx.nd.save(save.dict, sprintf("%s-%04d.params", prefix, iteration))
}

#' Check if the model has been serialized into RData-compatiable format.
#'
#' @return Logical indicator
#'
#' @export
is.serialized <- function(model) {
  if (!is.null(model[['is.serialized']])) {
    return(model[['is.serialized']])
  } else {
    return(FALSE)
  }
}

#' Serialize MXNet model into RData-compatiable format.
#'
#' @param model The mxnet model
#' 
#' @export
mx.serialize <- function(model) {
  if (!is.serialized(model)) {
    model_rdata <- list()
    model_rdata[['symbol_json']] <- model$symbol$as.json()
    model_rdata[['arg.params']] <- lapply(model$arg.params, as.array)
    model_rdata[['aux.params']] <- lapply(model$aux.params, as.array)
    model_rdata[['is.serialized']] <- TRUE
    class(model_rdata) <- "MXFeedForwardModel"
    return(model_rdata)
  } else {
    return(model)
  }
}

#' Unserialize MXNet model from Robject.
#'
#' @param model The mxnet model loaded from RData files.
#' 
#' @export
mx.unserialize <- function(model) {
  if (!is.serialized(model)) {
    return(model)
  } else {
    symbol <- mx.symbol.load.json(model$symbol_json)
    arg.params <- lapply(model$arg.params, mx.nd.array)
    aux.params <- lapply(model$aux.params, mx.nd.array)
    model <- list(symbol=symbol, arg.params=arg.params, aux.params=aux.params)
    return(structure(model, class="MXFeedForwardModel"))    
  }
}
