# slice the shape on the highest dimension
mx.model.slice.shape <- function(shape, nsplit) {
  batchsize <- shape[[1]]
  step <- as.integer((batchsize + nsplit - 1) / nsplit)
  lapply(0:(nsplit - 1), function(k) {
    begin = min(k * step, batchsize)
    end = min((k + 1) * step, batchsize)
    s <- shape
    s[[1]] = end - begin
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
  class(model) <- "mx.model"
  return(model)
}

# Internal function to do multiple device training.
mx.model.train <- function(symbol, ctx, input.shape,
                           arg.params, aux.params,
                           begin.round, end.round, optimizer,
                           train.data, eval.data,
                           metric,
                           iter.end.callback,
                           epoch.end.callback) {
  ndevice <- length(ctx)
  sliceinfo <- mx.model.slice.shape(input.shape, ndevice)
  # create the executors
  train.execs <- lapply(1:ndevice, function(i) {
    mx.simple.bind(symbol, ctx=ctx[[i]], data=sliceinfo[[i]]$shape, grad.req=TRUE)
  })
  # set the parameters into executors
  for (texec in train.execs) {
    mx.exec.update.arg.arrays(texec, arg.params, match.name=TRUE)
    mx.exec.update.aux.arrays(texec, aux.params, match.name=TRUE)
  }
  # Get the input names
  input.names <- mx.model.check.arguments(symbol)
  # create the updaters
  updaters <- lapply(1:ndevice, function(i) {
    mx.opt.get.updater(optimizer, train.execs[[i]]$ref.arg.arrays)
  })

  for (iteration in begin.round:end.round) {
    nbatch <- 0
    train.metric <- metric$init()
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
      # get gradient from each device.
      grad.blocks <- lapply(train.execs, function(texec) {
        texec$grad.arrays
      })

      # update parameters
      arg.blocks <- lapply(1:ndevice, function(i) {
        updaters[[i]](train.execs[[i]]$ref.arg.arrays, grad.blocks[[i]])
      })
      # reset the parameters of executors
      for (i in 1:ndevice) {
        mx.exec.update.arg.arrays(train.execs[[i]], arg.blocks[[i]], skip.null=TRUE)
      }
      # Update the evaluation metrics
      for (i in 1 : ndevice) {
        train.metric <- metric$update(slices[[i]]$label, out.preds[[i]], train.metric)
      }
      nbatch <- nbatch + 1
      if (!is.null(epoch.end.callback)) {
        epoch.end.callback(iteration, nbatch, environment())
      }
    }
    # reset training data
    train.data$reset()
    result <- metric$get(train.metric)
    cat(paste0("Train-", result$name, "=", result$value, "\n"))
    # get the model out
    model <- mx.model.extract.model(symbol, train.execs)
    if (!is.null(iter.end.callback)) {
      iter.end.callback(iteration, environment())
    }
  }
  return(model)
}

# Initialize parameters
mx.model.init.params <- function(symbol, input.shape, initializer, ctx) {
  if (!is.MXSymbol(symbol)) stop("symbol need to be MXSymbol")
  slist <- mx.symbol.infer.shape(symbol, data=input.shape)
  if (is.null(slist)) stop("Not enough information to get shapes")
  arg.params <- mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown=TRUE)
  aux.params <- mx.init.create(initializer, slist$aux.shapes, ctx, skip.unknown=FALSE)
  return(list(arg.params=arg.params, aux.params=aux.params))
}

# Initialize the data iter
mx.model.init.iter <- function(X, y, is.train) {
  if (!is.MXDataIter(X)) {
    stop("Only accept MXDataIter for now")
  }
  return(X)
}

mx.model.FeedForward.create <-
function(symbol, X, y=NULL, ctx=NULL,
         num.round=10, optimizer="sgd",
         initializer=mx.init.uniform(0.01),
         eval.metric=mx.metric.accuracy,
         iter.end.callback=NULL, epoch.end.callback=NULL,
         ...) {
  X <- mx.model.init.iter(X, y)
  if (!X$iter.next()) {
    x$reset()
    if (!X$iter.next()) stop("Empty input")
  }
  input.shape <- dim((X$value())$data)
  params <- mx.model.init.params(symbol, input.shape, initializer, mx.cpu())
  if (is.null(ctx)) ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) stop("ctx must be mx.context or list of mx.context")
  if (is.character(optimizer)) {
    batchsize = input.shape[[1]]
    optimizer <- mx.opt.create(optimizer, rescale.grad=(1/batchsize), ...)
  }
  model <- mx.model.train(symbol, ctx, input.shape,
                          params$arg.params, params$aux.params,
                          1, num.round, optimizer=optimizer,
                          train.data=X, eval.data=NULL,
                          metric=eval.metric,
                          iter.end.callback=iter.end.callback,
                          epoch.end.callback=epoch.end.callback)
  return (model)
}

