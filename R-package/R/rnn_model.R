is.param.name <- function(name) {
    return (grepl('weight$', name) || grepl('bias$', name) ||
           grepl('gamma$', name) || grepl('beta$', name) )
}

# Initialize parameters
mx.model.init.params.rnn <- function(symbol, input.shape, initializer, ctx) {
  if (!is.mx.symbol(symbol)) stop("symbol need to be MXSymbol")
  slist <- symbol$infer.shape(input.shape)
  if (is.null(slist)) stop("Not enough information to get shapes")
  arg.params <- mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown=TRUE)
  aux.params <- mx.init.create(initializer, slist$aux.shapes, ctx, skip.unknown=FALSE)
  return(list(arg.params=arg.params, aux.params=aux.params))
}

# Initialize the data iter
mx.model.init.iter.rnn <- function(X, y, batch.size, is.train) {
  if (is.MXDataIter(X)) return(X)
  shape <- dim(X)
  if (is.null(shape)) {
    num.data <- length(X)
  } else {
    ndim <- length(shape)
    num.data <- shape[[ndim]]
  }
  if (is.null(y)) {
    if (is.train) stop("Need to provide parameter y for training with R arrays.")
    y <- c(1:num.data) * 0
  }

  batch.size <- min(num.data, batch.size)

  return(mx.io.arrayiter(X, y, batch.size=batch.size, shuffle=is.train))
}

# set up rnn model with rnn cells
setup.rnn.model <- function(rnn.sym, ctx,
                            num.rnn.layer, seq.len,
                            num.hidden, num.embed, num.label,
                            batch.size, input.size,
                            init.states.name,
                            initializer=mx.init.uniform(0.01),
                            dropout=0) {

    arg.names <- rnn.sym$arguments
    input.shapes <- list()
    for (name in arg.names) {
        if (name %in% init.states.name) {
            input.shapes[[name]] <- c(num.hidden, batch.size)
        }
        else if (grepl('data$', name) || grepl('label$', name) ) {
            if (seq.len == 1) {
                input.shapes[[name]] <- c(batch.size)
            } else {
            input.shapes[[name]] <- c(seq.len, batch.size)
            }
        }
    }
    params <- mx.model.init.params.rnn(rnn.sym, input.shapes, initializer, mx.cpu())
    args <- input.shapes
    args$symbol <- rnn.sym
    args$ctx <- ctx
    args$grad.req <- "add"
    rnn.exec <- do.call(mx.simple.bind, args)

    mx.exec.update.arg.arrays(rnn.exec, params$arg.params, match.name=TRUE)
    mx.exec.update.aux.arrays(rnn.exec, params$aux.params, match.name=TRUE)

    grad.arrays <- list()
    for (name in names(rnn.exec$ref.grad.arrays)) {
        if (is.param.name(name))
            grad.arrays[[name]] <- rnn.exec$ref.arg.arrays[[name]]*0
    }
    mx.exec.update.grad.arrays(rnn.exec, grad.arrays, match.name=TRUE)

    return (list(rnn.exec=rnn.exec, symbol=rnn.sym,
                 num.rnn.layer=num.rnn.layer, num.hidden=num.hidden,
                 seq.len=seq.len, batch.size=batch.size,
                 num.embed=num.embed))

}


calc.nll <- function(seq.label.probs, batch.size) {
    nll = - sum(log(seq.label.probs)) / batch.size
    return (nll)
}

get.label <- function(label, ctx) {
    label <- as.array(label)
    seq.len <- dim(label)[[1]]
    batch.size <- dim(label)[[2]]
    sm.label <- array(0, dim=c(seq.len*batch.size))
    for (seqidx in 1:seq.len) {
        sm.label[((seqidx-1)*batch.size+1) : (seqidx*batch.size)] <- label[seqidx,]
    }
    return (mx.nd.array(sm.label, ctx))
}


# training rnn model
train.rnn <- function (model, train.data, eval.data,
                       num.round, update.period,
                       init.states.name,
                       optimizer='sgd', ctx=mx.ctx.default(), 
                       epoch.end.callback,
                       batch.end.callback,
                       verbose=TRUE,
                       ...) {
    m <- model
    
    model <- list(symbol=model$symbol, arg.params=model$rnn.exec$ref.arg.arrays,
                  aux.params=model$rnn.exec$ref.aux.arrays)
    
    seq.len <- m$seq.len
    batch.size <- m$batch.size
    num.rnn.layer <- m$num.rnn.layer
    num.hidden <- m$num.hidden

    opt <- mx.opt.create(optimizer, rescale.grad=(1/batch.size), ...)

    updater <- mx.opt.get.updater(opt, m$rnn.exec$ref.arg.arrays)
    epoch.counter <- 0
    log.period <- max(as.integer(1000 / seq.len), 1)
    last.perp <- 10000000.0

    for (iteration in 1:num.round) {
        nbatch <- 0
        train.nll <- 0
        # reset states
        init.states <- list()
        for (name in init.states.name) {
            init.states[[name]] <- m$rnn.exec$ref.arg.arrays[[name]]*0
        }

        mx.exec.update.arg.arrays(m$rnn.exec, init.states, match.name=TRUE)

        tic <- Sys.time()

        train.data$reset()

        while (train.data$iter.next()) {
            # set rnn input
            rnn.input <- train.data$value()
            mx.exec.update.arg.arrays(m$rnn.exec, rnn.input, match.name=TRUE)

            mx.exec.forward(m$rnn.exec, is.train=TRUE)
            seq.label.probs <- mx.nd.choose.element.0index(m$rnn.exec$ref.outputs[["sm_output"]], get.label(m$rnn.exec$ref.arg.arrays[["label"]], ctx))

            mx.exec.backward(m$rnn.exec)
            init.states <- list()
            for (name in init.states.name) {
                init.states[[name]] <- m$rnn.exec$ref.arg.arrays[[name]]*0
            }

            mx.exec.update.arg.arrays(m$rnn.exec, init.states, match.name=TRUE)
            # update epoch counter
            epoch.counter <- epoch.counter + 1
            if (epoch.counter %% update.period == 0) {
                # the gradient of initial c and inital h should be zero
                init.grad <- list()
                for (name in init.states.name) {
                    init.grad[[name]] <- m$rnn.exec$ref.arg.arrays[[name]]*0
                }

                mx.exec.update.grad.arrays(m$rnn.exec, init.grad, match.name=TRUE)

                arg.blocks <- updater(m$rnn.exec$ref.arg.arrays, m$rnn.exec$ref.grad.arrays)

                mx.exec.update.arg.arrays(m$rnn.exec, arg.blocks, skip.null=TRUE)

                grad.arrays <- list()
                for (name in names(m$rnn.exec$ref.grad.arrays)) {
                    if (is.param.name(name))
                        grad.arrays[[name]] <- m$rnn.exec$ref.grad.arrays[[name]]*0
                }
                mx.exec.update.grad.arrays(m$rnn.exec, grad.arrays, match.name=TRUE)

            }

            train.nll <- train.nll + calc.nll(as.array(seq.label.probs), batch.size)

            nbatch <- nbatch + seq.len
            
            if (!is.null(batch.end.callback)) {
              batch.end.callback(iteration, nbatch, environment())
            }
            
            if ((epoch.counter %% log.period) == 0) {
                message(paste0("Epoch [", epoch.counter,
                           "] Train: NLL=", train.nll / nbatch,
                           ", Perp=", exp(train.nll / nbatch)))
            }
        }
        train.data$reset()
        # end of training loop
        toc <- Sys.time()
        message(paste0("Iter [", iteration,
                   "] Train: Time: ", as.numeric(toc - tic, units="secs"),
                   " sec, NLL=", train.nll / nbatch,
                   ", Perp=", exp(train.nll / nbatch)))

        if (!is.null(eval.data)) {
            val.nll <- 0.0
            # validation set, reset states
            init.states <- list()
            for (name in init.states.name) {
                init.states[[name]] <- m$rnn.exec$ref.arg.arrays[[name]]*0
            }
            mx.exec.update.arg.arrays(m$rnn.exec, init.states, match.name=TRUE)

            eval.data$reset()
            nbatch <- 0
            while (eval.data$iter.next()) {
                # set rnn input
                rnn.input <- eval.data$value()
                mx.exec.update.arg.arrays(m$rnn.exec, rnn.input, match.name=TRUE)
                mx.exec.forward(m$rnn.exec, is.train=FALSE)
                # probability of each label class, used to evaluate nll
                seq.label.probs <- mx.nd.choose.element.0index(m$rnn.exec$ref.outputs[["sm_output"]], get.label(m$rnn.exec$ref.arg.arrays[["label"]], ctx))
                # transfer the states
                init.states <- list()
                for (name in init.states.name) {
                    init.states[[name]] <- m$rnn.exec$ref.arg.arrays[[name]]*0
                }
                mx.exec.update.arg.arrays(m$rnn.exec, init.states, match.name=TRUE)
                val.nll <- val.nll + calc.nll(as.array(seq.label.probs), batch.size)
                nbatch <- nbatch + seq.len
            }
            eval.data$reset()
            perp <- exp(val.nll / nbatch)
            message(paste0("Iter [", iteration,
                       "] Val: NLL=", val.nll / nbatch,
                       ", Perp=", exp(val.nll / nbatch)))
        }
        # get the model out


        epoch_continue <- TRUE
        if (!is.null(epoch.end.callback)) {
          epoch_continue <- epoch.end.callback(iteration, 0, environment(), verbose = verbose)
        }
        
        if (!epoch_continue) {
          break
        }
    }

    return (m)
}

# check data and translate data into iterator if data is array/matrix
check.data <- function(data, batch.size, is.train) {
    if (!is.null(data) && !is.list(data) && !is.mx.dataiter(data)) {
        stop("The dataset should be either a mx.io.DataIter or a R list")
    }
    if (is.list(data)) {
        if (is.null(data$data) || is.null(data$label)){
            stop("Please provide dataset as list(data=R.array, label=R.array)")
        }
    data <- mx.model.init.iter.rnn(data$data, data$label, batch.size=batch.size, is.train = is.train)
    }
    if (!is.null(data) && !data$iter.next()) {
        data$reset()
        if (!data$iter.next()) stop("Empty input")
    }
    return (data)
}
