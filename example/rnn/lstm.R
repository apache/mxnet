require(mxnet)

# lstm cell symbol
lstm <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout=0) {
    if (dropout > 0)
        indata <- mx.symbol.Dropout(data=indata, p=dropout)
    i2h <- mx.symbol.FullyConnected(data=indata,
                                    weight=param$i2h.weight,
                                    bias=param$i2h.bias,
                                    num.hidden=num.hidden * 4,
                                    name=paste0("t", seqidx, ".l", layeridx, ".i2h"))
    h2h <- mx.symbol.FullyConnected(data=prev.state$h,
                                    weight=param$h2h.weight,
                                    bias=param$h2h.bias,
                                    num.hidden=num.hidden * 4,
                                    name=paste0("t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
    slice.gates <- mx.symbol.SliceChannel(gates, num.outputs=4,
                                          name=paste0("t", seqidx, ".l", layeridx, ".slice"))

    in.gate <- mx.symbol.Activation(slice.gates[[1]], act.type="sigmoid")
    in.transform <- mx.symbol.Activation(slice.gates[[2]], act.type="tanh")
    forget.gate <- mx.symbol.Activation(slice.gates[[3]], act.type="sigmoid")
    out.gate <- mx.symbol.Activation(slice.gates[[4]], act.type="sigmoid")
    next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
    next.h <- out.gate * mx.symbol.Activation(next.c, act.type="tanh")
    
    return (list(c=next.c, h=next.h))
}

# unrolled lstm network
lstm.unroll <- function(num.lstm.layer, seq.len, input.size,
                        num.hidden, num.embed, num.label, dropout=0.) {

    embed.weight <- mx.symbol.Variable("embed.weight")
    cls.weight <- mx.symbol.Variable("cls.weight")
    cls.bias <- mx.symbol.Variable("cls.bias")
    param.cells <- list()
    last.states <- list()
    for (i in 1:num.lstm.layer) {
        param.cells[[i]] <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                                 i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                                 h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                                 h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
        state <- list(c=mx.symbol.Variable(paste0("l", i, ".init.c")),
                      h=mx.symbol.Variable(paste0("l", i, ".init.h")))
        last.states[[i]] <- state
    }

    last.hidden <- list()
    label <- mx.symbol.Variable("label")
    for (seqidx in 1:seq.len) {
        # embeding layer
        data <- mx.symbol.Variable(paste0("t", seqidx, ".data"))

        hidden <- mx.symbol.Embedding(data=data, weight=embed.weight,
                                      input.dim=input.size,
                                      output.dim=num.embed,
                                      name=paste0("t", seqidx, ".embed"))
        
        # stack lstm
        for (i in 1:num.lstm.layer) {
            if (i==0) {
                dp <- 0
            }
            else {
                dp <- dropout
            }
            next.state <- lstm(num.hidden, indata=hidden,
                               prev.state=last.states[[i]],
                               param=param.cells[[i]],
                               seqidx=seqidx, layeridx=i, 
                               dropout=dp)
            hidden <- next.state$h
            last.states[[i]] <- next.state
        }
        # decoder
        if (dropout > 0)  
            hidden <- mx.symbol.Dropout(data=hidden, p=dropout)
        last.hidden <- c(last.hidden, hidden)
    }
    last.hidden$dim <- 0
    last.hidden$num.args <- seq.len
    concat <-mxnet:::mx.varg.symbol.Concat(last.hidden)
    fc <- mx.symbol.FullyConnected(data=concat,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num.hidden=num.label)
    loss.all <- mx.symbol.SoftmaxOutput(data=fc, label=label, name="sm")
    unpack.c <- list()
    unpack.h <- list()
    for (i in 1:num.lstm.layer) {
         state <- last.states[[i]]
         state <- list(c=mx.symbol.BlockGrad(state$c, name=paste0("l", i, ".last.c")),
                            h=mx.symbol.BlockGrad(state$h, name=paste0("l", i, ".last.h" )))
        last.states[[i]] <- state
        unpack.c <- c(unpack.c, state$c)
        unpack.h <- c(unpack.h, state$h)
    }
    list.all <- c(loss.all, unpack.c, unpack.h)

    return (mx.symbol.Group(list.all))
}

is.param.name <- function(name) {
    return (grepl('weight$', name) || grepl('bias$', name) || 
           grepl('gamma$', name) || grepl('beta$', name) )
}

mx.model.init.params <- function(symbol, input.shape, initializer, ctx) {
  if (!is.mx.symbol(symbol)) stop("symbol need to be MXSymbol")
  slist <- symbol$infer.shape(input.shape)
  if (is.null(slist)) stop("Not enough information to get shapes")
  arg.params <- mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown=TRUE)
  aux.params <- mx.init.create(initializer, slist$aux.shapes, ctx, skip.unknown=FALSE)
  return(list(arg.params=arg.params, aux.params=aux.params))
}

# set up rnn model with lstm cells
setup.rnn.model <- function(ctx,
                            num.lstm.layer, seq.len,
                            num.hidden, num.embed, num.label,
                            batch.size, input.size,
                            initializer=mx.init.uniform(0.01), 
                            dropout=0) {

    rnn.sym <- lstm.unroll(num.lstm.layer=num.lstm.layer,
                           num.hidden=num.hidden,
                           seq.len=seq.len,
                           input.size=input.size,
                           num.embed=num.embed,
                           num.label=num.label,
                           dropout=dropout)
    arg.names <- rnn.sym$arguments
    input.shapes <- list()
    for (name in arg.names) {
        if (grepl('init.c$', name) || grepl('init.h$', name)) {
            input.shapes[[name]] <- c(num.hidden, batch.size)
        }
        else if (grepl('data$', name)) {
            input.shapes[[name]] <- c(batch.size)
        }
    }
    
    params <- mx.model.init.params(rnn.sym, input.shapes, initializer, ctx)

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
                 num.lstm.layer=num.lstm.layer, num.hidden=num.hidden, 
                 seq.len=seq.len, batch.size=batch.size,
                 num.embed=num.embed))

}


get.rnn.inputs <- function(m, X, begin) {
    seq.len <- m$seq.len
    batch.size <- m$batch.size
    seq.labels <- array(0, dim=c(seq.len*batch.size))
    seq.data <- list()
    for (seqidx in 1:seq.len) {
        idx <- (begin + seqidx - 1) %% dim(X)[2] + 1
        next.idx <- (begin + seqidx) %% dim(X)[2] + 1
        x <- X[, idx]
        y <- X[, next.idx]

        seq.data[[paste0("t", seqidx, ".data")]] <- mx.nd.array(as.array(x))
        seq.labels[((seqidx-1)*batch.size+1) : (seqidx*batch.size)] <- y
    }
    seq.data$label <- mx.nd.array(seq.labels)
    return (seq.data)
}


calc.nll <- function(seq.label.probs, X, begin) {
    nll = - sum(log(seq.label.probs)) / length(X[,1])
    return (nll)
}

train.lstm <- function(model, X.train.batch, X.val.batch,
                       num.round, update.period,
                       optimizer='sgd', half.life=2, max.grad.norm = 5.0, ...) {
    X.train.batch.shape <- dim(X.train.batch)
    X.val.batch.shape <- dim(X.val.batch)
    cat(paste0("Training with train.shape=(", paste0(X.train.batch.shape, collapse=","), ")"), "\n")
    cat(paste0("Training with val.shape=(", paste0(X.val.batch.shape, collapse=","), ")"), "\n")

    m <- model
    seq.len <- m$seq.len
    batch.size <- m$batch.size
    num.lstm.layer <- m$num.lstm.layer
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
        for (i in 1:num.lstm.layer) {
            init.states[[paste0("l", i, ".init.c")]] <- mx.nd.zeros(c(num.hidden, batch.size))
            init.states[[paste0("l", i, ".init.h")]] <- mx.nd.zeros(c(num.hidden, batch.size))
        }
        mx.exec.update.arg.arrays(m$rnn.exec, init.states, match.name=TRUE) 

        tic <- Sys.time()

        stopifnot(dim(X.train.batch)[[2]] %% seq.len == 0)
        stopifnot(dim(X.val.batch)[[2]] %% seq.len == 0)

        for (begin in seq(1, dim(X.train.batch)[2], seq.len)) {
            # set rnn input
            rnn.input <- get.rnn.inputs(m, X.train.batch, begin=begin)
            mx.exec.update.arg.arrays(m$rnn.exec, rnn.input, match.name=TRUE) 

            mx.exec.forward(m$rnn.exec, is.train=TRUE)
            # probability of each label class, used to evaluate nll
            seq.label.probs <- mx.nd.choose.element.0index(m$rnn.exec$outputs[["sm_output"]], m$rnn.exec$arg.arrays[["label"]])
            mx.exec.backward(m$rnn.exec)
            # transfer the states
            init.states <- list()
            for (i in 1:num.lstm.layer) {
                init.states[[paste0("l", i, ".init.c")]] <- m$rnn.exec$outputs[[paste0("l", i, ".last.c_output")]]
                init.states[[paste0("l", i, ".init.h")]] <- m$rnn.exec$outputs[[paste0("l", i, ".last.h_output")]]
            }
            mx.exec.update.arg.arrays(m$rnn.exec, init.states, match.name=TRUE) 
            # update epoch counter
            epoch.counter <- epoch.counter + 1
            if (epoch.counter %% update.period == 0) {

                # the gradient of initial c and inital h should be zero
                init.grad <- list()
                for (i in 1:num.lstm.layer) {
                    init.grad[[paste0("l", i, ".init.c")]] <- m$rnn.exec$ref.arg.arrays[[paste0("l", i, ".init.c")]]*0
                    init.grad[[paste0("l", i, ".init.h")]] <- m$rnn.exec$ref.arg.arrays[[paste0("l", i, ".init.h")]]*0
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

            train.nll <- train.nll + calc.nll(as.array(seq.label.probs), X.train.batch, begin=begin)

            nbatch <- begin + seq.len
            if ((epoch.counter %% log.period) == 0) {
                cat(paste0("Epoch [", epoch.counter, 
                           "] Train: NLL=", train.nll / nbatch, 
                           ", Perp=", exp(train.nll / nbatch), "\n"))
            }
        }
        # end of training loop
        toc <- Sys.time()
        cat(paste0("Iter [", iteration, 
                   "] Train: Time: ", as.numeric(toc - tic, units="secs"),
                   " sec, NLL=", train.nll / nbatch,
                   ", Perp=", exp(train.nll / nbatch), "\n"))

        val.nll <- 0.0
        # validation set, reset states
        init.states <- list()
        for (i in 1:num.lstm.layer) {
            init.states[[paste0("l", i, ".init.c")]] <- mx.nd.zeros(c(num.hidden, batch.size))
            init.states[[paste0("l", i, ".init.h")]] <- mx.nd.zeros(c(num.hidden, batch.size))
        }
        mx.exec.update.arg.arrays(m$rnn.exec, init.states, match.name=TRUE) 

        for (begin in seq(1, dim(X.val.batch)[2], seq.len)) {
            # set rnn input
            rnn.input <- get.rnn.inputs(m, X.val.batch, begin=begin)
            mx.exec.update.arg.arrays(m$rnn.exec, rnn.input, match.name=TRUE) 
            mx.exec.forward(m$rnn.exec, is.train=FALSE)
            # probability of each label class, used to evaluate nll
            seq.label.probs <- mx.nd.choose.element.0index(m$rnn.exec$outputs[["sm_output"]], m$rnn.exec$arg.arrays[["label"]])
            # transfer the states
            init.states <- list()
            for (i in 1:num.lstm.layer) {
                init.states[[paste0("l", i, ".init.c")]] <- m$rnn.exec$outputs[[paste0("l", i, ".last.c_output")]]
                init.states[[paste0("l", i, ".init.h")]] <- m$rnn.exec$outputs[[paste0("l", i, ".last.h_output")]]
            }
            mx.exec.update.arg.arrays(m$rnn.exec, init.states, match.name=TRUE)
            val.nll <- val.nll + calc.nll(as.array(seq.label.probs), X.val.batch, begin=begin)
        }
        nbatch <- dim(X.val.batch)[2]
        perp <- exp(val.nll / nbatch)
        cat(paste0("Iter [", iteration,  
                   "] Val: NLL=", val.nll / nbatch,
                   ", Perp=", exp(val.nll / nbatch), "\n"))


    }
}