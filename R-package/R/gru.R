# gru cell symbol
gru <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout=0) {
    if (dropout > 0)
        indata <- mx.symbol.Dropout(data=indata, p=dropout)
    i2h <- mx.symbol.FullyConnected(data=indata,
                                    weight=param$gates.i2h.weight,
                                    bias=param$gates.i2h.bias,
                                    num.hidden=num.hidden * 2,
                                    name=paste0("t", seqidx, ".l", layeridx, ".gates.i2h"))
    h2h <- mx.symbol.FullyConnected(data=prev.state$h,
                                    weight=param$gates.h2h.weight,
                                    bias=param$gates.h2h.bias,
                                    num.hidden=num.hidden * 2,
                                    name=paste0("t", seqidx, ".l", layeridx, ".gates.h2h"))
    gates <- i2h + h2h
    slice.gates <- mx.symbol.SliceChannel(gates, num.outputs=2,
                                          name=paste0("t", seqidx, ".l", layeridx, ".slice"))
    update.gate <- mx.symbol.Activation(slice.gates[[1]], act.type="sigmoid")
    reset.gate <- mx.symbol.Activation(slice.gates[[2]], act.type="sigmoid")

    htrans.i2h <- mx.symbol.FullyConnected(data=indata,
                                           weight=param$trans.i2h.weight,
                                           bias=param$trans.i2h.bias,
                                           num.hidden=num.hidden,
                                           name=paste0("t", seqidx, ".l", layeridx, ".trans.i2h"))
    h.after.reset <- prev.state$h * reset.gate
    htrans.h2h <- mx.symbol.FullyConnected(data=h.after.reset,
                                           weight=param$trans.h2h.weight,
                                           bias=param$trans.h2h.bias,
                                           num.hidden=num.hidden,
                                           name=paste0("t", seqidx, ".l", layeridx, ".trans.h2h"))
    h.trans <- htrans.i2h + htrans.h2h
    h.trans.active <- mx.symbol.Activation(h.trans, act.type="tanh")
    next.h <- prev.state$h + update.gate * (h.trans.active - prev.state$h)
    return (list(h=next.h))
}

# unrolled gru network
gru.unroll <- function(num.gru.layer, seq.len, input.size,
                       num.hidden, num.embed, num.label, dropout=0) {
    embed.weight <- mx.symbol.Variable("embed.weight")
    cls.weight <- mx.symbol.Variable("cls.weight")
    cls.bias <- mx.symbol.Variable("cls.bias")
    param.cells <- lapply(1:num.gru.layer, function(i) {
        cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")),
                     gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")),
                     gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")),
                     gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")),
                     trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")),
                     trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")),
                     trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")),
                     trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
        return (cell)
    })
    last.states <- lapply(1:num.gru.layer, function(i) {
        state <- list(h=mx.symbol.Variable(paste0("l", i, ".init.h")))
        return (state)
    })

    # embeding layer
    label <- mx.symbol.Variable("label")
    data <- mx.symbol.Variable("data")
    embed <- mx.symbol.Embedding(data=data, input.dim=input.size,
                                 weight=embed.weight, output.dim=num.embed, name='embed')
    wordvec <- mx.symbol.SliceChannel(data=embed, num.outputs=seq.len, squeeze.axis=1)

    last.hidden <- list()
    for (seqidx in 1:seq.len) {
        hidden <- wordvec[[seqidx]]
        # stack GRU
        for (i in 1:num.gru.layer) {
            dp <- ifelse(i==1, 0, dropout)
            next.state <- gru(num.hidden, indata=hidden,
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

    label <- mx.symbol.transpose(data=label)
    label <- mx.symbol.Reshape(data=label, target.shape=c(0))

    loss.all <- mx.symbol.SoftmaxOutput(data=fc, label=label, name="sm")
    return (loss.all)
}

# gru inference model symbol
gru.inference.symbol <- function(num.gru.layer, seq.len, input.size,
                                 num.hidden, num.embed, num.label, dropout=0) {
    seqidx <- 1
    embed.weight <- mx.symbol.Variable("embed.weight")
    cls.weight <- mx.symbol.Variable("cls.weight")
    cls.bias <- mx.symbol.Variable("cls.bias")

    param.cells <- lapply(1:num.gru.layer, function(i) {
        cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")),
                     gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")),
                     gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")),
                     gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")),
                     trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")),
                     trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")),
                     trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")),
                     trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
        return (cell)
    })
    last.states <- lapply(1:num.gru.layer, function(i) {
        state <- list(h=mx.symbol.Variable(paste0("l", i, ".init.h")))
        return (state)
    })

    # embeding layer
    data <- mx.symbol.Variable("data")
    hidden <- mx.symbol.Embedding(data=data, input_dim=input.size,
                                  weight=embed.weight, output_dim=num.embed, name="embed")

    # stack GRU
    for (i in 1:num.gru.layer) {
        dp <- ifelse(i==1, 0, dropout)
        next.state <- gru(num.hidden, indata=hidden,
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

    fc <- mx.symbol.FullyConnected(data=hidden, num_hidden=num.label,
                                   weight=cls.weight, bias=cls.bias, name='pred')
    sm <- mx.symbol.SoftmaxOutput(data=fc, name='sm')
    unpack.h <- lapply(1:num.gru.layer, function(i) {
        state <- last.states[[i]]
        state.h <- mx.symbol.BlockGrad(state$h, name=paste0("l", i, ".last.h"))
        return (state.h)
    })

    list.all <- c(sm, unpack.h)
    return (mx.symbol.Group(list.all))
}

#' Training GRU Unrolled Model
#'
#' @param train.data mx.io.DataIter or list(data=R.array, label=R.array)
#'      The Training set.
#' @param eval.data mx.io.DataIter or list(data=R.array, label=R.array), optional
#'      The validation set used for validation evaluation during the progress.
#' @param num.gru.layer integer
#'      The number of the layer of gru.
#' @param seq.len integer
#'      The length of the input sequence.
#' @param num.hidden integer
#'      The number of hidden nodes.
#' @param num.embed integer
#'      The output dim of embedding.
#' @param num.label  integer
#'      The number of labels.
#' @param batch.size integer
#'      The batch size used for R array training.
#' @param input.size integer
#'       The input dim of one-hot encoding of embedding
#' @param ctx mx.context, optional
#'      The device used to perform training.
#' @param num.round integer, default=10
#'      The number of iterations over training data to train the model.
#' @param update.period integer, default=1
#'      The number of iterations to update parameters during training period.
#' @param initializer initializer object. default=mx.init.uniform(0.01)
#'      The initialization scheme for parameters.
#' @param dropout float, default=0
#'      A number in [0,1) containing the dropout ratio from the last hidden layer to the output layer.
#' @param optimizer string, default="sgd"
#'      The optimization method.
#' @param ... other parameters passing to \code{mx.gru}/.
#' @return model A trained gru unrolled model.
#'
#' @export
mx.gru <- function( train.data, eval.data=NULL,
                    num.gru.layer, seq.len,
                    num.hidden, num.embed, num.label,
                    batch.size, input.size,
                    ctx=mx.ctx.default(),
                    num.round=10, update.period=1,
                    initializer=mx.init.uniform(0.01),
                    dropout=0, optimizer='sgd',
                    ...) {
    # check data and change data into iterator
    train.data <- check.data(train.data, batch.size, TRUE)
    eval.data <- check.data(eval.data, batch.size, FALSE)

    # get unrolled gru symbol
    rnn.sym <- gru.unroll( num.gru.layer=num.gru.layer,
                           num.hidden=num.hidden,
                           seq.len=seq.len,
                           input.size=input.size,
                           num.embed=num.embed,
                           num.label=num.label,
                           dropout=dropout)

    init.states.name <- lapply(1:num.gru.layer, function(i) {
        state.h <- paste0("l", i, ".init.h")
        return (state.h)
    })

    # set up gru model
    model <- setup.rnn.model(rnn.sym=rnn.sym,
                             ctx=ctx,
                             num.rnn.layer=num.gru.layer,
                             seq.len=seq.len,
                             num.hidden=num.hidden,
                             num.embed=num.embed,
                             num.label=num.label,
                             batch.size=batch.size,
                             input.size=input.size,
                             init.states.name=init.states.name,
                             initializer=initializer,
                             dropout=dropout)

    # train gru model
    model <- train.rnn( model, train.data, eval.data,
                        num.round=num.round,
                        update.period=update.period,
                        ctx=ctx,
                        init.states.name=init.states.name,
                        ...)
    # change model into MXFeedForwardModel
    model <- list(symbol=model$symbol, arg.params=model$rnn.exec$ref.arg.arrays, aux.params=model$rnn.exec$ref.aux.arrays)
    return(structure(model, class="MXFeedForwardModel"))
}

#' Create a GRU Inference Model
#'
#' @param num.gru.layer integer
#'      The number of the layer of gru.
#' @param input.size integer
#'       The input dim of one-hot encoding of embedding
#' @param num.hidden integer
#'      The number of hidden nodes.
#' @param num.embed integer
#'      The output dim of embedding.
#' @param num.label  integer
#'      The number of labels.
#' @param batch.size integer, default=1
#'      The batch size used for R array training.
#' @param arg.params list
#'      The batch size used for R array training.
#' @param ctx mx.context, optional
#'      Model parameter, list of name to NDArray of net's weights.
#' @param dropout float, default=0
#'      A number in [0,1) containing the dropout ratio from the last hidden layer to the output layer.
#' @return model list(rnn.exec=integer, symbol=mxnet symbol, num.rnn.layer=integer, num.hidden=integer, seq.len=integer, batch.size=integer, num.embed=integer) 
#'      A gru inference model.
#'
#' @export
mx.gru.inference <- function(num.gru.layer,
                             input.size,
                             num.hidden,
                             num.embed,
                             num.label,
                             batch.size=1,
                             arg.params,
                             ctx=mx.cpu(),
                             dropout=0.) {
    sym <- gru.inference.symbol(num.gru.layer=num.gru.layer,
                                 input.size=input.size,
                                 num.hidden=num.hidden,
                                 num.embed=num.embed,
                                 num.label=num.label,
                                 dropout=dropout)

    init.states.name <- lapply(1:num.gru.layer, function(i) {
        state.h <- paste0("l", i, ".init.h")
        return (state.h)
    })

    seq.len <- 1
    # set up gru model
    model <- setup.rnn.model(rnn.sym=sym,
                             ctx=ctx,
                             num.rnn.layer=num.gru.layer,
                             seq.len=seq.len,
                             num.hidden=num.hidden,
                             num.embed=num.embed,
                             num.label=num.label,
                             batch.size=batch.size,
                             input.size=input.size,
                             init.states.name=init.states.name,
                             initializer=mx.init.uniform(0.01),
                             dropout=dropout)
    arg.names <- names(model$rnn.exec$ref.arg.arrays)
    for (k in names(arg.params)) {
        if ((k %in% arg.names) && is.param.name(k) ) {
            rnn.input <- list()
            rnn.input[[k]] <- arg.params[[k]]
            mx.exec.update.arg.arrays(model$rnn.exec, rnn.input, match.name=TRUE)
        }
    }
    init.states <- list()
    for (i in 1:num.gru.layer) {
        init.states[[paste0("l", i, ".init.h")]] <- model$rnn.exec$ref.arg.arrays[[paste0("l", i, ".init.h")]]*0
    }
    mx.exec.update.arg.arrays(model$rnn.exec, init.states, match.name=TRUE)

    return (model)
}

#' Using forward function to predict in gru inference model
#'
#' @param model gru model
#'      A gru inference model
#' @param input.data, array.matrix
#'      The input data for forward function
#' @param new.seq boolean, default=FALSE
#'      Whether the input is the start of a new sequence
#'
#' @return result A list(prob=prob, model=model) containing the result probability of each label and the model.
#'
#' @export
mx.gru.forward <- function(model, input.data, new.seq=FALSE) {
    if (new.seq == TRUE) {
        init.states <- list()
        for (i in 1:model$num.rnn.layer) {
            init.states[[paste0("l", i, ".init.h")]] <- model$rnn.exec$ref.arg.arrays[[paste0("l", i, ".init.h")]]*0
        }
        mx.exec.update.arg.arrays(model$rnn.exec, init.states, match.name=TRUE)
    }
    dim(input.data) <- c(model$batch.size)
    data <- list(data=mx.nd.array(input.data))
    mx.exec.update.arg.arrays(model$rnn.exec, data, match.name=TRUE)
    mx.exec.forward(model$rnn.exec, is.train=FALSE)
    init.states <- list()
    for (i in 1:model$num.rnn.layer) {
        init.states[[paste0("l", i, ".init.h")]] <- model$rnn.exec$ref.outputs[[paste0("l", i, ".last.h_output")]]
    }
    mx.exec.update.arg.arrays(model$rnn.exec, init.states, match.name=TRUE)
    prob <- model$rnn.exec$ref.outputs[["sm_output"]]
    return (list(prob=prob, model=model))
}

