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

    param.cells <- lapply(1:num.lstm.layer, function(i) {
        cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                     i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                     h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                     h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
        return (cell)
    })
    last.states <- lapply(1:num.lstm.layer, function(i) {
        state <- list(c=mx.symbol.Variable(paste0("l", i, ".init.c")),
                      h=mx.symbol.Variable(paste0("l", i, ".init.h")))
        return (state)
    })

    # embeding layer
    label <- mx.symbol.Variable("label")
    data <- mx.symbol.Variable("data")
    embed <- mx.symbol.Embedding(data=data, input_dim=input.size,
                                 weight=embed.weight, output_dim=num.embed, name="embed")
    wordvec <- mx.symbol.SliceChannel(data=embed, num_outputs=seq.len, squeeze_axis=1)

    last.hidden <- list()
    for (seqidx in 1:seq.len) {
        hidden <- wordvec[[seqidx]]
        # stack lstm
        for (i in 1:num.lstm.layer) {
            dp <- ifelse(i==1, 0, dropout)
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

    label <- mx.symbol.transpose(data=label)
    label <- mx.symbol.Reshape(data=label, target.shape=c(0))

    loss.all <- mx.symbol.SoftmaxOutput(data=fc, label=label, name="sm")
    return (loss.all)
}

# lstm inference model symbol
lstm.inference.symbol <- function(num.lstm.layer, input.size,
                                  num.hidden, num.embed, num.label, dropout=0.) {
    seqidx <- 0
    embed.weight <- mx.symbol.Variable("embed.weight")
    cls.weight <- mx.symbol.Variable("cls.weight")
    cls.bias <- mx.symbol.Variable("cls.bias")

    param.cells <- lapply(1:num.lstm.layer, function(i) {
        cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                     i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                     h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                     h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
        return (cell)
    })
    last.states <- lapply(1:num.lstm.layer, function(i) {
        state <- list(c=mx.symbol.Variable(paste0("l", i, ".init.c")),
                      h=mx.symbol.Variable(paste0("l", i, ".init.h")))
        return (state)
    })

    # embeding layer
    data <- mx.symbol.Variable("data")
    hidden <- mx.symbol.Embedding(data=data, input_dim=input.size,
                                  weight=embed.weight, output_dim=num.embed, name="embed")

    # stack lstm
    for (i in 1:num.lstm.layer) {
        dp <- ifelse(i==1, 0, dropout)
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

    fc <- mx.symbol.FullyConnected(data=hidden, num_hidden=num.label,
                                   weight=cls.weight, bias=cls.bias, name='pred')
    sm <- mx.symbol.SoftmaxOutput(data=fc, name='sm')
    unpack.c <- lapply(1:num.lstm.layer, function(i) {
        state <- last.states[[i]]
        state.c <- mx.symbol.BlockGrad(state$c, name=paste0("l", i, ".last.c"))
        return (state.c)
    })
    unpack.h <- lapply(1:num.lstm.layer, function(i) {
        state <- last.states[[i]]
        state.h <- mx.symbol.BlockGrad(state$h, name=paste0("l", i, ".last.h"))
        return (state.h)
    })

    list.all <- c(sm, unpack.c, unpack.h)
    return (mx.symbol.Group(list.all))
}



#' Training LSTM Unrolled Model
#'
#' @param train.data mx.io.DataIter or list(data=R.array, label=R.array)
#'      The Training set.
#' @param eval.data mx.io.DataIter or list(data=R.array, label=R.array), optional
#'      The validation set used for validation evaluation during the progress.
#' @param num.lstm.layer integer
#'      The number of the layer of lstm.
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
#' @param epoch.end.callback function, optional
#'     The callback when iteration ends.
#' @param batch.end.callback function, optional
#'     The callback when one mini-batch iteration ends.
#' @param ... other parameters passing to \code{mx.lstm}/.
#' @return model A trained lstm unrolled model.
#'
#' @export
mx.lstm <- function(train.data, eval.data=NULL,
                    num.lstm.layer, seq.len,
                    num.hidden, num.embed, num.label,
                    batch.size, input.size,
                    ctx=mx.ctx.default(),
                    num.round=10, update.period=1,
                    initializer=mx.init.uniform(0.01),
                    dropout=0, optimizer='sgd',
                    epoch.end.callback=NULL, batch.end.callback=NULL,
                    model,
                    arg.params,
                    ...) {
    # check data and change data into iterator
    train.data <- check.data(train.data, batch.size, TRUE)
    eval.data <- check.data(eval.data, batch.size, FALSE)
    
    

    # get unrolled lstm symbol
    if(missing(model)){
        rnn.sym <- lstm.unroll(num.lstm.layer=num.lstm.layer,
                           num.hidden=num.hidden,
                           seq.len=seq.len,
                           input.size=input.size,
                           num.embed=num.embed,
                           num.label=num.label,
                           dropout=dropout)
    } else {
      rnn.sym=model$symbol
    }

    init.states.c <- lapply(1:num.lstm.layer, function(i) {
        state.c <- paste0("l", i, ".init.c")
        return (state.c)
    })
    init.states.h <- lapply(1:num.lstm.layer, function(i) {
        state.h <- paste0("l", i, ".init.h")
        return (state.h)
    })
    init.states.name <- c(init.states.c, init.states.h)

    # set up lstm model
    model <- setup.rnn.model(rnn.sym=rnn.sym,
                             ctx=ctx,
                             num.rnn.layer=num.lstm.layer,
                             seq.len=seq.len,
                             num.hidden=num.hidden,
                             num.embed=num.embed,
                             num.label=num.label,
                             batch.size=batch.size,
                             input.size=input.size,
                             init.states.name=init.states.name,
                             initializer=initializer,
                             dropout=dropout)
    # restore states
    if (!missing(arg.params)){
      arg.names <- names(model$rnn.exec$ref.arg.arrays)
      for (k in names(arg.params)) {
        if ((k %in% arg.names) && is.param.name(k) ) {
          rnn.input <- list()
          rnn.input[[k]] <- arg.params[[k]]
          mx.exec.update.arg.arrays(model$rnn.exec, rnn.input, match.name=TRUE)
        }
      }
    }

    # train lstm model
    model <- train.rnn( model, train.data, eval.data,
                        num.round=num.round,
                        update.period=update.period,
                        ctx=ctx,
                        init.states.name=init.states.name,
                        epoch.end.callback=epoch.end.callback, 
                        batch.end.callback=batch.end.callback,
                        ...)
    # change model into MXFeedForwardModel
    model <- list(symbol=model$symbol, arg.params=model$rnn.exec$ref.arg.arrays, aux.params=model$rnn.exec$ref.aux.arrays)
    return(structure(model, class="MXFeedForwardModel"))
}


#' Create a LSTM Inference Model
#'
#' @param num.lstm.layer integer
#'      The number of the layer of lstm.
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
#'      A lstm inference model.
#'
#' @export
mx.lstm.inference <- function(num.lstm.layer,
                              input.size,
                              num.hidden,
                              num.embed,
                              num.label,
                              batch.size=1,
                              arg.params,
                              ctx=mx.cpu(),
                              dropout=0.) {
    sym <- lstm.inference.symbol(num.lstm.layer=num.lstm.layer,
                                 input.size=input.size,
                                 num.hidden=num.hidden,
                                 num.embed=num.embed,
                                 num.label=num.label,
                                 dropout=dropout)

    init.states.c <- lapply(1:num.lstm.layer, function(i) {
        state.c <- paste0("l", i, ".init.c")
        return (state.c)
    })
    init.states.h <- lapply(1:num.lstm.layer, function(i) {
        state.h <- paste0("l", i, ".init.h")
        return (state.h)
    })
    init.states.name <- c(init.states.c, init.states.h)

    seq.len <- 1
    # set up lstm model
    model <- setup.rnn.model(rnn.sym=sym,
                             ctx=ctx,
                             num.rnn.layer=num.lstm.layer,
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
    for (i in 1:num.lstm.layer) {
        init.states[[paste0("l", i, ".init.c")]] <- model$rnn.exec$ref.arg.arrays[[paste0("l", i, ".init.c")]]*0
        init.states[[paste0("l", i, ".init.h")]] <- model$rnn.exec$ref.arg.arrays[[paste0("l", i, ".init.h")]]*0
    }
    mx.exec.update.arg.arrays(model$rnn.exec, init.states, match.name=TRUE)

    return (model)
}

#' Using forward function to predict in lstm inference model
#'
#' @param model lstm model
#'      A Lstm inference model
#' @param input.data, array.matrix
#'      The input data for forward function
#' @param new.seq boolean, default=FALSE
#'      Whether the input is the start of a new sequence
#'
#' @return result A list(prob=prob, model=model) containing the result probability of each label and the model.
#'
#' @export
mx.lstm.forward <- function(model, input.data, new.seq=FALSE) {
    if (new.seq == TRUE) {
        init.states <- list()
        for (i in 1:model$num.rnn.layer) {
            init.states[[paste0("l", i, ".init.c")]] <- model$rnn.exec$ref.arg.arrays[[paste0("l", i, ".init.c")]]*0
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
        init.states[[paste0("l", i, ".init.c")]] <- model$rnn.exec$ref.outputs[[paste0("l", i, ".last.c_output")]]
        init.states[[paste0("l", i, ".init.h")]] <- model$rnn.exec$ref.outputs[[paste0("l", i, ".last.h_output")]]
    }
    mx.exec.update.arg.arrays(model$rnn.exec, init.states, match.name=TRUE)
    prob <- model$rnn.exec$ref.outputs[["sm_output"]]
    return (list(prob=prob, model=model))
}
