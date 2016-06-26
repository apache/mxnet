# rnn cell symbol
rnn <- function(num.hidden, indata, prev.state, param, seqidx, 
                layeridx, dropout=0., batch.norm=FALSE) {
    if (dropout > 0. )
        indata <- mx.symbol.Dropout(data=indata, p=dropout)
    i2h <- mx.symbol.FullyConnected(data=indata,
                                    weight=param$i2h.weight,
                                    bias=param$i2h.bias,
                                    num.hidden=num.hidden,
                                    name=paste0("t", seqidx, ".l", layeridx, ".i2h"))
    h2h <- mx.symbol.FullyConnected(data=prev.state$h,
                                    weight=param$h2h.weight,
                                    bias=param$h2h.bias,
                                    num.hidden=num.hidden,
                                    name=paste0("t", seqidx, ".l", layeridx, ".h2h"))
    hidden <- i2h + h2h

    hidden <- mx.symbol.Activation(data=hidden, act.type="tanh")
    if (batch.norm)
        hidden <- mx.symbol.BatchNorm(data=hidden)
    return (list(h=hidden))
}

# unrolled rnn network
rnn.unroll <- function(num.rnn.layer, seq.len, input.size, num.hidden, 
                       num.embed, num.label, dropout=0., batch.norm=FALSE) {
    embed.weight <- mx.symbol.Variable("embed.weight")
    cls.weight <- mx.symbol.Variable("cls.weight")
    cls.bias <- mx.symbol.Variable("cls.bias")
    param.cells <- lapply(1:num.rnn.layer, function(i) {
        cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                     i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                     h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                     h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
        return (cell)
    })
    last.states <- lapply(1:num.rnn.layer, function(i) {
        state <- list(h=mx.symbol.Variable(paste0("l", i, ".init.h")))
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
        # stack RNN
        for (i in 1:num.rnn.layer) {
            dp <- ifelse(i==1, 0, dropout)
            next.state <- rnn(num.hidden, indata=hidden,
                              prev.state=last.states[[i]],
                              param=param.cells[[i]],
                              seqidx=seqidx, layeridx=i, 
                              dropout=dp, batch.norm=batch.norm)
            hidden <- next.state$h
            last.states[[i]] <- next.state
        }
        # decoder
        if (dropout > 0.)
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

# rnn inference model symbol
rnn.inference.symbol <- function(num.rnn.layer, seq.len, input.size, num.hidden, 
                                 num.embed, num.label, dropout=0., batch.norm=FALSE) {
    seqidx <- 0
    embed.weight <- mx.symbol.Variable("embed.weight")
    cls.weight <- mx.symbol.Variable("cls.weight")
    cls.bias <- mx.symbol.Variable("cls.bias")
    param.cells <- lapply(1:num.rnn.layer, function(i) {
        cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                     i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                     h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                     h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
        return (cell)
    })
    last.states <- lapply(1:num.rnn.layer, function(i) {
        state <- list(h=mx.symbol.Variable(paste0("l", i, ".init.h")))
        return (state)
    })

    # embeding layer
    data <- mx.symbol.Variable("data")
    hidden <- mx.symbol.Embedding(data=data, input_dim=input.size,
                                 weight=embed.weight, output_dim=num.embed, name="embed")
    # stack RNN        
    for (i in 1:num.rnn.layer) {
        dp <- ifelse(i==1, 0, dropout)
        next.state <- rnn(num.hidden, indata=hidden,
                          prev.state=last.states[[i]],
                          param=param.cells[[i]],
                          seqidx=seqidx, layeridx=i, 
                          dropout=dp, batch.norm=batch.norm)
        hidden <- next.state$h
        last.states[[i]] <- next.state
    }
    # decoder
    if (dropout > 0.)
        hidden <- mx.symbol.Dropout(data=hidden, p=dropout)

    fc <- mx.symbol.FullyConnected(data=hidden,
                                   weight=cls.weight,
                                   bias=cls.bias,
                                   num_hidden=num.label)
    sm <- mx.symbol.SoftmaxOutput(data=fc, name='sm')
    unpack.h <- lapply(1:num.rnn.layer, function(i) {
        state <- last.states[[i]]
        state.h <- mx.symbol.BlockGrad(state$h, name=paste0("l", i, ".last.h"))
        return (state.h)
    })
    list.all <- c(sm, unpack.h)
    return (mx.symbol.Group(list.all))
}

#' Training RNN Unrolled Model
#'
#' @param train.data mx.io.DataIter or list(data=R.array, label=R.array)
#'      The Training set.
#' @param eval.data mx.io.DataIter or list(data=R.array, label=R.array), optional
#'      The validation set used for validation evaluation during the progress.
#' @param num.rnn.layer integer
#'      The number of the layer of rnn.
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
#' @param batch.norm boolean, default=FALSE
#'      Whether to use batch normalization.
#' @param ... other parameters passing to \code{mx.rnn}/.
#' @return model A trained rnn unrolled model.
#'
#' @export
mx.rnn <- function( train.data, eval.data=NULL,
                    num.rnn.layer, seq.len,
                    num.hidden, num.embed, num.label,
                    batch.size, input.size,
                    ctx=mx.ctx.default(),
                    num.round=10, update.period=1,
                    initializer=mx.init.uniform(0.01),
                    dropout=0, optimizer='sgd',
                    batch.norm=FALSE,
                    ...) {
    # check data and change data into iterator
    train.data <- check.data(train.data, batch.size, TRUE)
    eval.data <- check.data(eval.data, batch.size, FALSE)

    # get unrolled rnn symbol
    rnn.sym <- rnn.unroll( num.rnn.layer=num.rnn.layer,
                           num.hidden=num.hidden,
                           seq.len=seq.len,
                           input.size=input.size,
                           num.embed=num.embed,
                           num.label=num.label,
                           dropout=dropout,
                           batch.norm=batch.norm)
    init.states.name <- lapply(1:num.rnn.layer, function(i) {
        state <- paste0("l", i, ".init.h")
        return (state)
    })
    # set up rnn model
    model <- setup.rnn.model(rnn.sym=rnn.sym,
                             ctx=ctx,
                             num.rnn.layer=num.rnn.layer,
                             seq.len=seq.len,
                             num.hidden=num.hidden,
                             num.embed=num.embed,
                             num.label=num.label,
                             batch.size=batch.size,
                             input.size=input.size,
                             init.states.name=init.states.name,
                             initializer=initializer,
                             dropout=dropout)
    # train rnn model
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

#' Create a RNN Inference Model
#'
#' @param num.rnn.layer integer
#'      The number of the layer of rnn.
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
#' @param batch.norm boolean, default=FALSE
#'      Whether to use batch normalization.
#' @return model list(rnn.exec=integer, symbol=mxnet symbol, num.rnn.layer=integer, num.hidden=integer, seq.len=integer, batch.size=integer, num.embed=integer) 
#'      A rnn inference model.
#'
#' @export
mx.rnn.inference <- function( num.rnn.layer,
                              input.size,
                              num.hidden,
                              num.embed,
                              num.label,
                              batch.size=1,
                              arg.params,
                              ctx=mx.cpu(),
                              dropout=0.,
                              batch.norm=FALSE) {
    sym <- rnn.inference.symbol( num.rnn.layer=num.rnn.layer,
                                 input.size=input.size,
                                 num.hidden=num.hidden,
                                 num.embed=num.embed,
                                 num.label=num.label,
                                 dropout=dropout,
                                 batch.norm=batch.norm)
    # init.states.name <- c()
    # for (i in 1:num.rnn.layer) {
    #     init.states.name <- c(init.states.name, paste0("l", i, ".init.c"))
    #     init.states.name <- c(init.states.name, paste0("l", i, ".init.h"))
    # }
    init.states.name <- lapply(1:num.rnn.layer, function(i) {
        state <- paste0("l", i, ".init.h")
        return (state)
    })
    
    seq.len <- 1
    # set up rnn model
    model <- setup.rnn.model(rnn.sym=sym,
                             ctx=ctx,
                             num.rnn.layer=num.rnn.layer,
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
    for (i in 1:num.rnn.layer) {
        init.states[[paste0("l", i, ".init.h")]] <- model$rnn.exec$ref.arg.arrays[[paste0("l", i, ".init.h")]]*0
    }
    mx.exec.update.arg.arrays(model$rnn.exec, init.states, match.name=TRUE)

    return (model)
}

#' Using forward function to predict in rnn inference model
#'
#' @param model rnn model
#'      A rnn inference model
#' @param input.data, array.matrix
#'      The input data for forward function
#' @param new.seq boolean, default=FALSE
#'      Whether the input is the start of a new sequence
#'
#' @return result A list(prob=prob, model=model) containing the result probability of each label and the model.
#'
#' @export
mx.rnn.forward <- function(model, input.data, new.seq=FALSE) {
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
    #print (model$rnn.exec$ref)
    prob <- model$rnn.exec$ref.outputs[["sm_output"]]
    print ("prob")
    print (prob)
    return (list(prob=prob, model=model))
}
