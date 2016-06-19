require(mxnet)

train_model.fit <- function(args, network, data_loader) {
    model_prefix <- args$model_prefix

    # save model
    if (is.null(model_prefix)) {
        checkpoint <- NULL
    } else {
        checkpoint <- mx.callback.save.checkpoint(model_prefix)
    }

    # data
    data <- data_loader(args)
    train <- data$train
    val <- data$value
    
    
    # train
    if (is.null(args$gpus)) {
        devs <- mx.cpu()  
    } else {
        devs <- lapply(unlist(strsplit(args$gpus, ",")), function(i) {
            mx.gpu(as.integer(i))
        })
    }

    model = mx.model.FeedForward.create(
        X                  = train,
        eval.data          = val,
        ctx                = devs,
        symbol             = network,
        eval.metric        = mx.metric.accuracy,
        num.round          = args$num_round,
        learning.rate      = args$lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        kvstore            = args$kv_store,
        array.batch.size   = args$batch_size,
        epoch.end.callback = checkpoint,
        batch.end.callback = mx.callback.log.train.metric(50))

}
