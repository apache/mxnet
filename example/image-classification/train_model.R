# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

require(mxnet)

train_model.fit <- function(args, network, data_loader) {
    
    # log
    if(!is.null(args$log_file)){
      sink(file.path(args$log_dir, args$log_file), append = FALSE, 
           type=c("output", "message"))
      cat(paste0("Starting computation of ", args$network, " at ", Sys.time(), "\n"))
    }
    cat("Arguments")
    print(unlist(args))

    # save model
    if (is.null(args$model_prefix)) {
        checkpoint <- NULL
    } else {
        checkpoint <- mx.callback.save.checkpoint(args$model_prefix)
    }

    # load pretrained model
    if(!is.null(args$load_epoch)){
      if(is.null(args$model_prefix)) stop("model_prefix should not be empty")
      begin.round <- args$load_epoch
      model <- mx.model.load(args$model_prefix, iteration=begin.round)
      network <- model$symbol
      arg.params <- model$arg.params
      aux.params <- model$aux.params
    } else{
      arg.params <- NULL
      aux.params <- NULL
      begin.round <- 1
    }

    # data
    data <- data_loader(args)
    train <- data$train
    val <- data$value 
    
    # devices
    if (is.null(args$gpus)) {
        devs <- mx.cpu()  
    } else {
        devs <- lapply(unlist(strsplit(args$gpus, ",")), function(i) {
            mx.gpu(as.integer(i))
        })
    }

    # learning rate scheduler
    if (args$lr_factor < 1){
      epoch_size <- as.integer(max(args$num_examples/args$batch_size), 1)
      if(!is.null(args$lr_multifactor)){
        step <- as.integer(strsplit(args$lr_multifactor,",")[[1]])
        step.updated <- step - begin.round + 1
        step.updated <- step.updated[step.updated > 0]
        step_batch <- epoch_size*step.updated 
        lr_scheduler <- mx.lr_scheduler.MultiFactorScheduler(step=step_batch, factor_val=args$lr_factor)
      } else{
        lr_scheduler <- mx.lr_scheduler.FactorScheduler(
          step = as.integer(max(epoch_size * args$lr_factor_epoch, 1)),
          factor_val = args$lr_factor)
      }
    } else{
      lr_scheduler = NULL
    }

    # train
    model <- mx.model.FeedForward.create(
      X                  = train,
      eval.data          = val,
      ctx                = devs,
      symbol             = network,
      begin.round        = begin.round,
      eval.metric        = mx.metric.top_k_accuracy,
      num.round          = args$num_round,
      learning.rate      = args$lr,
      momentum           = args$mom,
      wd                 = args$wd,
      kvstore            = args$kv_store,
      array.batch.size   = args$batch_size,
      clip_gradient      = args$clip_gradient,
      lr_scheduler       = lr_scheduler,
      optimizer          = "sgd",
      initializer        = mx.init.Xavier(factor_type="in", magnitude=2),
      arg.params         = arg.params,
      aux.params         = aux.params,
      epoch.end.callback = checkpoint,
      batch.end.callback = mx.callback.log.train.metric(50))

}
