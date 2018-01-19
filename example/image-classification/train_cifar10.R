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
require(argparse)

get_iterator <- function(data.shape) {
  data_dir <- args$data_dir
  data.shape <- data.shape
  train <- mx.io.ImageRecordIter(
    path.imgrec     = paste0(data_dir, "train.rec"),
    batch.size      = args$batch_size,
    data.shape      = data.shape,
    rand.crop       = TRUE,
    rand.mirror     = TRUE,
    mean.img        = paste0(data_dir, "mean.bin")
  )
  
  val <- mx.io.ImageRecordIter(
    path.imgrec     = paste0(data_dir, "test.rec"),
    path.imglist    = paste0(data_dir, "test.lst"),
    batch.size      = args$batch_size,
    data.shape      = data.shape,
    rand.crop       = TRUE,
    rand.mirror     = TRUE,
    mean.img        = paste0(data_dir, "mean.bin")
  )
  ret <- list(train = train, value = val)
}

parse_args <- function() {
  parser <- ArgumentParser(description = 'train an image classifer on CIFAR10')
  parser$add_argument('--network',
                      type = 'character',
                      default = 'resnet-28-small',
                      choices = c('alexnet',
                                  'lenet',
                                  'resnet',
                                  'googlenet',
                                  'inception-bn-28-small',
                                  'resnet-28-small'),
                      help = 'the network to use')
  parser$add_argument('--data-dir',
                      type = 'character',
                      default = 'data/cifar10/',
                      help = 'the input data directory')
  # num-examples
  parser$add_argument('--cpu',
                      type = 'character',
                      default = F,
                      help = 'CPU will be used if true."')
  parser$add_argument('--gpus',
                      type = 'character',
                      default = "0",
                      help = 'the gpus will be used, e.g "0,1,2,3"')
  parser$add_argument('--batch-size',
                      type = 'integer',
                      default = 128,
                      help = 'the batch size')
  parser$add_argument('--lr',
                      type = 'double',
                      default = .05,
                      help = 'the initial learning rate')
  # lr-factor, lr-factor-epoch
  parser$add_argument('--model-prefix', type = 'character',
                      help = 'the prefix of the model to load/save')
  parser$add_argument('--resume-model-prefix', type = 'character',
                      help = 'resume prefix of the model to load/save')
  parser$add_argument('--num-round',
                      type = 'integer',
                      default = 10,
                      help = 'the number of iterations over training data to train the model')
  parser$add_argument('--kv-store',
                      type = 'character',
                      default = 'local',
                      help = 'the kvstore type')
  parser$parse_args()
}
args <- parse_args()

# load network definition
source(paste("symbol_", args$network, ".R", sep = ''))
print(paste0("Network used: ", args$network))
net <- get_symbol(10)

# save model
if (is.null(args$model_prefix)) {
  checkpoint <- NULL
} else {
  checkpoint <- mx.callback.save.checkpoint(args$model_prefix)
}

# data
data.shape <- c(28, 28, 3)
data <- get_iterator(data.shape = data.shape)
train <- data$train
val <- data$value

# train
if (args$cpu) {
  print("Computing with CPU")
  devs <- mx.cpu()
} else {
  print(paste0("GPU used: ", args$gpus))
  if (grepl(',', args$gpu)) {
    devs <- lapply(unlist(strsplit(args$gpus, ",")), function(i) {
      mx.gpu(as.integer(i))
    })
  } else {
    devs <- mx.gpu(as.integer(args$gpus))
  }
}

#train
model <- mx.model.FeedForward.create(
  X                  = train,
  eval.data          = val,
  ctx                = devs,
  symbol             = net,
  eval.metric        = mx.metric.accuracy,
  num.round          = args$num_round,
  learning.rate      = args$lr,
  momentum           = 0.9,
  wd                 = 0.00001,
  kvstore            = args$kv_store,
  array.batch.size   = args$batch_size,
  epoch.end.callback = checkpoint,
  batch.end.callback = mx.callback.log.train.metric(50),
  initializer        = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
  optimizer          = "sgd"
)
