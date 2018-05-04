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

# 
# This file shows how to train ImageNet dataset with several Convolutional Neural Network architectures in R.
# More information: https://blogs.technet.microsoft.com/machinelearning/2016/11/15/imagenet-deep-neural-network-training-using-microsoft-r-server-and-azure-gpu-vms/
#
# To train ResNet-18:
# Rscript train_imagenet.R --network resnet --depth 18 --batch-size 512 --lr 0.1 --lr-factor 0.94 --gpu 0,1,2,3 --num-round 120 /
# --data-dir /path/to/data --train-dataset train.rec --val-dataset val.rec --log-dir $PWD --log-file resnet18-log.txt /
# --model-prefix resnet18 --kv-store device
#

# Train imagenet
require(mxnet)
require(argparse)

# Iterator
get_iterator <- function(args) {
  data.shape <- c(args$data_shape, args$data_shape, 3)
  train = mx.io.ImageRecordIter(
    path.imgrec     = file.path(args$data_dir, args$train_dataset),
    batch.size      = args$batch_size,
    data.shape      = data.shape,
    mean.r          = 123.68,
    mean.g          = 116.779,
    mean.b          = 103.939,
    rand.crop       = TRUE,
    rand.mirror     = TRUE
  )
  
  val = mx.io.ImageRecordIter(
    path.imgrec     = file.path(args$data_dir, args$val_dataset),
    batch.size      = args$batch_size,
    data.shape      = data.shape,
    mean.r          = 123.68,
    mean.g          = 116.779,
    mean.b          = 103.939,
    rand.crop       = FALSE,
    rand.mirror     = FALSE
  )
  ret = list(train=train, value=val)
}

# parse arguments
parse_args <- function() {
  parser <- ArgumentParser(description='train an image classifer on ImageNet')
  parser$add_argument('--network', type='character', default='resnet',
                      choices = c('resnet', 'inception-bn', 'googlenet', 'inception-resnet-v1',
                                  'inception-resnet-v2'),
                      help = 'the cnn to use')
  parser$add_argument('--data-dir', type='character', help='the input data directory')
  parser$add_argument('--gpus', type='character',
                      help='the gpus will be used, e.g "0,1,2,3"')
  parser$add_argument('--batch-size', type='integer', default=128,
                      help='the batch size')
  parser$add_argument('--lr', type='double', default=.01,
                      help='the initial learning rate')
  parser$add_argument('--lr-factor', type='double', default=1,
                      help='times the lr with a factor for every lr-factor-epoch epoch')
  parser$add_argument('--lr-factor-epoch', type='double', default=1,
                      help='the number of epoch to factor the lr, could be .5')
  parser$add_argument('--lr-multifactor', type='character', 
                      help='the epoch at which the lr is changed, e.g "15,30,45"')
  parser$add_argument('--mom', type='double', default=.9,
                      help='momentum for sgd')
  parser$add_argument('--wd', type='double', default=.0001,
                      help='weight decay for sgd')
  parser$add_argument('--clip-gradient', type='double', default=5,
                      help='clip min/max gradient to prevent extreme value')
  parser$add_argument('--model-prefix', type='character',
                      help='the prefix of the model to load/save')
  parser$add_argument('--load-epoch', type='integer',
                      help="load the model on an epoch using the model-prefix")
  parser$add_argument('--num-round', type='integer', default=10,
                      help='the number of iterations over training data to train the model')
  parser$add_argument('--kv-store', type='character', default='local',
                      help='the kvstore type')
  parser$add_argument('--num-examples', type='integer', default=1281167,
                      help='the number of training examples')
  parser$add_argument('--num-classes', type='integer', default=1000,
                      help='the number of classes')
  parser$add_argument('--log-file', type='character', 
                      help='the name of log file')
  parser$add_argument('--log-dir', type='character', default="/tmp/",
                      help='directory of the log file')
  parser$add_argument('--train-dataset', type='character', default="train.rec",
                      help='train dataset name')
  parser$add_argument('--val-dataset', type='character', default="val.rec",
                      help="validation dataset name")
  parser$add_argument('--data-shape', type='integer', default=224,
                      help='set images shape')
  parser$add_argument('--depth', type='integer',
                      help='the depth for resnet, it can be a value among 18, 50, 101, 152, 200, 269')
  parser$parse_args()
}
args <- parse_args()

# network
if (args$network == 'inception-bn'){
  source("symbol_inception-bn.R")
} else if (args$network == 'googlenet'){
  if(args$data_shape < 299) stop(paste0("The data shape for ", args$network, " has to be at least 299"))
  source("symbol_googlenet.R")
} else if (args$network == 'inception-resnet-v1'){
  if(args$data_shape < 299) stop(paste0("The data shape for ", args$network, " has to be at least 299"))
  source("symbol_inception-resnet-v1.R")
}  else if (args$network == 'inception-resnet-v2'){
  if(args$data_shape < 299) stop(paste0("The data shape for ", args$network, " has to be at least 299"))
  source("symbol_inception-resnet-v2.R")
} else if (args$network == 'resnet'){
  source("symbol_resnet-v2.R")
} else{
  stop("Wrong network")
}
if (is.null(args$depth)){
  net <- get_symbol(args$num_classes)
} else{
  net <- get_symbol(args$num_classes, args$depth)
}

# train
source("train_model.R")
train_model.fit(args, net, get_iterator(args))


