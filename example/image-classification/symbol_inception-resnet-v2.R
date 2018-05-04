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

# Inception resnet v2, suitable for images with around 299 x 299
#
# Reference:
# Szegedy C, Ioffe S, Vanhoucke V. Inception-v4, inception-resnet and 
# the impact of residual connections on learning, 2016.
# Link to the paper: https://arxiv.org/abs/1602.07261
#
library(mxnet)

Conv <- function(data, num_filter, kernel=c(1, 1), stride=c(1, 1), pad=c(0, 0), 
                 name, suffix="", withRelu=TRUE, withBn=FALSE){
  conv <- mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, 
                                stride=stride, pad=pad,
                                name=paste0(name, suffix, "_conv2d"))
  if (withBn){
    conv <- mx.symbol.BatchNorm(data=conv, name=paste0(name, suffix, "_bn"))
  }
  if (withRelu){
    conv <- mx.symbol.Activation(data=conv, act_type="relu", 
                                 name=paste0(name, suffix, "_relu"))
  }
  
  return(conv)
}

# Input Shape is 299*299*3 (th)
InceptionResnetStem <- function(data,
                                num_1_1, num_1_2, num_1_3,
                                num_2_1,
                                num_3_1, num_3_2,
                                num_4_1, num_4_2, num_4_3, num_4_4,
                                num_5_1,
                                name){
  stem_3x3 <- Conv(data=data, num_filter=num_1_1, kernel=c(3, 3), stride=c(2, 2),
                   name=paste0(name, "_conv"))
  stem_3x3 <- Conv(data=stem_3x3, num_filter=num_1_2, kernel=c(3, 3), 
                   name=paste0(name, "_stem"), suffix="_conv")
  stem_3x3 <- Conv(data=stem_3x3, num_filter=num_1_3, kernel=c(3, 3), pad=c(1,1),
                   name=paste0(name, "_stem"), suffix="_conv_1")

  
  pool1 <- mx.symbol.Pooling(data=stem_3x3, kernel=c(3, 3), stride=c(2, 2), 
                             pool_type="max", name=paste0("max_", name, "_pool1"))
  
  stem_1_3x3 <- Conv(data=stem_3x3, num_filter=num_2_1, kernel=c(3, 3), stride=c(2, 2), 
                     name=paste0(name, "_stem_1"), suffix="_conv_1")

  merge_lst <- list()
  merge_lst <- c(pool1, stem_1_3x3)
  merge_lst$num.args <- length(merge_lst)
  merge_lst$name <- paste0(name, "_concat1")
  concat1 <- mxnet:::mx.varg.symbol.Concat(merge_lst)
  
  stem_1_1x1 <- Conv(data=concat1, num_filter=num_3_1, name=paste0(name, "_stem_1"), suffix='_conv_2')
  stem_1_3x3 <- Conv(data=stem_1_1x1, num_filter=num_3_2, kernel=c(3, 3), 
                     name=paste0(name, "_stem_1"), suffix='_conv_3')
  stem_2_1x1 <- Conv(data=concat1, num_filter=num_4_1, name=paste0(name, "_stem_2"), suffix='_conv_1')
  stem_2_7x1 <- Conv(data=stem_2_1x1, num_filter=num_4_2, kernel=c(7, 1), pad=c(3, 0),
                     name=paste0(name, "_stem_2"), suffix='_conv_2')
  stem_2_1x7 <- Conv(data=stem_2_7x1, num_filter=num_4_3, kernel=c(1, 7), pad=c(0, 3),
                     name=paste0(name, "_stem_2"), suffix='_conv_3')
  stem_2_3x3 <- Conv(data=stem_2_1x7, num_filter=num_4_4, kernel=c(3, 3), 
                     name=paste0(name, "_stem_2"), suffix='_conv_4')

  merge_lst <- list()
  merge_lst <- c(stem_1_3x3, stem_2_3x3)
  merge_lst$num.args <- length(merge_lst)
  merge_lst$name <- paste0(name, "_concat2")
  concat2 <- mxnet:::mx.varg.symbol.Concat(merge_lst)
  
  pool2 <- mx.symbol.Pooling(data=concat2, kernel=c(3, 3), stride=c(2, 2), 
                             pool_type="max", name=paste0("max_", name, "_pool2"))
  
  stem_3_3x3 <- Conv(data=concat2, num_filter=num_5_1, kernel=c(3, 3), stride=c(2, 2),
                     name=paste0(name, "_stem_3"), suffix='_conv_1', withRelu=FALSE)
  
  merge_lst <- list()
  merge_lst <- c(pool2, stem_3_3x3)
  merge_lst$num.args <- length(merge_lst)
  merge_lst$name <- paste0(name, "_concat3")
  concat3 <- mxnet:::mx.varg.symbol.Concat(merge_lst)

  bn1 <- mx.symbol.BatchNorm(data=concat3, name=paste0(name, "_bn1"))
  act1 <- mx.symbol.Activation(data=bn1, act_type="relu", name=paste0(name, "_relu1"))
  
  return(act1)
}

InceptionResnetV2A <- function(data,
                             num_1_1,
                             num_2_1, num_2_2,
                             num_3_1, num_3_2, num_3_3,
                             proj,
                             name,
                             scaleResidual=TRUE){
  init <- data
  
  a1 <- Conv(data=data, num_filter=num_1_1, name=paste0(name, "_a_1"), suffix="_conv")
  
  a2 <- Conv(data=data, num_filter=num_2_1, name=paste0(name, "_a_2"), suffix="_conv_1")
  a2 <- Conv(data=a2, num_filter=num_2_2, kernel=c(3, 3), pad=c(1, 1), 
             name=paste0(name, "_a_2"), suffix="_conv_2")
  
  a3 <- Conv(data=data, num_filter=num_3_1, name=paste0(name, "_a_3"), suffix="_conv_1")
  a3 <- Conv(data=a3, num_filter=num_3_2, kernel=c(3, 3), pad=c(1, 1), 
             name=paste0(name, "_a_3"), suffix="_conv_2")
  a3 <- Conv(data=a3, num_filter=num_3_3, kernel=c(3, 3), pad=c(1, 1), 
             name=paste0(name, "_a_3"), suffix="_conv_3")

  merge_lst <- list()
  merge_lst <- c(a1, a2, a3)
  merge_lst$num.args <- length(merge_lst)
  merge_lst$name <- paste0(name, "_a_concat1")
  merge <- mxnet:::mx.varg.symbol.Concat(merge_lst)
  
  conv <- Conv(data=merge, num_filter=proj, name=paste0(name, "_a_liner_conv"), 
               withRelu=FALSE)
  if(scaleResidual){
    conv <- conv*0.1
  }
  
  out <- init + conv
  bn <- mx.symbol.BatchNorm(data=out, name=paste0(name, "_a_bn1"))
  act <- mx.symbol.Activation(data=bn, act_type="relu", name=paste0(name, "_a_relu1"))
  
  return(act)
}

InceptionResnetV2B <- function(data,
                             num_1_1,
                             num_2_1, num_2_2, num_2_3,
                             proj,
                             name,
                             scaleResidual=TRUE){
  init <- data
  
  b1 <- Conv(data=data, num_filter=num_1_1, name=paste0(name, "_b_1"), suffix="_conv")
  
  b2 <- Conv(data=data, num_filter=num_2_1, name=paste0(name, "_b_2"), suffix="_conv_1")
  b2 <- Conv(data=b2, num_filter=num_2_2, kernel=c(1, 7), pad=c(0, 3), 
             name=paste0(name, "_b_2"), suffix="_conv_2")
  b2 <- Conv(data=b2, num_filter=num_2_3, kernel=c(7, 1), pad=c(3, 0), 
             name=paste0(name, "_b_2"), suffix="_conv_3")
  
  merge_lst <- list()
  merge_lst <- c(b1, b2)
  merge_lst$num.args <- length(merge_lst)
  merge_lst$name <- paste0(name, "_b_concat1")
  merge <- mxnet:::mx.varg.symbol.Concat(merge_lst)
  
  conv <- Conv(data=merge, num_filter=proj, name=paste0(name, "_b_liner_conv"), 
               withRelu=FALSE)
  if(scaleResidual){
    conv <- conv*0.1
  }
  
  out <- init + conv
  bn <- mx.symbol.BatchNorm(data=out, name=paste0(name, "_b_bn1"))
  act <- mx.symbol.Activation(data=bn, act_type="relu", name=paste0(name, "_b_relu1"))
  
  return(act)
}

InceptionResnetV2C <- function(data,
                             num_1_1,
                             num_2_1, num_2_2, num_2_3,
                             proj,
                             name,
                             scaleResidual=TRUE){
  
  init <- data
  
  c1 <- Conv(data=data, num_filter=num_1_1, name=paste0(name, "_c_1"), suffix="_conv")
  
  c2 <- Conv(data=data, num_filter=num_2_1, name=paste0(name, "_c_2"), suffix="_conv_1")
  c2 <- Conv(data=c2, num_filter=num_2_2, kernel=c(1, 3), pad=c(0, 1), 
             name=paste0(name, "_c_2"), suffix="_conv_2")
  c2 <- Conv(data=c2, num_filter=num_2_3, kernel=c(3, 1), pad=c(1, 0), 
             name=paste0(name, "_c_2"), suffix="_conv_3")
  
  merge_lst <- list()
  merge_lst <- c(c1, c2)
  merge_lst$num.args <- length(merge_lst)
  merge_lst$name <- paste0(name, "_c_concat1")
  merge <- mxnet:::mx.varg.symbol.Concat(merge_lst)
  
  conv <- Conv(data=merge, num_filter=proj, name=paste0(name, "_b_liner_conv"), 
               withRelu=FALSE)
  if(scaleResidual){
    conv <- conv*0.1
  }
  
  out <- init + conv
  bn <- mx.symbol.BatchNorm(data=out, name=paste0(name, "_c_bn1"))
  act <- mx.symbol.Activation(data=bn, act_type="relu", name=paste0(name, "_c_relu1"))
  
  return(act)
}

ReductionResnetV2A <- function(data,
                             num_2_1,
                             num_3_1, num_3_2, num_3_3,
                             name){
  
  ra1 <- mx.symbol.Pooling(data=data, kernel=c(3, 3), stride=c(2, 2), 
                           pool_type="max", name=paste0("max_", name, "_pool1"))
  
  ra2 <- Conv(data=data, num_filter=num_2_1, kernel=c(3, 3), stride=c(2, 2), 
              name=paste0(name, "_ra_2"), suffix="_conv", withRelu=FALSE)
  
  ra3 <- Conv(data=data, num_filter=num_3_1, name=paste0(name, "_ra_3"), suffix="_conv_1")
  ra3 <- Conv(data=ra3, num_filter=num_3_2, kernel=c(3, 3), pad=c(1, 1), 
              name=paste0(name, "_ra_3"), suffix="_conv_2")
  ra3 <- Conv(data=ra3, num_filter=num_3_3, kernel=c(3, 3), stride=c(2, 2), 
              name=paste0(name, "_ra_3"), suffix="_conv_3", withRelu=FALSE)
  
  merge_lst <- list()
  merge_lst <- c(ra1, ra2, ra3)
  merge_lst$num.args <- length(merge_lst)
  merge_lst$name <- paste0(name, "_ra_concat1")
  m <- mxnet:::mx.varg.symbol.Concat(merge_lst)
  
  m <- mx.symbol.BatchNorm(data=m, name=paste0(name, "_ra_bn1"))
  m <- mx.symbol.Activation(data=m, act_type="relu", name=paste0(name, "_ra_relu1"))
  
  return(m)
}

ReductionResnetV2B <- function(data,
                             num_2_1, num_2_2,
                             num_3_1, num_3_2,
                             num_4_1, num_4_2, num_4_3,
                             name){
  rb1 <- mx.symbol.Pooling(data=data, kernel=c(3, 3), stride=c(2, 2), 
                           pool_type="max", name=paste0("max_", name, "_pool1"))
  
  rb2 <- Conv(data=data, num_filter=num_2_1, name=paste0(name, "_rb_2"), suffix="_conv_1")
  rb2 <- Conv(data=rb2, num_filter=num_2_2, kernel=c(3, 3), stride=c(2, 2), 
              name=paste0(name, "_rb_2"), suffix="_conv_2", withRelu=FALSE)
  
  rb3 <- Conv(data=data, num_filter=num_3_1, name=paste0(name, "_rb_3"), suffix="_conv_1")
  rb3 <- Conv(data=rb3, num_filter=num_3_2, kernel=c(3, 3), stride=c(2, 2), 
              name=paste0(name, "_rb_3"), suffix="_conv_2", withRelu=FALSE)
  
  rb4 <- Conv(data=data, num_filter=num_4_1, name=paste0(name, "_rb_4"), suffix="_conv_1")
  rb4 <- Conv(data=rb4, num_filter=num_4_2, kernel=c(3, 3), pad=c(1, 1), 
              name=paste0(name, "_rb_4"), suffix="_conv_2")
  rb4 <- Conv(data=rb4, num_filter=num_4_3, kernel=c(3, 3), stride=c(2, 2), 
              name=paste0(name, "_rb_4"), suffix="_conv_3", withRelu=FALSE)
  
  merge_lst <- list()
  merge_lst <- c(rb1, rb2, rb3, rb4)
  merge_lst$num.args <- length(merge_lst)
  merge_lst$name <- paste0(name, "_rb_concat1")
  m <- mxnet:::mx.varg.symbol.Concat(merge_lst)
  
  m <- mx.symbol.BatchNorm(data=m, name=paste0(name, "_rb_bn1"))
  m <- mx.symbol.Activation(data=m, act_type="relu", name=paste0(name, "_rb_relu1"))
  
  return(m)
}

circle_in3a <- function(data,
                        num_1_1,
                        num_2_1, num_2_2,
                        num_3_1, num_3_2, num_3_3,
                        proj,
                        name,
                        scale,
                        round){
  in3a <- data
  for(i in 1:round){
    in3a <- InceptionResnetV2A(in3a,
                             num_1_1,
                             num_2_1, num_2_2,
                             num_3_1, num_3_2, num_3_3,
                             proj,
                             paste0(name, "_", i),
                             scaleResidual=scale)
  }
  return(in3a)
  
}

circle_in2b <- function(data,
                        num_1_1,
                        num_2_1, num_2_2, num_2_3,
                        proj,
                        name,
                        scale,
                        round){
  in2b <- data
  for(i in 1:round){
    in2b <- InceptionResnetV2B(in2b,
                             num_1_1,
                             num_2_1, num_2_2, num_2_3,
                             proj,
                             paste0(name, "_", i),
                             scaleResidual=scale)
  }
  return(in2b)
}

circle_in2c <- function(data,
                        num_1_1,
                        num_2_1, num_2_2, num_2_3,
                        proj,
                        name,
                        scale,
                        round){
  in2c <- data
  for(i in 1:round){
    in2c <- InceptionResnetV2C(in2c,
                             num_1_1,
                             num_2_1, num_2_2, num_2_3,
                             proj,
                             paste0(name, "_", i),
                             scaleResidual=scale)
  }
  return(in2c)
}

# create inception-resnet-v1
get_symbol <- function(num_classes=1000, scale=TRUE){
  
  # input shape 229*229*3
  data <- mx.symbol.Variable(name="data")
  
  # stage stem
  num_1_1 <- 32
  num_1_2 <- 32
  num_1_3 <- 64
  num_2_1 <- 96
  num_3_1 <- 64
  num_3_2 <- 96
  num_4_1 <- 64
  num_4_2 <- 64
  num_4_3 <- 64
  num_4_4 <- 96
  num_5_1 <- 192
  
  in_stem <- InceptionResnetStem(data,
                                 num_1_1, num_1_2, num_1_3,
                                 num_2_1,
                                 num_3_1, num_3_2,
                                 num_4_1, num_4_2, num_4_3, num_4_4,
                                 num_5_1,
                                 "stem_stage")
  
  # stage 5 x Inception Resnet A
  num_1_1 <- 32
  num_2_1 <- 32
  num_2_2 <- 32
  num_3_1 <- 32
  num_3_2 <- 48
  num_3_3 <- 64
  proj <- 384
  
  in3a <- circle_in3a(in_stem,
                      num_1_1,
                      num_2_1, num_2_2,
                      num_3_1, num_3_2, num_3_3,
                      proj,
                      "in3a",
                      scale,
                      5)
  
  # stage Reduction Resnet A
  num_1_1 <- 384
  num_2_1 <- 256
  num_2_2 <- 256
  num_2_3 <- 384
  
  re3a <- ReductionResnetV2A(in3a,
                           num_1_1,
                           num_2_1, num_2_2, num_2_3,
                           "re3a")
  
  # stage 10 x Inception Resnet B
  num_1_1 <- 192
  num_2_1 <- 128
  num_2_2 <- 160
  num_2_3 <- 192
  proj <- 1152
  
  in2b <- circle_in2b(re3a,
                      num_1_1,
                      num_2_1, num_2_2, num_2_3,
                      proj,
                      "in2b",
                      scale,
                      10)
  
  # stage Reduction Resnet B
  num_1_1 <- 256
  num_1_2 <- 384
  num_2_1 <- 256
  num_2_2 <- 288
  num_3_1 <- 256
  num_3_2 <- 288
  num_3_3 <- 320
  
  re4b <- ReductionResnetV2B(in2b,
                           num_1_1, num_1_2,
                           num_2_1, num_2_2,
                           num_3_1, num_3_2, num_3_3,
                           "re4b")
  
  # stage 5 x Inception Resnet C
  num_1_1 <- 192
  num_2_1 <- 192
  num_2_2 <- 224
  num_2_3 <- 256
  proj <-  2144
  
  in2c <- circle_in2c(re4b,
                      num_1_1,
                      num_2_1, num_2_2, num_2_3,
                      proj,
                      "in2c",
                      scale,
                      5)
  
  # stage Average Pooling
  pool <- mx.symbol.Pooling(data=in2c, kernel=c(8, 8), stride=c(1, 1), 
                            pool_type="avg", name="global_pool")
  
  # stage Dropout
  dropout <- mx.symbol.Dropout(data=pool, p=0.2)
  # dropout =  mx.symbol.Dropout(data=pool, p=0.8)
  flatten <- mx.symbol.Flatten(data=dropout, name="flatten")
  
  # output
  fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name="fc1")
  softmax <- mx.symbol.SoftmaxOutput(data=fc1, name="softmax")
  
  return(softmax)
}