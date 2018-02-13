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

###
# Reproducing parper:
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
###

library(mxnet)

residual_unit <- function(data, num_filter, stride, dim_match, name, bottle_neck=TRUE, bn_mom=0.9, workspace=512){
  if(bottle_neck){
    bn1 <- mx.symbol.BatchNorm(data=data, fix_gamma=FALSE, eps=2e-5, 
                               momentum=bn_mom, name=paste0(name,'_bn1'))
    act1 <- mx.symbol.Activation(data=bn1, act_type='relu', 
                                 name=paste0(name, '_relu1'))
    conv1 <- mx.symbol.Convolution(data=act1, num_filter=as.integer(num_filter*0.25), 
                                   kernel=c(1,1), stride=c(1,1), pad=c(0,0),
                                   no_bias=TRUE, workspace=workspace, 
                                   name=paste0(name,'_conv1'))
    bn2 <- mx.symbol.BatchNorm(data=conv1, fix_gamma=FALSE, eps=2e-5, 
                               momentum=bn_mom, name=paste0(name, '_bn2'))
    act2 <- mx.symbol.Activation(data=bn2, act_type='relu', name=paste0(name, '_relu2'))
    conv2 <- mx.symbol.Convolution(data=act2, num_filter=as.integer(num_filter*0.25), 
                                   kernel=c(3,3), stride=stride, pad=c(1,1),
                                   no_bias=TRUE, workspace=workspace, 
                                   name=paste0(name, '_conv2'))
    bn3 <- mx.symbol.BatchNorm(data=conv2, fix_gamma=FALSE, eps=2e-5, 
                               momentum=bn_mom, name=paste0(name, '_bn3'))
    act3 <- mx.symbol.Activation(data=bn3, act_type='relu', name=paste0(name,'_relu3'))
    conv3 <- mx.symbol.Convolution(data=act3, num_filter=num_filter, kernel=c(1,1), 
                                   stride=c(1,1), pad=c(0,0), no_bias=TRUE,
                                   workspace=workspace, name=paste0(name, '_conv3'))
    if (dim_match){
      shortcut <- data
    } else{
      shortcut <- mx.symbol.Convolution(data=act1, num_filter=num_filter, 
                                        kernel=c(1,1), stride=stride, no_bias=TRUE,
                                        workspace=workspace, name=paste0(name,'_sc'))
    }
    return (conv3 + shortcut)
  } else{
    bn1 <- mx.symbol.BatchNorm(data=data, fix_gamma=FALSE, momentum=bn_mom, 
                               eps=2e-5, name=paste0(name,'_bn1'))
    act1 <- mx.symbol.Activation(data=bn1, act_type='relu', name=paste0(name, '_relu1'))
    conv1 <- mx.symbol.Convolution(data=act1, num_filter=num_filter, kernel=c(3,3), 
                                   stride=stride, pad=c(1,1), no_bias=TRUE, 
                                   workspace=workspace, name=paste0(name,'_conv1'))
    bn2 <- mx.symbol.BatchNorm(data=conv1, fix_gamma=FALSE, momentum=bn_mom, 
                               eps=2e-5, name=paste0(name, '_bn2'))
    act2 <- mx.symbol.Activation(data=bn2, act_type='relu', 
                                 name=paste0(name, '_relu2'))
    conv2 <- mx.symbol.Convolution(data=act2, num_filter=num_filter, kernel=c(3,3), 
                                   stride=c(1,1), pad=c(1,1), no_bias=TRUE, 
                                   workspace=workspace, name=paste0(name, '_conv2'))
    if (dim_match){
      shortcut = data
    } else {
      shortcut <- mx.symbol.Convolution(data=act1, num_filter=num_filter, kernel=c(1,1), 
                                        stride=stride, no_bias=TRUE,
                                        workspace=workspace, name=paste0(name,'_sc'))
    }
    return (conv2 + shortcut)
  }
}



resnet <- function(units, num_stage, filter_list, num_class, bottle_neck=TRUE, 
                   bn_mom=0.9, workspace=512){
  num_unit <- length(units)
  if(num_unit != num_stage) stop("Number of units different from num_stage")
  data <- mx.symbol.Variable(name='data')
  data <- mx.symbol.BatchNorm(data=data, fix_gamma=TRUE, eps=2e-5, momentum=bn_mom, 
                              name='bn_data')
  body <- mx.symbol.Convolution(data=data, num_filter=filter_list[1], kernel=c(7, 7), 
                                stride=c(2,2), pad=c(3, 3),
                                no_bias=TRUE, name="conv0", workspace=workspace)
  body <- mx.symbol.BatchNorm(data=body, fix_gamma=FALSE, eps=2e-5, 
                              momentum=bn_mom, name='bn0')
  body <- mx.symbol.Activation(data=body, act_type='relu', name='relu0')
  body <- mx.symbol.Pooling(data=body, kernel=c(3, 3), stride=c(2,2), 
                            pad=c(1,1), pool_type='max')
  
  
  for(i in 1:num_stage){
    if(i==1) stride <- c(1,1)
    else stride <- c(2,2)
    body <- residual_unit(body, filter_list[i+1], stride, FALSE,
                          name=paste0('stage', i, '_unit1') , 
                          bottle_neck=bottle_neck, workspace=workspace)
                          for(j in 1:(units[i]-1)){
                            body <- residual_unit(body, filter_list[i+1], c(1,1), 
                                                  TRUE, name=paste0('stage',i, '_unit', j + 1),
                                                  bottle_neck=bottle_neck, 
                                                  workspace=workspace)
                          }
  }
  bn1 <- mx.symbol.BatchNorm(data=body, fix_gamma=FALSE, eps=2e-5, 
                             momentum=bn_mom, name='bn1')
  relu1 <- mx.symbol.Activation(data=bn1, act_type='relu', name='relu1')
  # Although kernel is not used here when global_pool=TRUE, we should put one
  pool1 <-  mx.symbol.Pooling(data=relu1, global_pool=TRUE, kernel=c(7, 7), 
                              pool_type='avg', name='pool1')
  flat <- mx.symbol.Flatten(data=pool1)
  fc1 <- mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
  resnet <- mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
  return(resnet)
}

get_symbol <- function(num_class, depth=18){
  if (depth == 18){
    units <- c(2, 2, 2, 2)
  } else if (depth == 34){
    units = c(3, 4, 6, 3)
  } else if (depth == 50){
    units = c(3, 4, 6, 3)
  } else if (depth == 101){
    units = c(3, 4, 23, 3)
  } else if (depth == 152){
    units = c(3, 8, 36, 3)
  } else if (depth == 200){
    units = c(3, 24, 36, 3) 
  } else if (depth == 269){
    units = c(3, 30, 48, 8)
  } else{
    stop(paste0("no experiments done on depth ", depth))
  }
  
  if (depth >=50){
    filter_list <- c(64, 256, 512, 1024, 2048)
    bottle_neck <- TRUE
  } else{
    filter_list <- c(64, 64, 128, 256, 512)
    bottle_neck <- FALSE
  }
  bn_mom <- 0.9 #momentum of batch normalization
  workspace <- 500
  symbol <- resnet(units=units, num_stage=4, filter_list=filter_list, 
                   num_class=num_class, bottle_neck=bottle_neck, 
                   bn_mom=bn_mom, workspace=workspace)
  return(symbol)
}





