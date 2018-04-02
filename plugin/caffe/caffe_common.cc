/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2016 by Contributors
 * \file caffe_common.h
 * \brief Common functions for caffeOp and caffeLoss symbols
 * \author Haoran Wang
*/
#include<mshadow/tensor.h>
#include<caffe/common.hpp>
#include"caffe_common.h"

namespace mxnet {
namespace op {
namespace caffe {

// Cpu implementation of set_mode
template<>
void CaffeMode::SetMode<mshadow::cpu>() {
  ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
}

// Gpu implementation of set_mode
template<>
void CaffeMode::SetMode<mshadow::gpu>() {
  ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
}

}  // namespace caffe
}  // namespace op
}  // namespace mxnet
