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
 * \file caffe_operator_gpu.cc
 * \brief caffe operator
 * \author Haoran Wang
*/
#include "./caffe_op-inl.h"
namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(CaffeOpParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new CaffeOp<gpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new CaffeOp<gpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 layer is not supported by caffe";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  return op;
}

}  // namespace op
}  // namespace mxnet
