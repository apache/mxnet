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
 * Copyright (c) 2015 by Contributors
 * \file lrn.cu
 * \brief
 * \author Bing Xu
*/

#include "./lrn-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn_lrn-inl.h"
#endif

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(LRNParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNLocalResponseNormOp<DType>(param);
  })
#else
#if CUDA_VERSION == 7000
  LOG(FATAL) << "Due to old CUDA compiler bug, LRN is disabled."
             << "Please upgrade CUDA to 7.5+ or use CUDNN";
#else
  op = new LocalResponseNormOp<gpu>(param);
#endif  // CUDA_VERSION
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet


