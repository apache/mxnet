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
 * \file rnn.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./rnn-inl.h"
#include <algorithm>
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn_rnn-inl.h"
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(RNNParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CuDNNRNNOp<DType>(param);
  })
#else
  LOG(FATAL) << "RNN is only available for cuDNN at the moment.";
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
  return op;
}

}  // namespace op
}  // namespace mxnet
