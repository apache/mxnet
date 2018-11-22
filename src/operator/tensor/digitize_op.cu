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
* Copyright (c) 2018 by Contributors
* \file digitize_op.cu
* \brief GPU Implementation of the digitize op
* \author Contributors
*/

#include "./digitize_op.h"

#include <thrust/binary_search.h>
#include <thrust/distance.h>

namespace mxnet{
namespace op {

  NNVM_REGISTER_OP(diag)
      .set_attr<FCompute>("FCompute<gpu>", DigitizeOpForward < gpu > );


  template<>
  void DigitizeOp::ForwardKernel::Map<gpu>(int i,
                                           const DType *in_data,
                                           DType *out_data,
                                           const mshadow::Tensor<gpu, 1, BType> bins,
                                           const bool right) {
    auto data = in_data[i];
    auto elem = right ? thrust::lower_bound(bins.dptr_, bins.dptr_ + bins.size(0), data)
                      : thrust::upper_bound(bins.dptr_, bins.dptr_ + bins.size(0), data);

    out_data[i] = thrust::distance(bins.dptr_, elem);
  }

}  // namespace op
}  // namespace mxnet