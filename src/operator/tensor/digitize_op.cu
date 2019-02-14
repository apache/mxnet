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
* \author Jose Luis Contreras, Anton Chernov and contributors
*/

#include "./digitize_op.h"
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>

namespace mxnet {
namespace op {


template<>
struct ForwardKernel<gpu> {
  template<typename DType, typename OType>
  static MSHADOW_XINLINE void Map(int i,
                                  DType *in_data,
                                  OType *out_data,
                                  DType *bins,
                                  size_t batch_size,
                                  size_t bins_length,
                                  bool right) {
    const auto data = in_data[i];
    const auto batch_index = static_cast<size_t>(i) / batch_size;

    const auto begin = bins + bins_length * batch_index;
    const auto end = begin + bins_length;

    const auto elem = right ? thrust::lower_bound(thrust::device, begin, end, data)
                            : thrust::upper_bound(thrust::device, begin, end, data);

    const auto index = static_cast<uint64_t>(thrust::distance(begin, elem));
    out_data[i] = static_cast<OType>(index);
  }
};

NNVM_REGISTER_OP(digitize)
.set_attr<FCompute>("FCompute<gpu>", DigitizeOpForward<gpu>);

}  // namespace op
}  // namespace mxnet
