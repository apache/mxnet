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

namespace mxnet {
namespace op {


template<>
struct ForwardKernel<gpu> {
  template<typename DType, typename BType>
  static MSHADOW_XINLINE void Map(int i,
                                  DType *in_data,
                                  OType *out_data,
                                  DType *bins,
                                  size_t batch_size,
                                  size_t bins_length,
                                  bool right) {

    const auto data = in_data[i];
    const auto batch = i / batch_size;

    auto
        elem = right ? thrust::lower_bound(bins + bins_length * batch,
                                        bins + bins_length * (batch + 1),
                                        data)
                     : thrust::upper_bound(bins + bins_length * batch,
                                        bins + bins_length * (batch + 1),
                                        data);

    auto index = thrust::distance(bins, elem);
    out_data[i] = OType(index);
  }
};


    const auto data = in_data[i];
    const auto batch = i / batch_size;

    auto
        elem = right ? thrust::lower_bound(bins + bins_length * batch,
                                        bins + bins_length * (batch + 1),
                                        data)
                     : thrust::upper_bound(bins + bins_length * batch,
                                        bins + bins_length * (batch + 1),
                                        data);

    out_data[i] = thrust::distance(bins, elem);
  }
};


NNVM_REGISTER_OP(digitize)
.set_attr<FCompute>("FCompute<gpu>", DigitizeOpForward<gpu>);

}  // namespace op
}  // namespace mxnet
