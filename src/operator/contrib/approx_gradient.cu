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
 * \file approx_gradient.cu
 * \brief GPU Implementation of calculation of numerical gradient approximation
 * \author Andrei Ivanov
 */
#include "./approx_gradient-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {


template<>
size_t GetAdditionalMemorySizeA<gpu>(const int num_items) {
  float *d_in = nullptr;
  float *d_out = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_in, d_out, num_items);
  return temp_storage_bytes;
}

template<>
void ApproxGradientAction<gpu>(mshadow::Stream<gpu> *s,
                         float *workSpaceMemory,
                         size_t extraStorageBytes,
                         const TBlob& in0,
                         const TBlob& in1,
                         const TBlob& gradCoord,
                         const std::vector<OpReqType>& req,
                         const ApproxGradientParam& param) {
  const int num_items = in0.Size();

  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(in0.type_flag_, DType, {
      Kernel<vector_increment<kWriteTo>, gpu>::Launch(
        s, num_items, workSpaceMemory, in0.dptr<DType>(), in1.dptr<DType>(), param.eps);
  });

  float *pOutPntr = workSpaceMemory + num_items + extraStorageBytes;
  const auto index = param.index;
  if (!param.batched_mode) {
    cub::DeviceReduce::Sum(workSpaceMemory + num_items, extraStorageBytes,
                           workSpaceMemory, pOutPntr, num_items);

    MSHADOW_TYPE_SWITCH(in0.type_flag_, DType, {
      Kernel<approx_gradient<kWriteTo>, gpu>::Launch(
        s, 1, gradCoord.dptr<DType>() + index, pOutPntr);
    });
  } else {
    for (int i = 0; i < index; i++) {
      cub::DeviceReduce::Sum(workSpaceMemory + num_items, extraStorageBytes,
                             workSpaceMemory + i * index, pOutPntr, index);

      MSHADOW_TYPE_SWITCH(in0.type_flag_, DType, {
        Kernel<approx_gradient<kWriteTo>, gpu>::Launch(
          s, 1, gradCoord.dptr<DType>() + i, pOutPntr);
      });
    }
  }
}

NNVM_REGISTER_OP(_contrib_approx_gradient)
.set_attr<FCompute>("FCompute<gpu>", ApproxGradient<gpu>);

}  // namespace op
}  // namespace mxnet
