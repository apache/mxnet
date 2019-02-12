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
 * \file allclose_op.cu
 * \brief GPU Implementation of allclose op
 * \author Andrei Ivanov
 */
#include "./allclose_op-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

template<>
size_t GetAdditionalMemorySize<gpu>(const int num_items) {
  float *d_in = nullptr;
  float *d_out = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Min(nullptr, temp_storage_bytes, d_in, d_out, num_items);
  return temp_storage_bytes;
}

template<>
void AllCloseAction<gpu>(mshadow::Stream<gpu> *s,
                         int *workspaceMemory,
                         size_t extraStorageBytes,
                         const TBlob& in0,
                         const TBlob& in1,
                         const std::vector<OpReqType>& req,
                         const AllCloseParam& param,
                         int *outPntr) {
  const int num_items = in0.Size();
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(in0.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<allclose_forward<req_type>, gpu>::Launch(
        s, num_items, workspaceMemory, in0.dptr<DType>(), in1.dptr<DType>(),
        param.rtol, param.atol, param.equal_nan);
    });
  });

  cub::DeviceReduce::Min(workspaceMemory + num_items, extraStorageBytes,
                         workspaceMemory, outPntr, num_items);
}

NNVM_REGISTER_OP(_contrib_allclose)
.set_attr<FCompute>("FCompute<gpu>", AllClose<gpu>);

}  // namespace op
}  // namespace mxnet
