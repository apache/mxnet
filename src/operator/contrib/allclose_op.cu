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

template<typename T>
size_t GetAdditionalMemory(mshadow::Stream<gpu> *s, const int num_items) {
  T *d_in = nullptr;
  T *d_out = nullptr;
  size_t temp_storage_bytes = 0;
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  cub::DeviceReduce::Min(nullptr, temp_storage_bytes, d_in, d_out, num_items, stream);
  return temp_storage_bytes;
}

template<>
size_t GetAdditionalMemoryLogical<gpu>(mshadow::Stream<gpu> *s, const int num_items) {
  return GetAdditionalMemory<INTERM_DATA_TYPE>(s, num_items);
}

template<>
void GetResultLogical<gpu>(mshadow::Stream<gpu> *s, INTERM_DATA_TYPE *workMem,
                           size_t extraStorageBytes, int num_items, INTERM_DATA_TYPE *outPntr) {
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  cub::DeviceReduce::Min(workMem + num_items, extraStorageBytes,
                         workMem, outPntr, num_items, stream);
}

NNVM_REGISTER_OP(_contrib_allclose)
.set_attr<FCompute>("FCompute<gpu>", AllClose<gpu>);

}  // namespace op
}  // namespace mxnet
