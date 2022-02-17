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
 * \file np_percentile_op.cu
 * \brief GPU Implementation of Numpy-compatible percentile
 */

#include "np_percentile_op-inl.h"

namespace mxnet {
namespace op {

struct is_valid_check {
  template <typename QType>
  MSHADOW_XINLINE static void Map(int i, char* invalid_ptr, const QType* data) {
    if (data[i] < 0.0 || data[i] > 100)
      *invalid_ptr = 1;
  }
};

template <typename QType, typename gpu>
bool CheckInvalidInput(mshadow::Stream<gpu>* s,
                       const QType* data,
                       const size_t& data_size,
                       char* is_valid_ptr) {
  using namespace mxnet_op;
  int32_t is_valid = 0;
  Kernel<set_zero, gpu>::Launch(s, 1, is_valid_ptr);
  Kernel<is_valid_check, gpu>::Launch(s, data_size, is_valid_ptr, data);
  CUDA_CALL(cudaMemcpyAsync(&is_valid,
                            is_valid_ptr,
                            sizeof(char),
                            cudaMemcpyDeviceToHost,
                            mshadow::Stream<gpu>::GetStream(s)));
  CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
  return is_valid == 0;
}

NNVM_REGISTER_OP(_npi_percentile)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", NumpyPercentileForward<gpu>);

}  // namespace op
}  // namespace mxnet
