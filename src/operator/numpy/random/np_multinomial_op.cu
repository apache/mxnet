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
 * \file np_multinomial_op.cu
 * \brief Operator for numpy sampling from multinomial distributions
 */
#include "./np_multinomial_op.h"

namespace mxnet {
namespace op {

template <typename DType>
void CheckPvalGPU(const OpContext& ctx, DType* input, int prob_length) {
  std::vector<DType> pvals_(prob_length);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>());
  CUDA_CALL(cudaMemcpyAsync(
      &pvals_[0], input, sizeof(DType) * prob_length, cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  DType sum = DType(0.0);
  for (int i = 0; i < prob_length; ++i) {
    sum += pvals_[i];
    CHECK(sum <= DType(1.0 + 1e-12)) << "sum(pvals[:-1]) > 1.0";
  }
}

NNVM_REGISTER_OP(_npi_multinomial)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", NumpyMultinomialForward<gpu>);

}  // namespace op
}  // namespace mxnet
