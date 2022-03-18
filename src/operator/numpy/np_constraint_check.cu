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
 * \file np_constraint_check.cu
 * \brief helper function for constraint check
 */

#include "./np_constraint_check.h"

namespace mxnet {
namespace op {

template <>
void GetReduceOutput<gpu>(mshadow::Stream<gpu>* s, const TBlob& output_blob, bool* red_output) {
  bool tmp            = true;
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  CUDA_CALL(cudaMemcpyAsync(
      &tmp, output_blob.dptr<bool>(), sizeof(bool), cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  *red_output = static_cast<bool>(tmp);
}

NNVM_REGISTER_OP(_npx_constraint_check)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs&, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", ConstraintCheckForward<gpu>);

}  // namespace op
}  // namespace mxnet
