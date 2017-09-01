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
 * \file sequence_mask.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./sequence_mask-inl.h"


namespace mshadow {
namespace cuda {

////////////////////////////////////////////////////////////////////////////////
// Cross-Entropy loss
template<int n_bits, typename DType>
__global__ void SequenceMaskKernel(Tensor<gpu, 3, DType> dst,
                    const Tensor<gpu, 1, DType> lengths, DType value) {
  const index_t smax = dst.size(0);
  const index_t bmax = lengths.size(1);
  const index_t nmax = dst.size(2);
  unsigned int batch = threadIdx.x + blockIdx.x * blockDim.x;

  // early return if out of bounds
  if (batch >= bmax)
    return;

  // loop over batches
    for (index_t s = lengths[batch]; s < smax; ++s)
      for (index_t r = 0; r < nmax; ++r)
      dst[s][batch][r] = value;
}

////////////////////////////////////////////////////////////////////////////////

template<typename DType>
inline void SequenceMask(const Tensor<gpu, 3, DType> &dst,
                         const Tensor<gpu, 1, DType> &lengths, DType value) {
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(1));
  CheckLaunchParam(dimGrid, dimBlock, "SequenceMask");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  SequenceMaskKernel<kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(dst, lengths, value);
}

////////////////////////////////////////////////////////////////////////////////
}  // namespace cuda

template<typename DType>
inline void SequenceMask(Tensor<gpu, 3, DType> dst,
                   const Tensor<gpu, 1, DType> &lengths, DType value) {
  cuda::SequenceMask(dst, lengths, value);
}

}  // namespace mshadow

////////////////////////////////////////////////////////////////////////////////

namespace mxnet {
namespace op {
template <> Operator *CreateOp<gpu>(SequenceMaskParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType,
                           { op = new SequenceMaskOp<gpu, DType>(param); })
  return op;
}

}  // namespace op
}  // namespace mxnet
