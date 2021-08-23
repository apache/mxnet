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
 * \file sequence_mask-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_NN_SEQUENCE_MASK_INL_H_
#define MXNET_OPERATOR_NN_SEQUENCE_MASK_INL_H_

namespace mxnet {
namespace op {
namespace mxnet_op {

template <typename DType, typename LType>
inline void SequenceMask(const mshadow::Tensor<cpu, 3, DType> &dst,
                         const mshadow::Tensor<cpu, 1, LType> label, DType value) {
  for (index_t b = 0; b < dst.size(1); ++b)
    for (index_t s = label[b]; s < dst.size(0); ++s)
      for (index_t r = 0; r < dst.size(2); ++r)
        dst[s][b][r] = value;
}

#ifdef __CUDACC__
template<int n_bits, typename DType, typename LType>
__global__ void SequenceMaskKernel(mshadow::Tensor<gpu, 3, DType> dst,
                                   const mshadow::Tensor<gpu, 1, LType> lengths, DType value) {
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

template<typename DType, typename LType>
inline void SequenceMask(const mshadow::Tensor<gpu, 3, DType> &dst,
                         const mshadow::Tensor<gpu, 1, LType> &lengths, DType value) {
  using namespace mshadow;
  using namespace mshadow::cuda;
  dim3 dimBlock(kBaseThreadNum);
  dim3 dimGrid(dst.size(1));
  CheckLaunchParam(dimGrid, dimBlock, "SequenceMask");
  cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
  SequenceMaskKernel<kBaseThreadBits, DType><<<dimGrid, dimBlock, 0, stream>>>(dst, lengths, value);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SequenceMaskKernel);
}
#endif

}  // namespace mxnet_op

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_SEQUENCE_MASK_INL_H_
