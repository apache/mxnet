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
 * \file ndarray_function-inl.cuh
 * \brief Implementation of ndarray function kernels on GPU
 */
#ifndef MXNET_NDARRAY_NDARRAY_FUNCTION_INL_CUH_
#define MXNET_NDARRAY_NDARRAY_FUNCTION_INL_CUH_

namespace mxnet {
namespace ndarray {

/*!
 * \brief GPU kernel to perform RSP tensor addition: out += in
 * Parallelization by non-zero input elements: 1 thread/element
 */
struct ElementWiseRspAdditionKernel {
  /*!
   * \brief
   * \param tid         global thread id
   * \param data_out    rsp output data
   * \param row_flg     rsp output inclusive prefix sum array over non-zero marked rows
   * \param row_idx_in  rsp input non-zero row indices
   * \param data_in     rsp input data
   * \param nnr_in      rsp input number of non-zero rows
   * \param row_length  rsp input and output number of elements per row
   */
  template<typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* data_out,
                                             const IType* row_flg,
                                             const IType* row_idx_in,
                                             const DType* data_in,
                                             const nnvm::dim_t nnr_in,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    if (tid < nnr_in * row_length) {
      dim_t in_row = tid / row_length;
      dim_t in_col = tid % row_length;
      dim_t out_row = row_flg[row_idx_in[in_row]] - 1;
      dim_t out_idx = out_row * row_length + in_col;
      data_out[out_idx] += data_in[tid];
    }
  }
};

}  // namespace ndarray
}  // namespace mxnet

#endif  // MXNET_NDARRAY_NDARRAY_FUNCTION_INL_CUH_
