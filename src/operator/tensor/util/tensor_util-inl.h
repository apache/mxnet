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
 *  Copyright (c) 2017 by Contributors
 * \file tensor_util-inl.h
 * \brief commonly utilized tensor operator CPU kernels
 */
#ifndef MXNET_OPERATOR_TENSOR_UTIL_TENSOR_UTIL_INL_H_
#define MXNET_OPERATOR_TENSOR_UTIL_TENSOR_UTIL_INL_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>

namespace mxnet {
namespace op {

/*!
 * \brief kernel to flag indices that appear in row_idx array with 1.
 */
struct MarkRowFlgKernel {
  /*!
   * \brief
   * \param tid      global thread id
   * \param row_flg  flag array for indices
   * \param row_idx  row index array storing indices of rows
   */
  template<typename IType, typename DType>
  MSHADOW_XINLINE static void Map(int tid,
                                  DType* row_flg,
                                  const IType* row_idx) {
    nnvm::dim_t idx = static_cast<nnvm::dim_t>(row_idx[tid]);
    row_flg[idx] = 1;
  }
};

/*!
 * \brief kernel for filling the row index array of an rsp tensor.
 * Parallelized by tensor rows: 1 thread/row
 */
struct FillRspRowIdxKernel {
  /*!
   * \brief
   * \param tid          global thread id
   * \param row_idx      row index array to store indices of non-zero rows
   * \param row_flg_sum  inclusive prefix sum array over 0/1 marked row flag array
   * \param num_rows     rsp tensor number of rows (shape)
   */
  template<typename RType>
  MSHADOW_XINLINE static void Map(int tid,
                                  RType* row_idx,
                                  const nnvm::dim_t* row_flg_sum,
                                  const nnvm::dim_t num_rows) {
    if (tid < num_rows) {
      nnvm::dim_t prev = (tid == 0) ? 0 : row_flg_sum[tid-1];
      if (row_flg_sum[tid] > prev) {
        row_idx[prev] = static_cast<RType>(tid);
      }
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_UTIL_TENSOR_UTIL_INL_H_
