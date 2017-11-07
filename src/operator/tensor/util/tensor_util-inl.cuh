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
 * \file tensor_util-inl.cuh
 * \brief commonly utilized tensor operator GPU kernels
 */
#ifndef MXNET_OPERATOR_TENSOR_UTIL_TENSOR_UTIL_INL_CUH_
#define MXNET_OPERATOR_TENSOR_UTIL_TENSOR_UTIL_INL_CUH_

#include <cub/cub.cuh>
#include <mxnet/base.h>
#include <mxnet/operator.h>

namespace mxnet {
namespace op {

/*!
 * \brief Thread kernel for marking non-zero rows of a tensor.
 * Parallelized by tensor rows: 1 thread/row
 */
struct MarkRspRowThreadKernel {
  /*!
   * \brief
   * \param tid         global thread id
   * \param row_flg     row flag array to mark non-zero rows
   * \param dns         dense matrix data
   * \param num_rows    number of rows (size of first dimension of tensor)
   * \param row_length  number of elements per row
   */
  template<typename DType>
  __device__ __forceinline__ static void Map(int tid,
                                             nnvm::dim_t* row_flg,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    if (tid < num_rows) {
      dim_t j = 0;
      dim_t offset = tid * row_length;
      for (; j < row_length; ++j) {
        if (dns[offset+j] != 0) {
          break;
        }
      }
      if (j < row_length) {
        row_flg[tid] = 1;  // mark as one for non-zero row
      } else {
        row_flg[tid] = 0;  // mark as zero for zero row
      }
    }
  }
};

/*!
 * \brief Warp kernel for marking non-zero rows of a tensor.
 * Parallelized by tensor rows: 1 warp/row
 */
struct MarkRspRowWarpKernel {
  template<typename DType>
  __device__ __forceinline__ static void Map(int tid,
                                             nnvm::dim_t* row_flg,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    typedef cub::WarpReduce<dim_t> WarpReduce;
    const dim_t warps_per_block = mshadow::cuda::kBaseThreadNum / 32;
    __shared__ typename WarpReduce::TempStorage temp_storage[warps_per_block];

    const dim_t warp_id   = tid / 32;          // global warp   id
    const dim_t warp_lane = threadIdx.x / 32;  // local  warp   id within thread block
    const dim_t lane      = tid & (32-1);      // local  thread id within warp

    if (warp_id < num_rows) {
      dim_t flg = 0;
      dim_t offset = warp_id * row_length;
      for (dim_t j = lane; j < row_length; j+=32) {
        if (dns[offset+j] != 0) {
          // avoid break: causes slower performance on sparse tensors (<20% density),
          // due to thread divergence
          flg++;
        }
      }
      dim_t aggr = WarpReduce(temp_storage[warp_lane]).Sum(flg);
      if (lane == 0) {
        if (aggr > 0) {
          row_flg[warp_id] = 1;  // mark as one for non-zero row
        } else {
          row_flg[warp_id] = 0;  // mark as zero for zero row
        }
      }
    }
  }
};

/*!
 * \brief Block kernel for marking non-zero rows of a tensor.
 * Parallelized by tensor rows: 1 threadBlock/row
 */
struct MarkRspRowBlockKernel {
  template<typename DType>
  __device__ __forceinline__ static void Map(int tid,
                                             nnvm::dim_t* row_flg,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    using mshadow::cuda::kBaseThreadNum;
    typedef cub::BlockReduce<dim_t, kBaseThreadNum> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    if (blockIdx.x < num_rows) {
      dim_t flg = 0;
      dim_t offset = blockIdx.x * row_length;
      for (dim_t j = threadIdx.x; j < row_length; j+=kBaseThreadNum) {
        if (dns[offset+j] != 0) {
          // avoid break: causes slower performance on sparse tensors (<20% density),
          // due to thread divergence
          flg++;
        }
      }
      dim_t aggr = BlockReduce(temp_storage).Sum(flg);
      if (threadIdx.x == 0) {
        if (aggr > 0) {
          row_flg[blockIdx.x] = 1;  // mark as one for non-zero row
        } else {
          row_flg[blockIdx.x] = 0;  // mark as zero for zero row
        }
      }
    }
  }
};

/*!
 * \brief GPU kernel to flag non-zero rows of an rsp tensor with 1.
 * Parallelized by tensor rows: 1 thread/row
 */
struct MarkRspRowFlgKernel {
  /*!
   * \brief
   * \param tid      global thread id
   * \param row_flg  array to flag storage indices of non-zero rows
   * \param row_idx  rsp tensor row index array storing indices of non-zero rows
   * \param nnr      rsp tensor number of non-zero rows (storage shape)
   */
  template<typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             IType* row_flg,
                                             const IType* row_idx,
                                             const nnvm::dim_t nnr) {
    if (tid < nnr) {
      row_flg[row_idx[tid]] = 1;
    }
  }
};

/*!
 * \brief GPU kernel to flag non-zero rows of an rsp tensor with indices.
 * Parallelized by matrix rows: 1 thread/row
 */
struct IndexRspRowFlgKernel {
  /*!
   * \brief
   * \param tid      global thread id
   * \param row_flg  array to flag storage indices of non-zero rows
   * \param row_idx  rsp tensor row index array storing indices of non-zero rows
   * \param nnr      rsp tensor number of non-zero rows (storage shape)
   */
  template<typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             RType* row_flg,
                                             const RType* row_idx,
                                             const nnvm::dim_t nnr) {
    if (tid < nnr) {
      row_flg[row_idx[tid]] = tid+1;
    }
  }
};

/*!
 * \brief GPU kernel for marking non-zero columns of a csr matrix.
 * Parallelized by matrix rows: 1 warp/row
 */
struct MarkCsrColWarpKernel {
  /*!
   * \brief
   * \param tid       global thread id
   * \param flg       flg array to mark non-zero columns
   * \param col_idx   csr matrix column indices
   * \param indptr    csr matrix row index pointer
   * \param num_rows  csr matrix number of rows
   * \param num_cols  csr matrix number of columns
   */
  template<typename CType, typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             nnvm::dim_t* flg,
                                             const CType* col_idx,
                                             const IType* indptr,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    typedef unsigned long long int uint64_cu;
    static_assert(sizeof(uint64_cu) == sizeof(nnvm::dim_t), "unexpected sizeof dim_t");

    const nnvm::dim_t warp_id = tid / 32;      // global warp   id
    const nnvm::dim_t lane    = tid & (32-1);  // local  thread id within warp

    if (warp_id < num_rows) {
      uint64_cu zero = 0;
      uint64_cu one = 1;
      for (IType j = indptr[warp_id]+lane; j < indptr[warp_id+1]; j+=32) {
        atomicCAS(reinterpret_cast<uint64_cu*>(flg+col_idx[j]), zero, one);
      }
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_UTIL_TENSOR_UTIL_INL_CUH_
