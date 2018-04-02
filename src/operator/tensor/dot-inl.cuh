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
 * \file dot-inl.cuh
 * \brief implementation of matrix dot op on GPU
 */
#ifndef MXNET_OPERATOR_TENSOR_DOT_INL_CUH_
#define MXNET_OPERATOR_TENSOR_DOT_INL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "./util/tensor_util-inl.h"
#include "./util/tensor_util-inl.cuh"

namespace mxnet {
namespace op {

/*!
 * \brief GPU scalar kernel of dot(csr, dns1) = dns2
 * Parallelization by output matrix elements: 1 thread/element
 */
template<int req>
struct DotCsrDnsDnsScalarKernel {
  /*!
   * \brief This function represents performing an inner product between a row of lhs
   * and a column of rhs and then assigning the value to out[tid].
   * \param tid         global thread id
   * \param out         output matrix data
   * \param data_l      csr matrix data
   * \param indptr_l    csr matrix row index pointer
   * \param col_idx_l   csr matrix column indices
   * \param data_r      dns1 matrix data
   * \param num_cols_r  dns1 matrix number of columns
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const nnvm::dim_t num_cols_r) {
    const nnvm::dim_t irow = tid / num_cols_r;  // row id of the lhs
    const nnvm::dim_t icol = tid % num_cols_r;  // col id of the rhs
    DType sum = 0;
    for (IType j = indptr_l[irow]; j < indptr_l[irow+1]; ++j) {
      const CType cur_col = col_idx_l[j];  // corresponding row id of the rhs
      sum += data_l[j] * data_r[cur_col*num_cols_r+icol];
    }
    KERNEL_ASSIGN(out[tid], req, sum);
  }
};

/*!
 * \brief GPU vector kernel of dot(csr, dns1) = dns2
 * Parallelization by output matrix elements: 1 warp/element
 */
template<int req>
struct DotCsrDnsDnsVectorKernel {
  /*!
   * \brief see DotCsrDnsDnsScalarKernel Map for documentation.
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const nnvm::dim_t num_cols_r) {
    using nnvm::dim_t;
    __shared__ volatile DType vals[mshadow::cuda::kBaseThreadNum];
    const dim_t warp_id = tid / 32;           // global warp id
    const dim_t lane = tid & (32-1);          // local thread id within warp
    const dim_t irow = warp_id / num_cols_r;  // lhs row that this warp computes
    const dim_t kcol = warp_id % num_cols_r;  // rhs column that this warp computes

    // Range of nnz elements in this row
    const dim_t low  = static_cast<dim_t>(indptr_l[irow]);
    const dim_t high = static_cast<dim_t>(indptr_l[irow+1]);

    // Compute running sum per thread
    DType sum = 0;
    for (dim_t j = low+lane; j < high; j+=32) {
      sum += data_l[j] * data_r[col_idx_l[j]*num_cols_r + kcol];
    }
    vals[threadIdx.x] = sum; __syncwarp();

    // Parallel reduction in shared memory
    if (lane < 16) {vals[threadIdx.x] += vals[threadIdx.x+16];} __syncwarp();
    if (lane <  8) {vals[threadIdx.x] += vals[threadIdx.x+ 8];} __syncwarp();
    if (lane <  4) {vals[threadIdx.x] += vals[threadIdx.x+ 4];} __syncwarp();
    if (lane <  2) {vals[threadIdx.x] += vals[threadIdx.x+ 2];} __syncwarp();
    if (lane <  1) {vals[threadIdx.x] += vals[threadIdx.x+ 1];} __syncwarp();

    if (lane == 0) {
      KERNEL_ASSIGN(out[irow*num_cols_r+kcol], req, vals[threadIdx.x]);
    }
  }
};

/*!
 * \brief GPU scalar kernel of dot(csr.T, dns1) = dns2
 * Parallelization by output matrix elements: 1 thread/element
 */
template<int req>
struct DotCsrTransDnsDnsScalarKernel {
  /*!
   * \brief This function represents performing an inner product between a column of lhs
   * and a column of rhs and then assigning the value to out[tid].
   * \param tid         global thread id
   * \param out         output matrix
   * \param data_l      csr matrix data
   * \param indptr_l    csr matrix row index pointer
   * \param col_idx_l   csr matrix column indices
   * \param data_r      dns1 matrix data of rhs
   * \param num_rows_l  csr matrix number of rows (= number of columns of csr.T)
   * \param num_cols_r  dns1 matrix number of columns
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const nnvm::dim_t num_rows_l,
                                             const nnvm::dim_t num_cols_r) {
    using nnvm::dim_t;
    const dim_t irow = tid / num_cols_r;  // col id of the lhs
    const dim_t icol = tid % num_cols_r;  // col id of the rhs
    DType sum = 0;

    // Each thread scans each column with binary search to find nnz elements in its row
    for (dim_t k = 0; k < num_rows_l; ++k) {
      const dim_t low = static_cast<dim_t>(indptr_l[k]);
      const dim_t high = static_cast<dim_t>(indptr_l[k+1]);
      if (low == high || irow < col_idx_l[low] || irow > col_idx_l[high-1]) continue;
      dim_t j = high, l = low, r = high - 1;
      while (l <= r) {
        dim_t m = l + (r - l) / 2;
        if (col_idx_l[m] == irow) {
          j = m; break;
        }
        if (col_idx_l[m] < irow) {
          l = m + 1;
        } else {
          r = m - 1;
        }
      }
      if (j < high) {
        sum += data_l[j] * data_r[k*num_cols_r+icol];
      }
    }
    KERNEL_ASSIGN(out[tid], req, sum);
  }
};

/*!
 * \brief GPU warp kernel of dot(csr.T, dns1) = dns2
 * Parallelization by columns: 1 warp computes one lhs column for one rhs column
 */
struct DotCsrTransDnsDnsWarpKernel {
  /*!
   * \brief see DotCsrTransDnsDnsScalarKernel Map for documentation.
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const nnvm::dim_t num_cols_r) {
    using nnvm::dim_t;
    const dim_t warp_id = tid / 32;           // global warp id
    const dim_t lane = tid & (32-1);          // local thread id within warp
    const dim_t icol = warp_id / num_cols_r;  // lhs column that this warp computes
    const dim_t kcol = warp_id % num_cols_r;  // rhs column that this warp computes

    // Compute range of nnz elements in this column
    const dim_t low  = static_cast<dim_t>(indptr_l[icol]);
    const dim_t high = static_cast<dim_t>(indptr_l[icol+1]);

    // Iterate through the nnz elements in this column
    for (dim_t j = low+lane; j < high; j+=32) {
      const dim_t irow = static_cast<dim_t>(col_idx_l[j]);
      const DType val = data_l[j]*data_r[icol*num_cols_r+kcol];
      atomicAdd(static_cast<DType *>(&(out[irow*num_cols_r+kcol])), val);
    }
  }
};

/*!
 * \brief GPU thread block kernel of dot(csr.T, dns1) = dns2
 * Parallelization by columns: 1 thread block computes one lhs column for all rhs columns
 */
struct DotCsrTransDnsDnsThreadBlockKernel {
  /*!
   * \brief see DotCsrTransDnsDnsScalarKernel Map for documentation.
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const nnvm::dim_t num_cols_r) {
    using nnvm::dim_t;
    const dim_t warps_per_block = blockDim.x / 32;  // number of warps in this thread block
    const dim_t warp_id = tid / 32;                 // global warp id
    const dim_t lane = tid & (32-1);                // local thread id within warp
    const dim_t icol = blockIdx.x;                  // lhs column that this thread block computes
    const dim_t kcol = warp_id % warps_per_block;   // rhs column where warp starts computing (offset)

    // Compute range of nnz elements in this lhs column
    const dim_t low  = static_cast<dim_t>(indptr_l[icol]);
    const dim_t high = static_cast<dim_t>(indptr_l[icol+1]);

    // Iterate through the nnz elements in this lhs column
    for (dim_t j = low+lane; j < high; j+=32) {
      const dim_t irow = static_cast<dim_t>(col_idx_l[j]);
      const DType datum_l = data_l[j];
      // Iterate over rhs columns that this warp computes
      for (dim_t k = kcol; k < num_cols_r; k+=warps_per_block) {
        const DType val = datum_l*data_r[icol*num_cols_r+k];
        atomicAdd(static_cast<DType *>(&(out[irow*num_cols_r+k])), val);
      }
    }
  }
};

/*!
 * \brief GPU warp block kernel of dot(csr.T, dns1) = dns2
 * Parallelization by columns: 1 warp computes one lhs column for all rhs columns
 */
struct DotCsrTransDnsDnsWarpBlockKernel {
  /*!
   * \brief see DotCsrTransDnsDnsScalarKernel Map for documentation.
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const nnvm::dim_t num_cols_r) {
    using nnvm::dim_t;
    const dim_t warp_id = tid / 32;   // global warp id
    const dim_t lane = tid & (32-1);  // local thread id within warp
    const dim_t icol = warp_id;       // lhs column that this warp computes

    // Compute range of nnz elements in this column
    const dim_t low  = static_cast<dim_t>(indptr_l[icol]);
    const dim_t high = static_cast<dim_t>(indptr_l[icol+1]);

    // Iterate through the nnz elements in lhs column
    for (dim_t j = low+lane; j < high; j+=32) {
      const dim_t irow = static_cast<dim_t>(col_idx_l[j]);
      const DType datum_l = data_l[j];
      // Iterate over all rhs columns
      for (dim_t k = 0; k < num_cols_r; k++) {
        const DType val = datum_l*data_r[icol*num_cols_r+k];
        atomicAdd(static_cast<DType *>(&(out[irow*num_cols_r+k])), val);
      }
    }
  }
};

/*!
 * \brief GPU warp kernel of dot(csr.T, dns) = rsp
 * Parallelization by columns: 1 warp computes one lhs column for one rhs column
 */
struct DotCsrTransDnsRspWarpKernel {
  /*!
   * \brief
   * \param tid              global thread id
   * \param out              output rsp matrix data
   * \param row_flg_sum_out  inclusive prefix sum array over 0/1 marked row flag array
   * \param data_l           csr matrix data
   * \param indptr_l         csr matrix row index pointer
   * \param col_idx_l        csr matrix column indices
   * \param data_r           dns matrix data
   * \param num_cols_r       dns matrix number of columns
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const nnvm::dim_t* row_flg_sum_out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const nnvm::dim_t num_cols_r) {
    using nnvm::dim_t;
    const dim_t warp_id = tid / 32;           // global warp id
    const dim_t lane = tid & (32-1);          // local thread id within warp
    const dim_t icol = warp_id / num_cols_r;  // lhs column that this warp computes
    const dim_t kcol = warp_id % num_cols_r;  // rhs column that this warp computes

    // Compute range of nnz elements in this column
    const dim_t low  = static_cast<dim_t>(indptr_l[icol]);
    const dim_t high = static_cast<dim_t>(indptr_l[icol+1]);

    // Iterate through the nnz elements in this column
    for (dim_t j = low+lane; j < high; j+=32) {
      const dim_t irow = static_cast<dim_t>(col_idx_l[j]);
      const dim_t rsp_row = row_flg_sum_out[irow]-1;
      const DType val = data_l[j]*data_r[icol*num_cols_r+kcol];
      atomicAdd(static_cast<DType *>(&(out[rsp_row*num_cols_r+kcol])), val);
    }
  }
};

/*!
 * \brief GPU Kernel of dot(csr.T, rsp1) = rsp2
 * Parallelization by rows: 1 thread/row
 * TODO: write a faster kernel optimized for GPU
 */
struct DotCsrTransRspRspByRowsKernel {
  /*!
   * \brief
   * \param tid           global thread id
   * \param out           output rsp matrix data
   * \param row_idx_out   output rsp matrix non-zero row indices
   * \param data_l        csr matrix data
   * \param indptr_l      csr matrix row index pointer
   * \param col_idx_l     csr matrix column indices
   * \param data_r        rsp1 matrix data
   * \param row_idx_r     rsp1 matrix non-zero row indices
   * \param num_cols_r    rsp1 matrix number of cols
   * \param nnr_r         rsp1 matrix number of non-zero rows
   * \param nnr_out       output rsp matrix number of non-zero rows
   */
  template<typename DType, typename IType, typename CType, typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const RType* row_idx_out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const RType* row_idx_r,
                                             const nnvm::dim_t num_cols_r,
                                             const nnvm::dim_t nnr_r,
                                             const nnvm::dim_t nnr_out) {
    using nnvm::dim_t;
    // This thread computes non-zero row 'tid' of the output matrix
    // The actual row id corresponding to the lhs row is row_idx_out[tid]
    if (tid < nnr_out) {
      const dim_t offset_out = tid * num_cols_r;
      // Iterate over rhs matrix rows (or, equivalently, lhs columns worthy taking a look at)
      for (dim_t i = 0; i < nnr_r; i++) {
        const RType j = row_idx_r[i];  // j is the actual rhs row id (= lhs column id)
        if (indptr_l[j] == indptr_l[j+1]) continue;
        const dim_t offset_r = i * num_cols_r;
        // Iterate over lhs column j to find possible non-zero value in this row
        // TODO: remove sequential search, this is a bottleneck
        for (IType k = indptr_l[j]; k < indptr_l[j+1]; k++) {
          const CType col_idx = col_idx_l[k];
          if (col_idx == row_idx_out[tid]) {
            for (dim_t l = 0; l < num_cols_r; l++) {
              out[offset_out+l] += data_l[k] * data_r[offset_r+l];
            }
          } else if (col_idx > row_idx_out[tid]) {
            break;
          }
        }
      }
    }
  }
};

/*!
 * \brief GPU Kernel of dot(csr, rsp) = dns
 * Parallelization by output elements: 1 thread/element
 */
struct DotCsrRspDnsScalarKernel {
  /*!
   * \brief
   * \param tid        global thread id
   * \param out        output dns matrix data
   * \param data_l     csr matrix data
   * \param indptr_l   csr matrix row index pointer
   * \param col_idx_l  csr matrix column indices
   * \param data_r     rsp matrix data
   * \param row_idx_r  rsp matrix non-zero row indices
   * \param row_flg_r  rsp matrix auxiliary array holding storage indices of non-zero rows
   * \param nnr_r      rsp matrix number of non-zero rows
   * \param num_rows   output dns matrix number of rows
   * \param num_cols   output dns matrix number of columns
   */
  template<typename DType, typename IType, typename CType, typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const RType* row_idx_r,
                                             const RType* row_flg_r,
                                             const nnvm::dim_t nnr_r,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    if (tid < num_rows*num_cols) {
      const dim_t i = static_cast<dim_t>(tid) / num_cols;  // i = row this thread computes
      const dim_t k = static_cast<dim_t>(tid) % num_cols;  // k = col this thread computes
      // Compute inner product of i-th row and k-th col
      DType sum = 0;
      for (IType j = indptr_l[i]; j < indptr_l[i+1]; j++) {
        const dim_t csr_col = col_idx_l[j];
        const dim_t rsp_row_idx = row_flg_r[csr_col];
        if (rsp_row_idx > 0) {
          sum += data_l[j] * data_r[(rsp_row_idx-1)*num_cols+k];
        }
      }
      if (sum != 0) {
        out[i*num_cols+k] += sum;
      }
    }
  }
};

/*!
 * \brief GPU Kernel to re-arrange nnz elements to csc order
 * Parallelization by output elements: 1 thread/row of csr
 */
struct CscDataIndicesKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             const DType* csr_data,
                                             const IType* csr_indices,
                                             const CType* csr_indptr,
                                             DType* csc_data,
                                             int* csc_indices,
                                             int* csc_indptr,
                                             int* workspace,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    if (tid < num_rows) {
      printf("%d, %d ", tid, csc_indptr[tid]);
      for (CType i = csr_indptr[tid]; i < csr_indptr[tid + 1]; ++i) {
        // target column
        IType target_col = csr_indices[i];
        int target_offset = atomicAdd(&workspace[target_col], 1);
        int new_pos = csc_indptr[target_col] + target_offset;
        csc_data[new_pos] = csr_data[i];
        csc_indices[new_pos] = tid;
      }
    }
  }
};

/*!
 * \brief GPU Kernel of getting count for every column
 * Parallelization by output elements: 1 thread/element
 */
struct CsrTransHistogramKernel {
  /*!
   * \brief
   * \param tid          global thread id
   * \param in_indices   csr matrix column indices
   * \param out_indptr   csr matrix row pointer
   * \param nnz          number of non-zero elements in csr
   */
  template<typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             const IType* in_indices,
                                             int* out_indptr,
                                             const nnvm::dim_t nnz) {
    if (tid < nnz) {
      atomicAdd(&out_indptr[in_indices[tid]], 1);
    }
  }
};

/*!
 * \brief GPU Kernel of dot(dns, csr.T) = dns
 * Parallelization by output elements: 1 thread/element
 */
struct DotDnsCsrTransDnsKernel {
  /*!
   * \brief
   * \param tid          global thread id
   * \param lhs_data     lhs dense matrix data
   * \param rhs_data     csr matrix data
   * \param rhs_indices  csr matrix column indices
   * \param rhs_indptr   csr matrix row pointer
   * \param out          output matrix data
   * \param lhs_num_cols lhs dns matrix number of columns
   * \param out_num_rows output dns matrix number of rows
   * \param out_num_cols output dns matrix number of columns
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             const DType* lhs_data,
                                             const DType* rhs_data,
                                             const IType* rhs_indices,
                                             const CType* rhs_indptr,
                                             DType* out,
                                             const nnvm::dim_t lhs_num_cols,
                                             const nnvm::dim_t out_num_rows,
                                             const nnvm::dim_t out_num_cols) {
    using nnvm::dim_t;
    if (tid < out_num_rows*out_num_cols) {
      const dim_t i = static_cast<dim_t>(tid) / out_num_cols;  // i = row this thread computes
      const dim_t k = static_cast<dim_t>(tid) % out_num_cols;  // k = col this thread computes
      // Compute inner product of i-th row and k-th col
      DType sum = 0;
      for (CType col_id = rhs_indptr[k]; col_id < rhs_indptr[k + 1]; ++col_id) {
        sum += lhs_data[i * lhs_num_cols + rhs_indices[col_id]] * rhs_data[col_id];
      }
      out[tid] = sum;
    }
  }
};

/*!
 * \brief GPU Impl of dot(csr, dns1) = dns2 and dot(csr.T, dns1) = dns2
 */
inline void DotCsrDnsDnsImpl(const OpContext& ctx,
                             const gpu& gpu_dev,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (!lhs.storage_initialized()) {
    Fill(s, *ret, req, 0);
    return;
  }

  using mshadow::cuda::kBaseThreadNum;
  using mxnet_op::Kernel;
  using mxnet_op::set_zero;
  using nnvm::dim_t;

  const dim_t num_rows_l = lhs.shape()[0];
  const dim_t num_cols_r = rhs.shape_[1];
  const dim_t threads_per_warp = mxnet_op::cuda_get_device_prop().warpSize;
  const dim_t threads_per_block = kBaseThreadNum;
  dim_t num_threads;
  // TODO: remove kernel dependency on warpSize=32
  if (threads_per_warp != 32) {
    LOG(FATAL) << "DotCsrDnsDnsImpl GPU kernels expect warpSize=32";
  }

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob& data_r = rhs;
  const TBlob data_out = *ret;

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        if (kWriteTo == req) {
          num_threads = data_out.Size();
          Kernel<set_zero, gpu>::Launch(s, num_threads, data_out.dptr<DType>());
        }
        if (trans_lhs) {
          // Different kernel versions are optimized for different matrix instances
          // TODO: switch between kernel versions depending on input
          // (1) 'Scalar kernel'       (one thread       computing one output element                )
          // (2) 'Warp kernel'         (one warp         computing one lhs column for one rhs column )
          // (3) 'Thread block kernel' (one thread block computing one lhs column for all rhs columns)
          // (4) 'Warp block kernel'   (one warp         computing one lhs column for all rhs columns)
          const int kernel_version = 0;
          switch (kernel_version) {
            case 1:
              num_threads = data_out.Size();
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                Kernel<DotCsrTransDnsDnsScalarKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_rows_l, num_cols_r);
              });
              break;
            case 2:
              num_threads = threads_per_warp * num_rows_l * num_cols_r;
              Kernel<DotCsrTransDnsDnsWarpKernel, gpu>::Launch(s, num_threads,
                  data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                  col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              break;
            case 3:
              num_threads = threads_per_block * num_rows_l;
              Kernel<DotCsrTransDnsDnsThreadBlockKernel, gpu>::Launch(s, num_threads,
                  data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                  col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              break;
            case 4:
              num_threads = threads_per_warp * num_rows_l;
              Kernel<DotCsrTransDnsDnsWarpBlockKernel, gpu>::Launch(s, num_threads,
                  data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                  col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              break;
            default:
              num_threads = threads_per_warp * num_rows_l * num_cols_r;
              Kernel<DotCsrTransDnsDnsWarpKernel, gpu>::Launch(s, num_threads,
                  data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                  col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              break;
          }
        } else {
          // Different kernel versions are optimized for different matrix instances
          // (1) 'Scalar kernel' (one thread computing one output element)
          // (2) 'Vector kernel' (one warp   computing one output element)
          const int kernel_version = 0;
          switch (kernel_version) {
            case 1:
              num_threads = data_out.Size();
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                Kernel<DotCsrDnsDnsScalarKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            case 2:
              num_threads = threads_per_warp * num_rows_l * num_cols_r;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                Kernel<DotCsrDnsDnsVectorKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            default:
              if (num_cols_r > 4) {
                num_threads = data_out.Size();
                MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                  Kernel<DotCsrDnsDnsScalarKernel<ReqType>, gpu>::Launch(s, num_threads,
                      data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                      col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
                });
              } else {
                num_threads = threads_per_warp * num_rows_l * num_cols_r;
                MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                  Kernel<DotCsrDnsDnsVectorKernel<ReqType>, gpu>::Launch(s, num_threads,
                      data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                      col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
                });
              }
              break;
          }
        }
      });
    });
  });
}

/*!
 * \brief GPU Impl of dot(csr, dns) = rsp and dot(csr.T, dns) = rsp
 */
inline void DotCsrDnsRspImpl(const OpContext& ctx,
                             const gpu& gpu_dev,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(ret->storage_type(), kRowSparseStorage);
  CHECK_EQ(req, kWriteTo);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (!lhs.storage_initialized()) {
    FillZerosRspImpl(s, *ret);
    return;
  }

  using mshadow::Shape1;
  using mxnet_op::Kernel;
  using mxnet_op::set_zero;
  using nnvm::dim_t;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob& data_r = rhs;

  const dim_t num_rows_l = lhs.shape()[0];
  const dim_t num_cols_l = lhs.shape()[1];
  const dim_t num_cols_r = rhs.shape_[1];
  const dim_t threads_per_warp = mxnet_op::cuda_get_device_prop().warpSize;
  dim_t num_threads;
  // TODO: remove kernel dependency on warpSize=32
  if (threads_per_warp != 32) {
    LOG(FATAL) << "DotCsrDnsRspImpl GPU kernels expect warpSize=32";
  }

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        if (trans_lhs) {
          // Compute number of non-zero rows (nnr) of output matrix
          // - alloc temp storage for row_flg array and for cub's prefix sum
          // - mark non-zero columns of csr matrix in row_flg
          // - compute inclusive prefix sum over marked array
          // - copy last value (nnr_out) from device to host
          dim_t* row_flg_out = NULL;
          void* d_temp_storage = NULL;
          size_t temp_storage_bytes = 0;
          cub::DeviceScan::InclusiveSum(d_temp_storage,
                                        temp_storage_bytes,
                                        row_flg_out,
                                        row_flg_out,
                                        num_cols_l,
                                        mshadow::Stream<gpu>::GetStream(s));
          mshadow::Tensor<gpu, 1, char> workspace = ctx.requested[0]
              .get_space_typed<gpu, 1, char>(Shape1(num_cols_l * sizeof(dim_t) +
                                                    temp_storage_bytes), s);
          row_flg_out = reinterpret_cast<dim_t*>(workspace.dptr_);
          d_temp_storage = workspace.dptr_ + num_cols_l*sizeof(dim_t);
          num_threads = num_cols_l;
          Kernel<set_zero, gpu>::Launch(s, num_threads, row_flg_out);
          num_threads = num_rows_l * threads_per_warp;
          Kernel<MarkCsrColWarpKernel, gpu>::Launch(s, num_threads,
              row_flg_out, col_idx_l.dptr<CType>(), indptr_l.dptr<IType>(),
              num_rows_l, num_cols_l);
          cub::DeviceScan::InclusiveSum(d_temp_storage,
                                        temp_storage_bytes,
                                        row_flg_out,
                                        row_flg_out,
                                        num_cols_l,
                                        mshadow::Stream<gpu>::GetStream(s));
          dim_t nnr_out = 0;
          CUDA_CALL(cudaMemcpy(&nnr_out, &row_flg_out[num_cols_l-1], sizeof(dim_t),
                               cudaMemcpyDeviceToHost));
          if (0 == nnr_out) {
            FillZerosRspImpl(s, *ret);
            return;
          }

          // Allocate output matrix space
          ret->CheckAndAlloc({Shape1(nnr_out)});
          const TBlob data_out_blob = ret->data();
          const TBlob row_idx_out_blob = ret->aux_data(rowsparse::kIdx);
          MSHADOW_IDX_TYPE_SWITCH(row_idx_out_blob.type_flag_, RType, {  // row idx type
            DType* data_out = data_out_blob.dptr<DType>();
            RType* row_idx_out = row_idx_out_blob.dptr<RType>();
            num_threads = nnr_out * num_cols_r;
            Kernel<set_zero, gpu>::Launch(s, num_threads, data_out);
            num_threads = nnr_out;
            Kernel<set_zero, gpu>::Launch(s, num_threads, row_idx_out);

            // Fill row_idx array of output matrix, using the row_flg values
            num_threads = num_cols_l;
            Kernel<FillRspRowIdxKernel, gpu>::Launch(s, num_threads,
                row_idx_out, row_flg_out, num_cols_l);

            // Perform matrix-matrix multiply
            num_threads = threads_per_warp * num_rows_l * num_cols_r;
            Kernel<DotCsrTransDnsRspWarpKernel, gpu>::Launch(s, num_threads,
                data_out, row_flg_out,
                data_l.dptr<DType>(), indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(),
                data_r.dptr<DType>(), num_cols_r);
          });
        } else {
          LOG(FATAL) << "DotCsrDnsRspImpl has not implemented dot(csr, dns) = rsp yet.";
        }
      });
    });
  });
}

/*!
 * \brief GPU Impl of dot(csr, rsp1) = rsp2 and dot(csr.T, rsp1) = rsp2
 * TODO: Optimize for GPU; this is a baseline implementation providing
 *       the operator functionality, it is not yet fully optimized for GPU.
 */
inline void DotCsrRspRspImpl(const OpContext& ctx,
                             const gpu& gpu_dev,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  if (kNullOp == req) return;
  // Reuse dot(csr, dns) implementation if rhs rsp matrix is in fact dense
  if (rhs.storage_shape()[0] == rhs.shape()[0]) {
    DotCsrDnsRspImpl(ctx, gpu_dev, lhs, rhs.data(), req, trans_lhs, ret);
    return;
  }
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(rhs.storage_type(), kRowSparseStorage);
  CHECK_EQ(ret->storage_type(), kRowSparseStorage);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (!lhs.storage_initialized() || !rhs.storage_initialized()) {
    FillZerosRspImpl(s, *ret);
    return;
  }
  CHECK_EQ(req, kWriteTo);

  using mshadow::Shape1;
  using mxnet_op::Kernel;
  using mxnet_op::set_zero;
  using nnvm::dim_t;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob data_r = rhs.data();
  const TBlob row_idx_r = rhs.aux_data(rowsparse::kIdx);

  const dim_t num_rows_l = lhs.shape()[0];
  const dim_t num_cols_l = lhs.shape()[1];
  const dim_t num_cols_r = rhs.shape()[1];
  const dim_t nnr_r = rhs.storage_shape()[0];
  const dim_t threads_per_warp = mxnet_op::cuda_get_device_prop().warpSize;
  dim_t num_threads;
  // TODO: remove kernel dependency on warpSize=32
  if (threads_per_warp != 32) {
    LOG(FATAL) << "DotCsrRspRspImpl GPU kernels expect warpSize=32";
  }

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(row_idx_r.type_flag_, RType, {  // row idx type
          if (trans_lhs) {
            // Compute number of non-zero rows (nnr) of output matrix
            // - alloc temp storage for row_flg array and for cub's prefix sum
            // - mark non-zero columns of csr matrix in row_flg
            // - compute inclusive prefix sum over marked array
            // - copy last value (nnr_out) from device to host
            dim_t* row_flg_out = NULL;
            void* d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(d_temp_storage,
                                          temp_storage_bytes,
                                          row_flg_out,
                                          row_flg_out,
                                          num_cols_l,
                                          mshadow::Stream<gpu>::GetStream(s));
            mshadow::Tensor<gpu, 1, char> workspace = ctx.requested[0]
                .get_space_typed<gpu, 1, char>(Shape1(num_cols_l * sizeof(dim_t) +
                                                      temp_storage_bytes), s);
            row_flg_out = reinterpret_cast<dim_t*>(workspace.dptr_);
            d_temp_storage = workspace.dptr_ + num_cols_l*sizeof(dim_t);
            num_threads = num_cols_l;
            Kernel<set_zero, gpu>::Launch(s, num_threads, row_flg_out);
            num_threads = num_rows_l * threads_per_warp;
            Kernel<MarkCsrColWarpKernel, gpu>::Launch(s, num_threads,
                row_flg_out, col_idx_l.dptr<CType>(), indptr_l.dptr<IType>(),
                num_rows_l, num_cols_l);
            cub::DeviceScan::InclusiveSum(d_temp_storage,
                                          temp_storage_bytes,
                                          row_flg_out,
                                          row_flg_out,
                                          num_cols_l,
                                          mshadow::Stream<gpu>::GetStream(s));
            dim_t nnr_out = 0;
            CUDA_CALL(cudaMemcpy(&nnr_out, &row_flg_out[num_cols_l-1], sizeof(dim_t),
                                 cudaMemcpyDeviceToHost));
            if (0 == nnr_out) {
              FillZerosRspImpl(s, *ret);
              return;
            }

            // Allocate output matrix space
            ret->CheckAndAlloc({mshadow::Shape1(nnr_out)});
            const TBlob data_out_blob = ret->data();
            const TBlob row_idx_out_blob = ret->aux_data(rowsparse::kIdx);
            DType* data_out = data_out_blob.dptr<DType>();
            RType* row_idx_out = row_idx_out_blob.dptr<RType>();
            num_threads = nnr_out * num_cols_r;
            Kernel<set_zero, gpu>::Launch(s, num_threads, data_out);
            num_threads = nnr_out;
            Kernel<set_zero, gpu>::Launch(s, num_threads, row_idx_out);

            // Fill row_idx array of output matrix, using the row_flg values
            num_threads = num_cols_l;
            Kernel<FillRspRowIdxKernel, gpu>::Launch(s, num_threads,
                row_idx_out, row_flg_out, num_cols_l);

            // Perform matrix-matrix multiply
            num_threads = nnr_out;
            Kernel<DotCsrTransRspRspByRowsKernel, gpu>::Launch(s, num_threads,
                data_out, row_idx_out,
                data_l.dptr<DType>(), indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(),
                data_r.dptr<DType>(), row_idx_r.dptr<RType>(),
                num_cols_r, nnr_r, nnr_out);
          } else {
            LOG(FATAL) << "DotCsrRspRspImpl has not implemented dot(csr, rsp1) = rsp2 yet.";
          }
        });
      });
    });
  });
}

/*!
 * \brief GPU Impl of dot(csr, rsp) = dns and dot(csr.T, rsp) = dns
 */
inline void DotCsrRspDnsImpl(const OpContext& ctx,
                             const gpu& gpu_dev,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  // Reuse dot(csr, dns) implementation if rhs rsp matrix is in fact dense
  if (rhs.storage_shape()[0] == rhs.shape()[0]) {
    DotCsrDnsDnsImpl(ctx, gpu_dev, lhs, rhs.data(), req, trans_lhs, ret);
    return;
  }
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(rhs.storage_type(), kRowSparseStorage);

  using mxnet_op::Kernel;
  using mxnet_op::set_zero;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (!lhs.storage_initialized() || !rhs.storage_initialized()) {
    if (kWriteTo == req) {
      MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {  // data type
        Kernel<set_zero, gpu>::Launch(s, ret->Size(), ret->dptr<DType>());
      });
    }
    return;
  }

  using nnvm::dim_t;
  const dim_t num_rows = ret->shape_[0];
  const dim_t num_cols = ret->shape_[1];
  const dim_t nnr_r = rhs.storage_shape()[0];
  dim_t num_threads;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob data_r = rhs.data();
  const TBlob row_idx_r = rhs.aux_data(rowsparse::kIdx);

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(row_idx_r.type_flag_, RType, {  // row idx type
          if (kWriteTo == req) {
            num_threads = num_rows*num_cols;
            Kernel<set_zero, gpu>::Launch(s, num_threads, ret->dptr<DType>());
          }
          if (trans_lhs) {
            LOG(FATAL) << "DotCsrRspDnsImpl has not implemented dot(csr.T, rsp) = dns yet.";
          } else {
            // TODO: Consider implementing a vector kernel for SpMV (similar to DotCsrDnsDns)
            // Alloc temp storage for row_flg array
            RType* row_flg_r = ctx.requested[0]
                .get_space_typed<gpu, 1, RType>(mshadow::Shape1(rhs.shape()[0]), s).dptr_;
            num_threads = rhs.shape()[0];
            Kernel<set_zero, gpu>::Launch(s, num_threads, row_flg_r);
            // Set row_flg index array
            num_threads = nnr_r;
            Kernel<IndexRspRowFlgKernel, gpu>::Launch(s, num_threads,
                row_flg_r, row_idx_r.dptr<RType>(), nnr_r);
            // Perform sparse matrix-matrix multiply
            num_threads = num_rows*num_cols;
            Kernel<DotCsrRspDnsScalarKernel, gpu>::Launch(s, num_threads,
                ret->dptr<DType>(),
                data_l.dptr<DType>(), indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(),
                data_r.dptr<DType>(), row_idx_r.dptr<RType>(), row_flg_r, rhs.storage_shape()[0],
                num_rows, num_cols);
          }
        });
      });
    });
  });
}

/*
 * \brief GPU Impl of dot(dns, csr) = csr
 */
template<typename gpu>
inline void DotDnsCsrCsrImpl(const OpContext& ctx,
                             const TBlob& lhs, const NDArray& rhs,
                             const OpReqType req, NDArray* ret) {
  LOG(FATAL) << "dot(dense, csr) = csr is not implemented on GPU";
}

/*
 * \brief GPU Impl of dot(dns, csr) = dns and dot(dns, csr.T) = dns
 */
template<typename gpu>
inline void DotDnsCsrDnsImpl(const OpContext& ctx,
                             const TBlob& dns, const NDArray& rhs,
                             const OpReqType req, NDArray* ret,
                             const bool transpose_b) {
  CHECK_EQ(req, kWriteTo);
  CHECK_EQ(rhs.storage_type(), kCSRStorage);

  using namespace mshadow;
  using namespace mshadow::expr;
  using nnvm::dim_t;

  /* Initialize data structures */
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  TBlob csr_data = rhs.data();
  TBlob csr_indices = rhs.aux_data(csr::kIdx);
  TBlob csr_indptr = rhs.aux_data(csr::kIndPtr);
  if (!rhs.storage_initialized()) {
    FillZerosCsrImpl(s, *ret);
    return;
  }

  // if dot(dense, csr) = dns, transform to csc first
  if (!transpose_b) {
    // LOG(FATAL) << "dot(dns, csr) = dns not implemented yet";
    const nnvm::dim_t csr_rows = rhs.shape()[0];
    const nnvm::dim_t csr_cols = rhs.shape()[1];
    const nnvm::dim_t dns_rows = dns.shape_[0];
    const nnvm::dim_t nnz = rhs.storage_shape().Size();

    MSHADOW_SGL_DBL_TYPE_SWITCH(csr_data.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(csr_indices.type_flag_, IType, {
        MSHADOW_IDX_TYPE_SWITCH(csr_indptr.type_flag_, CType, {
          DType* csc_data_ptr = NULL;
          int* csc_indices_ptr = NULL;
          int* csc_indptr_ptr = NULL;
          int* col_counters = NULL;
          void* temp_storage = NULL;
          size_t temp_storage_bytes = 0;
          CType out_num_rows = ret->shape()[0];
          CType out_num_cols = ret->shape()[1];
          // Get necessary temporary storage amount
          cub::DeviceScan::ExclusiveSum(NULL,
                                        temp_storage_bytes,
                                        csc_indices_ptr,
                                        csc_indices_ptr,
                                        csr_cols+1,
                                        Stream<gpu>::GetStream(s));
          Tensor<gpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<gpu, 1, char>(
              Shape1(nnz*sizeof(DType) + nnz*sizeof(int) +
                     (csr_cols + 1)*sizeof(int) +
                     (csr_cols + 1)*sizeof(int) +
                     temp_storage_bytes),
              s);
          csc_data_ptr = reinterpret_cast<DType*>(workspace.dptr_);
          csc_indices_ptr = reinterpret_cast<int*>(workspace.dptr_ + nnz*sizeof(DType));
          csc_indptr_ptr = reinterpret_cast<int*>(workspace.dptr_ + nnz*sizeof(DType) +
                                               nnz*sizeof(int));
          col_counters = reinterpret_cast<int*>(workspace.dptr_ + nnz*sizeof(DType) +
                                           nnz*sizeof(int) + (csr_cols+1)*sizeof(int));
          temp_storage = reinterpret_cast<void*>(workspace.dptr_ + nnz*sizeof(DType) +
                                            nnz*sizeof(int) + (csr_cols+1)*sizeof(int) +
                                            (csr_cols + 1)*sizeof(int));
          mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(
            s, dns_rows*csr_cols, ret->data().dptr<DType>());
          // Reset values for indptr, ready for histogramming
          mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(
            s, csr_cols + 1, csc_indptr_ptr);
          // Histogramming on col id
          mxnet_op::Kernel<CsrTransHistogramKernel, gpu>::Launch(
            s, nnz, csr_indices.dptr<IType>(), csc_indptr_ptr, nnz);
          cub::DeviceScan::ExclusiveSum(temp_storage,
                                        temp_storage_bytes,
                                        csc_indptr_ptr,
                                        csc_indptr_ptr,
                                        csr_cols+1,
                                        Stream<gpu>::GetStream(s));
          // Reset values for col_counter, ready for the final transform
          mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(
            s, csr_cols+1, col_counters);
          // Transform to CSC
          mxnet_op::Kernel<CscDataIndicesKernel, gpu>::Launch(
            s, csr_rows, csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
            csr_indptr.dptr<CType>(), csc_data_ptr, csc_indices_ptr,
            csc_indptr_ptr, col_counters, csr_rows, csr_cols);

          mxnet_op::Kernel<DotDnsCsrTransDnsKernel, gpu>::Launch(
            s, out_num_rows * out_num_cols, dns.dptr<DType>(),
            csc_data_ptr, csc_indices_ptr, csc_indptr_ptr,
            ret->data().dptr<DType>(), dns.shape_[1],
            out_num_rows, out_num_cols);
        });
      });
    });
  } else {
    MSHADOW_SGL_DBL_TYPE_SWITCH(csr_data.type_flag_, DType, {     // data type
      MSHADOW_IDX_TYPE_SWITCH(csr_indices.type_flag_, IType, {     // indptr type
        MSHADOW_IDX_TYPE_SWITCH(csr_indptr.type_flag_, CType, {  // colidx type
          CType out_num_rows = ret->shape()[0];
          CType out_num_cols = ret->shape()[1];
          mxnet_op::Kernel<DotDnsCsrTransDnsKernel, gpu>::Launch(
            s, out_num_rows * out_num_cols, dns.dptr<DType>(),
            csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
            csr_indptr.dptr<CType>(), ret->data().dptr<DType>(),
            dns.shape_[1], out_num_rows, out_num_cols);
        });
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_DOT_INL_CUH_
