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

namespace mxnet {
namespace op {
using mshadow::cuda::kBaseThreadNum;

/*!
 * \brief Scalar kernel of dot(csr, dns1) = dns2
 * Parallelization by output matrix elements: 1 thread/element
 */
template<int req>
struct DotCsrDnsDnsScalarKernel {
  /*!
   * \brief This function represents performing an inner product between a row of lhs
   * and a column of rhs and then assigning the value to out[i].
   * \param i i-th element in out 1D view
   * \param out output matrix
   * \param data_l csr values of lhs
   * \param indptr_l csr indptr of lhs
   * \param col_idx_l csr col_idx of lhs
   * \param data_r dense data of rhs
   * \param num_cols number of columns of output
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* data_l, const IType* indptr_l,
                                  const CType* col_idx_l, const DType* data_r,
                                  const int num_cols) {
    const int irow = i / num_cols;  // row id of the lhs
    const int icol = i % num_cols;  // col id of the rhs
    DType sum = 0;
    for (IType j = indptr_l[irow]; j < indptr_l[irow+1]; ++j) {
      const CType cur_col = col_idx_l[j];  // corresponding row id of the rhs
      sum += data_l[j] * data_r[cur_col*num_cols+icol];
    }
    KERNEL_ASSIGN(out[i], req, sum);
  }
};

/*!
 * \brief Vector kernel of dot(csr, dns1) = dns2
 * Parallelization by output matrix elements: 1 warp/element
 */
template<int req>
struct DotCsrDnsDnsVectorKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid, DType* out, const DType* data_l, const IType* indptr_l,
                                             const CType* col_idx_l, const DType* data_r,
                                             const int num_cols_r) {
    __shared__ volatile DType vals[kBaseThreadNum];

    const int warp_id = tid / 32;           // global warp id
    const int lane = tid & (32-1);          // local thread id within warp
    const int irow = warp_id / num_cols_r;  // lhs row that this warp computes
    const int kcol = warp_id % num_cols_r;  // rhs column that this warp computes

    // Range of nnz elements in this row
    const int low  = static_cast<int>(indptr_l[irow]);
    const int high = static_cast<int>(indptr_l[irow+1]);

    // Compute running sum per thread
    DType sum = 0;
    for (int j = low+lane; j < high; j+=32) {
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
 * \brief Scalar kernel of dot(csr.T(), dns1) = dns2
 * Parallelization by output matrix elements: 1 thread/element
 */
template<int req>
struct DotCsrTransDnsDnsScalarKernel {
  /*!
   * \brief This function represents performing an inner product between a column of lhs
   * and a column of rhs and then assigning the value to out[i].
   * \param i i-th element in out 1D view
   * \param out output matrix
   * \param data_l csr values of lhs
   * \param indptr_l csr indptr of lhs
   * \param col_idx_l csr col_idx of lhs
   * \param data_r dense data of rhs
   * \param num_rows_l number of rows of lhs
   * \param num_cols number of columns of outputs
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* data_l, const IType* indptr_l,
                                  const CType* col_idx_l, const DType* data_r, const int num_rows_l,
                                  const int num_cols) {
    const int irow = i / num_cols;  // col id of the lhs
    const int icol = i % num_cols;  // col id of the rhs
    DType sum = 0;

    // Each thread scans each column with binary search to find nnz elements in its row
    for (int k = 0; k < num_rows_l; ++k) {
      const IType low = indptr_l[k];
      const IType high = indptr_l[k+1];
      if (low == high || irow < col_idx_l[low] || irow > col_idx_l[high-1]) continue;
      int j = -1, l = low, r = high - 1;
      while (l <= r) {
        int m = l + (r - l) / 2;
        if (col_idx_l[m] == irow) {
          j = m; break;
        }
        if (col_idx_l[m] < irow) {
          l = m + 1;
        } else {
          r = m - 1;
        }
      }
      if (j >= 0) {
        sum += data_l[j] * data_r[k*num_cols+icol];
      }
    }
    KERNEL_ASSIGN(out[i], req, sum);
  }
};

/*!
 * \brief Warp kernel of dot(csr.T(), dns1) = dns2
 * Parallelization by columns: 1 warp computes one lhs column for one rhs column
 */
template<int req>
struct DotCsrTransDnsDnsWarpKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid, DType* out, const DType* data_l, const IType* indptr_l,
                                             const CType* col_idx_l, const DType* data_r,
                                             const int num_cols_r) {
    const int warp_id = tid / 32;           // global warp id
    const int lane = tid & (32-1);          // local thread id within warp
    const int icol = warp_id / num_cols_r;  // lhs column that this warp computes
    const int kcol = warp_id % num_cols_r;  // rhs column that this warp computes

    // Compute range of nnz elements in this column
    const int low  = static_cast<int>(indptr_l[icol]);
    const int high = static_cast<int>(indptr_l[icol+1]);

    // Iterate through the nnz elements in this column
    for (int j = low+lane; j < high; j+=32) {
      const int irow = static_cast<int>(col_idx_l[j]);
      const DType val = data_l[j]*data_r[icol*num_cols_r+kcol];
      atomicAdd(static_cast<DType *>(&(out[irow*num_cols_r+kcol])), val);
    }
  }
};

/*!
 * \brief Thread block kernel of dot(csr.T(), dns1) = dns2
 * Parallelization by columns: 1 thread block computes one lhs column for all rhs columns
 */
template<int req>
struct DotCsrTransDnsDnsThreadBlockKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid, DType* out, const DType* data_l, const IType* indptr_l,
                                             const CType* col_idx_l, const DType* data_r,
                                             const int num_cols_r) {
    const int warps_per_block = blockDim.x / 32;  // number of warps in this thread block
    const int warp_id = tid / 32;                 // global warp id
    const int lane = tid & (32-1);                // local thread id within warp
    const int icol = blockIdx.x;                  // lhs column that this thread block computes
    const int kcol = warp_id % warps_per_block;   // rhs column where warp starts computing (offset)

    // Compute range of nnz elements in this lhs column
    const int low  = static_cast<int>(indptr_l[icol]);
    const int high = static_cast<int>(indptr_l[icol+1]);

    // Iterate through the nnz elements in this lhs column
    for (int j = low+lane; j < high; j+=32) {
      const int irow = static_cast<int>(col_idx_l[j]);
      const DType datum_l = data_l[j];
      // Iterate over rhs columns that this warp computes
      for (int k = kcol; k < num_cols_r; k+=warps_per_block) {
        const DType val = datum_l*data_r[icol*num_cols_r+k];
        atomicAdd(static_cast<DType *>(&(out[irow*num_cols_r+k])), val);
      }
    }
  }
};

/*!
 * \brief Warp block kernel of dot(csr.T(), dns1) = dns2
 * Parallelization by columns: 1 warp computes one lhs column for all rhs columns
 */
template<int req>
struct DotCsrTransDnsDnsWarpBlockKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid, DType* out, const DType* data_l, const IType* indptr_l,
                                             const CType* col_idx_l, const DType* data_r,
                                             const int num_cols_r) {
    const int warp_id = tid / 32;   // global warp id
    const int lane = tid & (32-1);  // local thread id within warp
    const int icol = warp_id;       // lhs column that this warp computes

    // Compute range of nnz elements in this column
    const int low  = static_cast<int>(indptr_l[icol]);
    const int high = static_cast<int>(indptr_l[icol+1]);

    // Iterate through the nnz elements in lhs column
    for (int j = low+lane; j < high; j+=32) {
      const int irow = static_cast<int>(col_idx_l[j]);
      const DType datum_l = data_l[j];
      // Iterate over all rhs columns
      for (int k = 0; k < num_cols_r; k++) {
        const DType val = datum_l*data_r[icol*num_cols_r+k];
        atomicAdd(static_cast<DType *>(&(out[irow*num_cols_r+k])), val);
      }
    }
  }
};

inline void DotCsrDnsDnsImpl(mshadow::Stream<gpu>* s,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  if (!lhs.storage_initialized()) return;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob& data_r = rhs;
  const TBlob data_out = *ret;

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        if (kWriteTo == req) {
          mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(s, data_out.Size(), data_out.dptr<DType>());
        }
        int num_threads;
        const int threads_per_warp = 32;
        const int threads_per_block = kBaseThreadNum;
        const int num_rows_l = lhs.shape()[0];
        const int num_cols_r = rhs.shape_[1];
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
                mxnet_op::Kernel<DotCsrTransDnsDnsScalarKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_rows_l, num_cols_r);
              });
              break;
            case 2:
              num_threads = threads_per_warp * num_rows_l * num_cols_r;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                mxnet_op::Kernel<DotCsrTransDnsDnsWarpKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            case 3:
              num_threads = threads_per_block * num_rows_l;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                mxnet_op::Kernel<DotCsrTransDnsDnsThreadBlockKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            case 4:
              num_threads = threads_per_warp * num_rows_l;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                mxnet_op::Kernel<DotCsrTransDnsDnsWarpBlockKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            default:
              num_threads = threads_per_warp * num_rows_l * num_cols_r;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                mxnet_op::Kernel<DotCsrTransDnsDnsWarpKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
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
                mxnet_op::Kernel<DotCsrDnsDnsScalarKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            case 2:
              num_threads = threads_per_warp * num_rows_l * num_cols_r;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                mxnet_op::Kernel<DotCsrDnsDnsVectorKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            default:
              if (num_cols_r > 4) {
                num_threads = data_out.Size();
                MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                  mxnet_op::Kernel<DotCsrDnsDnsScalarKernel<ReqType>, gpu>::Launch(s, num_threads,
                      data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                      col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
                });
              } else {
                num_threads = threads_per_warp * num_rows_l * num_cols_r;
                MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                  mxnet_op::Kernel<DotCsrDnsDnsVectorKernel<ReqType>, gpu>::Launch(s, num_threads,
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
 * \brief Impl of dot(csr.T, dns) = rsp
 */
inline void DotCsrDnsRspImpl(mshadow::Stream<gpu>* s,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  LOG(FATAL) << "DotCsrDnsRspImpl gpu version is not implemented.";
}

/*!
 * \brief Impl of dot(csr.T, rsp) = rsp2
 */
inline void DotCsrRspRspImpl(mshadow::Stream<gpu>* s,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  LOG(FATAL) << "DotCsrRspRspImpl gpu version is not implemented.";
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_DOT_INL_CUH_
