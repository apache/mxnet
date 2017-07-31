/*!
 *  Copyright (c) 2017 by Contributors
 * \file dot-inl.cuh
 * \brief implementation of matrix dot op on GPU
 */
#ifndef MXNET_OPERATOR_TENSOR_DOT_INL_CUH_
#define MXNET_OPERATOR_TENSOR_DOT_INL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>

// TODO(stefan): change dot interface s.t. it includes OpContext
namespace mxnet {
namespace op {
using nnvm::dim_t;

/*!
 * \brief Scalar kernel of dot(csr, dns1) = dns2
 * Parallelization by output matrix elements: 1 thread/element
 */
template<int req>
struct DotCsrDnsDnsScalarKernel {
  /*!
   * \brief This function represents performing an inner product between a row of lhs
   * and a column of rhs and then assigning the value to out[tid].
   * \param tid         global thread id
   * \param out         output matrix
   * \param data_l      csr values of lhs
   * \param indptr_l    csr indptr of lhs
   * \param col_idx_l   csr col_idx of lhs
   * \param data_r      dense data of rhs
   * \param num_cols_r  number of columns of output matrix
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const dim_t num_cols_r) {
    const dim_t irow = tid / num_cols_r;  // row id of the lhs
    const dim_t icol = tid % num_cols_r;  // col id of the rhs
    DType sum = 0;
    for (IType j = indptr_l[irow]; j < indptr_l[irow+1]; ++j) {
      const CType cur_col = col_idx_l[j];  // corresponding row id of the rhs
      sum += data_l[j] * data_r[cur_col*num_cols_r+icol];
    }
    KERNEL_ASSIGN(out[tid], req, sum);
  }
};

/*!
 * \brief Vector kernel of dot(csr, dns1) = dns2
 * Parallelization by output matrix elements: 1 warp/element
 */
template<int req>
struct DotCsrDnsDnsVectorKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const dim_t num_cols_r) {
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
 * \brief Scalar kernel of dot(csr.T, dns1) = dns2
 * Parallelization by output matrix elements: 1 thread/element
 */
template<int req>
struct DotCsrTransDnsDnsScalarKernel {
  /*!
   * \brief This function represents performing an inner product between a column of lhs
   * and a column of rhs and then assigning the value to out[tid].
   * \param tid         global thread id
   * \param out         output matrix
   * \param data_l      csr values of lhs
   * \param indptr_l    csr indptr of lhs
   * \param col_idx_l   csr col_idx of lhs
   * \param data_r      dense data of rhs
   * \param num_rows_l  number of rows of lhs (= number of columns of csr.T)
   * \param num_cols_r  number of columns of output matrix
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const dim_t num_rows_l,
                                             const dim_t num_cols_r) {
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
 * \brief Warp kernel of dot(csr.T, dns1) = dns2
 * Parallelization by columns: 1 warp computes one lhs column for one rhs column
 */
template<int req>
struct DotCsrTransDnsDnsWarpKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const dim_t num_cols_r) {
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
 * \brief Thread block kernel of dot(csr.T, dns1) = dns2
 * Parallelization by columns: 1 thread block computes one lhs column for all rhs columns
 */
template<int req>
struct DotCsrTransDnsDnsThreadBlockKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const dim_t num_cols_r) {
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
 * \brief Warp block kernel of dot(csr.T, dns1) = dns2
 * Parallelization by columns: 1 warp computes one lhs column for all rhs columns
 */
template<int req>
struct DotCsrTransDnsDnsWarpBlockKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const DType* data_l,
                                             const IType* indptr_l,
                                             const CType* col_idx_l,
                                             const DType* data_r,
                                             const dim_t num_cols_r) {
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
 * \brief GPU auxiliary kernel to flag non-zero rows of rsp tensor with indices
 */
struct SetRspRowFlgKernel {
  template<typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             RType* row_flg,
                                             const RType* row_idx,
                                             const dim_t nnr) {
    if (tid < nnr) {
      row_flg[row_idx[tid]] = tid+1;
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
   * \param out        dense output matrix
   * \param data_l     lhs csr matrix data
   * \param indptr_l   lhs csr matrix index pointer
   * \param col_idx_l  lhs csr matrix column indices
   * \param data_r     rhs rsp matrix data
   * \param row_idx_r  rhs rsp matrix nnr indices
   * \param row_flg_r  auxiliary index array holding
   * \param nnr_r      rhs rsp matrix number of non-zero rows
   * \param num_rows   dense output matrix number of rows
   * \param num_cols   dense output matrix number of columns
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
                                             const dim_t nnr_r,
                                             const dim_t num_rows,
                                             const dim_t num_cols) {
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
 * \brief GPU Impl of dot(csr, dns1) = dns2 and dot(csr.T, dns1) = dns2
 */
inline void DotCsrDnsDnsImpl(mshadow::Stream<gpu>* s,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  if (!lhs.storage_initialized()) return;

  using mshadow::cuda::kBaseThreadNum;
  using mxnet_op::Kernel;
  using mxnet_op::set_zero;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob& data_r = rhs;
  const TBlob data_out = *ret;

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        if (kWriteTo == req) {
          Kernel<set_zero, gpu>::Launch(s, data_out.Size(), data_out.dptr<DType>());
        }
        const dim_t threads_per_warp = mxnet_op::cuda_get_device_prop().warpSize;
        const dim_t threads_per_block = kBaseThreadNum;
        const dim_t num_rows_l = lhs.shape()[0];
        const dim_t num_cols_r = rhs.shape_[1];
        dim_t num_threads;
        // TODO: remove kernel dependency on warpSize=32
        if (threads_per_warp != 32) {
          LOG(FATAL) << "DotCsrDnsDnsImpl GPU kernels expect warpSize=32";
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
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                Kernel<DotCsrTransDnsDnsWarpKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            case 3:
              num_threads = threads_per_block * num_rows_l;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                Kernel<DotCsrTransDnsDnsThreadBlockKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            case 4:
              num_threads = threads_per_warp * num_rows_l;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                Kernel<DotCsrTransDnsDnsWarpBlockKernel<ReqType>, gpu>::Launch(s, num_threads,
                    data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                    col_idx_l.dptr<CType>(), data_r.dptr<DType>(), num_cols_r);
              });
              break;
            default:
              num_threads = threads_per_warp * num_rows_l * num_cols_r;
              MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
                Kernel<DotCsrTransDnsDnsWarpKernel<ReqType>, gpu>::Launch(s, num_threads,
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
inline void DotCsrDnsRspImpl(mshadow::Stream<gpu>* s,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  // TODO(stefan): Implement dot(csr.T, dns) = rsp
  LOG(FATAL) << "DotCsrDnsRspImpl gpu version is not implemented.";
}

/*!
 * \brief GPU Impl of dot(csr, rsp1) = rsp2 and dot(csr.T, rsp1) = rsp2
 */
inline void DotCsrRspRspImpl(mshadow::Stream<gpu>* s,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  // TODO(stefan): Implement dot(csr.T, rsp1) = rsp2
  LOG(FATAL) << "DotCsrRspRspImpl gpu version is not implemented.";
}

/*!
 * \brief GPU Impl of dot(csr, rsp) = dns and dot(csr.T, rsp) = dns
 */
inline void DotCsrRspDnsImpl(mshadow::Stream<gpu>* s,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  // reuse dot(csr, dns) implementation if rhs rsp matrix is in fact dense
  if (rhs.storage_shape()[0] == rhs.shape()[0]) {
    DotCsrDnsDnsImpl(s, lhs, rhs.data(), req, trans_lhs, ret);
    return;
  }
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(rhs.storage_type(), kRowSparseStorage);

  using mxnet_op::Kernel;
  using mxnet_op::set_zero;

  if (!lhs.storage_initialized() || !rhs.storage_initialized()) {
    if (kWriteTo == req) {
      MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {  // data type
        Kernel<set_zero, gpu>::Launch(s, ret->Size(), ret->dptr<DType>());
      });
    }
    return;
  }

  const dim_t num_rows = ret->shape_[0];
  const dim_t num_cols = ret->shape_[1];
  const dim_t nnr_r = rhs.storage_shape()[0];
  dim_t num_threads;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob data_r = rhs.data();
  const TBlob row_idx_r = rhs.aux_data(rowsparse::kIdx);

  MSHADOW_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(row_idx_r.type_flag_, RType, {  // col idx type
          if (kWriteTo == req) {
            num_threads = num_rows*num_cols;
            Kernel<set_zero, gpu>::Launch(s, num_threads, ret->dptr<DType>());
          }
          if (trans_lhs) {
            LOG(FATAL) << "DotCsrRspDnsImpl has not implemented dot(csr.T, rsp) = dns yet";
          } else {
            // TODO: Consider implementing a vector kernel for SpMV (similar to DotCsrDnsDns)
            // Alloc temp storage for row_flg array
            // TODO(stefan): use temporary workspace from OpContext
            RType* row_flg_r;
            CUDA_CALL(cudaMalloc(&row_flg_r, rhs.shape()[0]*sizeof(RType)));
            num_threads = rhs.shape()[0];
            Kernel<set_zero, gpu>::Launch(s, num_threads, row_flg_r);
            // Set row_flg index array
            num_threads = nnr_r;
            Kernel<SetRspRowFlgKernel, gpu>::Launch(s, num_threads,
                row_flg_r, row_idx_r.dptr<RType>(), nnr_r);
            // Perform sparse matrix-matrix multiply
            num_threads = num_rows*num_cols;
            Kernel<DotCsrRspDnsScalarKernel, gpu>::Launch(s, num_threads,
                ret->dptr<DType>(),
                data_l.dptr<DType>(), indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(),
                data_r.dptr<DType>(), row_idx_r.dptr<RType>(), row_flg_r, rhs.storage_shape()[0],
                num_rows, num_cols);
            // Dealloc temp storage
            CUDA_CALL(cudaFree(row_flg_r));
          }
        });
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_DOT_INL_CUH_
