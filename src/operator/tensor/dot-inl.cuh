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

/*!
 * \brief Kernel of dot(csr, dns1) = dns2
 * Parallelization by output matrix elements
 */
template<int req>
struct DotCsrDnsDns {
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
 * \brief Kernel of dot(csr.T(), dns1) = dns2
 * Parallelization by output matrix elements
 */
template<int req>
struct DotCsrTransDnsDns {
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

  MSHADOW_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        if (trans_lhs) {
          MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
            mxnet_op::Kernel<DotCsrTransDnsDns<ReqType>, gpu>::Launch(s, data_out.Size(),
                data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                col_idx_l.dptr<CType>(), data_r.dptr<DType>(), lhs.shape()[0],
                data_out.shape_[1]);
          });
        } else {
          MXNET_ASSIGN_REQ_SWITCH(req, ReqType, {
            mxnet_op::Kernel<DotCsrDnsDns<ReqType>, gpu>::Launch(s, data_out.Size(),
                data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                col_idx_l.dptr<CType>(), data_r.dptr<DType>(), rhs.shape_[1]);
          });
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
