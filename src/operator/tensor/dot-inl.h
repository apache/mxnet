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
 * \file dot-inl.h
 * \brief Function definition of matrix dot operator
 */

#ifndef MXNET_OPERATOR_TENSOR_DOT_INL_H_
#define MXNET_OPERATOR_TENSOR_DOT_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <type_traits>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"
#ifdef __CUDACC__
#include "./dot-inl.cuh"
#endif  // __CUDACC__

namespace mxnet {
namespace op {

struct DotParam : public dmlc::Parameter<DotParam> {
  bool transpose_a;
  bool transpose_b;
  DMLC_DECLARE_PARAMETER(DotParam) {
    DMLC_DECLARE_FIELD(transpose_a)
      .describe("If true then transpose the first input before dot.")
      .set_default(false);
    DMLC_DECLARE_FIELD(transpose_b)
      .describe("If true then transpose the second input before dot.")
      .set_default(false);
  }
};

template<typename xpu>
void DotForward_(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(outputs[0].type_flag_ == kFloat32 || outputs[0].type_flag_ == kFloat64)
      << "dot only supports float32 and float64";
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if (inputs[0].ndim() == 1 && inputs[1].ndim() == 1) {
      CHECK_NE(req[0], kAddTo) << "AddTo not yet supported";
      Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
      VectorDot(out,
                inputs[0].get<xpu, 1, DType>(s),
                inputs[1].get<xpu, 1, DType>(s));
    } else {
      int ma, na, mb, nb, m, n;
      if (param.transpose_a) {
        ma = inputs[0].size(0);
        na = inputs[0].Size()/ma;
        m = na;
      } else {
        na = inputs[0].size(inputs[0].ndim()-1);
        ma = inputs[0].Size()/na;
        m = ma;
      }
      if (param.transpose_b) {
        nb = inputs[1].size(inputs[1].ndim()-1);
        mb = inputs[1].Size()/nb;
        n = mb;
      } else {
        mb = inputs[1].size(0);
        nb = inputs[1].Size()/mb;
        n = nb;
      }
      Tensor<xpu, 2, DType> input0 =
      inputs[0].get_with_shape<xpu, 2, DType>(Shape2(ma, na), s);
      Tensor<xpu, 2, DType> input1 =
      inputs[1].get_with_shape<xpu, 2, DType>(Shape2(mb, nb), s);
      Tensor<xpu, 2, DType> out =
      outputs[0].get_with_shape<xpu, 2, DType>(Shape2(m, n), s);
      if (param.transpose_a && param.transpose_b) {
        ASSIGN_DISPATCH(out, req[0], dot(input0.T(), input1.T()));
      } else if (!param.transpose_a && param.transpose_b) {
        ASSIGN_DISPATCH(out, req[0], dot(input0, input1.T()));
      } else if (param.transpose_a && !param.transpose_b) {
        ASSIGN_DISPATCH(out, req[0], dot(input0.T(), input1));
      } else {
        ASSIGN_DISPATCH(out, req[0], dot(input0, input1));
      }
    }
  });
}

template<typename xpu>
void DotBackward_(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(req[0], kWriteInplace);
  CHECK_NE(req[1], kWriteInplace);
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if (inputs[1].ndim() == 1 && inputs[2].ndim() == 1) {
      Tensor<xpu, 1, DType> mout_grad = inputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> mlhs_data = inputs[1].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> mrhs_data = inputs[2].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> mlhs_grad = outputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> mrhs_grad = outputs[1].get<xpu, 1, DType>(s);
      ASSIGN_DISPATCH(mrhs_grad, req[1],
                      broadcast_scalar(mout_grad, mlhs_data.shape_) * mlhs_data);
      ASSIGN_DISPATCH(mlhs_grad, req[0],
                      broadcast_scalar(mout_grad, mlhs_data.shape_) * mrhs_data);
    } else {
      int ma, na, mb, nb, m, n;
      if (param.transpose_a) {
        ma = outputs[0].size(0);
        na = outputs[0].Size()/ma;
        m = na;
      } else {
        na = outputs[0].size(outputs[0].ndim()-1);
        ma = outputs[0].Size()/na;
        m = ma;
      }
      if (param.transpose_b) {
        nb = outputs[1].size(outputs[1].ndim()-1);
        mb = outputs[1].Size()/nb;
        n = mb;
      } else {
        mb = outputs[1].size(0);
        nb = outputs[1].Size()/mb;
        n = nb;
      }
      Tensor<xpu, 2, DType> mout_grad =
      inputs[0].get_with_shape<xpu, 2, DType>(Shape2(m, n), s);
      Tensor<xpu, 2, DType> mlhs_data =
      inputs[1].get_with_shape<xpu, 2, DType>(Shape2(ma, na), s);
      Tensor<xpu, 2, DType> mrhs_data =
      inputs[2].get_with_shape<xpu, 2, DType>(Shape2(mb, nb), s);
      Tensor<xpu, 2, DType> mlhs_grad =
      outputs[0].get_with_shape<xpu, 2, DType>(Shape2(ma, na), s);
      Tensor<xpu, 2, DType> mrhs_grad =
      outputs[1].get_with_shape<xpu, 2, DType>(Shape2(mb, nb), s);
      if (param.transpose_a && param.transpose_b) {
        // Gradient of z = dot(x.T, y.T)
        // dy = dot(x, dz).T = dot(dz.T, x.T)
        // dx = dot(dz, y).T = dot(y.T, dz.T)
        ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mout_grad.T(), mlhs_data.T()));
        ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mrhs_data.T(), mout_grad.T()));
      } else if (!param.transpose_a && param.transpose_b) {
        // Gradient of z = dot(x, y.T)
        // dy = dot(x.T, dz).T = dot(dz.T, x)
        // dx = dot(dz, y)
        ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mout_grad.T(), mlhs_data));
        ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mout_grad, mrhs_data));
      } else if (param.transpose_a && !param.transpose_b) {
        // Gradient of z = dot(x.T, y)
        // dy = dot(x, dz)
        // dx = dot(dz, y.T).T = dot(y, dz.T)
        ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mlhs_data, mout_grad));
        ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mrhs_data, mout_grad.T()));
      } else {
        // Gradient of z = dot(x, y)
        // dy = dot(x.T, dz)
        // dx = dot(dz, y.T)
        ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mlhs_data.T(), mout_grad));
        ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mout_grad, mrhs_data.T()));
      }
    }
  });
}

inline bool DotForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                       const Context& ctx,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  // csr has many zero columns, so the result of dot(csr.T, matrix) should be rsp
  // TODO(stefan/haibin/jun): check type_assign return value
  if (param.transpose_a && kCSRStorage == (*in_attrs)[0]) {
    type_assign(&((*out_attrs)[0]), kRowSparseStorage);
  } else {
    type_assign(&((*out_attrs)[0]), kDefaultStorage);
  }
  return true;
}

inline bool DotBackwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                        const Context& ctx,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 2U);
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  type_assign(&((*out_attrs)[0]), kDefaultStorage);
  if (!param.transpose_a && kCSRStorage == (*in_attrs)[1]) {
    type_assign(&((*out_attrs)[1]), kRowSparseStorage);
  } else {
    type_assign(&((*out_attrs)[1]), kDefaultStorage);
  }
  return true;
}

/*!
 * \brief CPU Kernel of dot(csr, dns1) = dns2
 * Parallelization by row blocks
 */
struct DotCsrDnsDnsByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const nnvm::dim_t seg_len,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = std::min(seg_start + seg_len, num_rows);
    for (dim_t j = seg_start; j < seg_end; ++j) {
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_out = j * num_cols;
      for (IType k = indptr_l[j]; k < indptr_l[j+1]; ++k) {
        const DType val = data_l[k];
        const dim_t offset_r = col_idx_l[k] * num_cols;
        for (dim_t l = 0; l < num_cols; ++l) {
          out[offset_out+l] += data_r[offset_r+l] * val;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(csr.T(), dns1) = dns2
 * Parallelization by row blocks
 */
struct DotCsrTransDnsDnsByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const nnvm::dim_t seg_len,
                                  const nnvm::dim_t num_rows_l,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = (i + 1) * seg_len;
    for (dim_t j = 0; j < num_rows_l; ++j) {
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_r = j * num_cols;
      for (IType k = indptr_l[j]; k < indptr_l[j+1]; ++k) {
        const CType col_idx = col_idx_l[k];
        if (col_idx < seg_start || col_idx >= seg_end) continue;
        const dim_t offset_out = col_idx * num_cols;
        const DType val = data_l[k];
        for (dim_t l = 0; l < num_cols; ++l) {
          out[offset_out+l] += data_r[offset_r+l] * val;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(csr.T(), dns) = rsp
 * Parallelization by row blocks.
 * This kernel fills up the row_idx array of the rsp 
 * with 1 for nonzero rows and 0 for zero rows.
 * The matrix will be compacted after this kernel call.
 */
struct DotCsrTransDnsRspByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   */
  template<typename DType, typename RType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  RType* row_idx,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const nnvm::dim_t seg_len,
                                  const nnvm::dim_t num_rows_l,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = (i + 1) * seg_len;
    for (dim_t j = 0; j < num_rows_l; ++j) {
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_r = j * num_cols;
      for (IType k = indptr_l[j]; k < indptr_l[j+1]; ++k) {
        const CType col_idx = col_idx_l[k];
        if (col_idx < seg_start || col_idx >= seg_end) continue;
        const dim_t offset_out = col_idx * num_cols;
        row_idx[col_idx] = 1;
        const DType val = data_l[k];
        for (dim_t l = 0; l < num_cols; ++l) {
          out[offset_out+l] += data_r[offset_r+l] * val;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(csr, rsp) = dns
 * Parallelization by row blocks
 */
struct DotCsrRspDnsByRowBlocks {
  /*!
   * \brief
   * \param i         the i-th thread
   * \param nnr_r     storage_shape[0] of the rsp
   * \param num_rows  dns.shape[0]
   * \param num_cols  dns.shape[1]
   */
  template<typename DType, typename IType, typename CType, typename RType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const RType* row_idx_r,
                                  const nnvm::dim_t nnr_r,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols,
                                  const nnvm::dim_t seg_len) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = std::min(seg_start + seg_len, num_rows);
    for (dim_t j = seg_start; j < seg_end; ++j) {
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_out = j * num_cols;
      // Use binary search to find the lower_bound of val in row_idx array
      const RType* first = row_idx_r;
      const RType* last = row_idx_r + nnr_r;
      const CType val = col_idx_l[indptr_l[j]];
      const RType* it;
      int count = last - first, step;
      while (count > 0) {
        it = first;
        step = count / 2;
        it += step;
        if (*it < val) {
          first = ++it;
          count -= step + 1;
        } else {
          count = step;
        }
      }
      const RType* row_idx_ptr = first;
      // end of binary search
      if (row_idx_ptr == row_idx_r+nnr_r || *row_idx_ptr > col_idx_l[indptr_l[j+1]-1]) continue;
      for (IType k = indptr_l[j]; k < indptr_l[j+1] && row_idx_ptr != row_idx_r+nnr_r;) {
        if (col_idx_l[k] == *row_idx_ptr) {
          const dim_t offset_r = (row_idx_ptr - row_idx_r) * num_cols;
          for (dim_t l = 0; l < num_cols; ++l) {
            out[offset_out+l] += data_l[k] * data_r[offset_r+l];
          }
          ++k;
          ++row_idx_ptr;
        } else if (col_idx_l[k] < *row_idx_ptr) {
          ++k;
        } else {
          ++row_idx_ptr;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(csr.T(), rsp1) = rsp2, with row_idx marked for non-zero rows
 * Parallelization by row blocks
 */
struct DotCsrTransRspRspByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   * \param num_rows_l number of rows of lhs matrix
   * \param nnr_r number of non-zero rows of rhs matrix
   * \param num_rows number of rows of out matrix
   * \param num_cols number of cols of out matrix
   */
  template<typename DType, typename IType, typename CType, typename RType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  RType* row_idx_out,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const RType* row_idx_r,
                                  const nnvm::dim_t num_rows_l,
                                  const nnvm::dim_t nnr_r,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols,
                                  const nnvm::dim_t seg_len) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = (i + 1) * seg_len;
    for (dim_t rid = 0; rid < nnr_r; ++rid) {
      const RType j = row_idx_r[rid];
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_r = rid * num_cols;
      for (IType k = indptr_l[j]; k < indptr_l[j+1]; ++k) {
        const CType col_idx = col_idx_l[k];
        if (col_idx < seg_start || col_idx >= seg_end) continue;
        row_idx_out[col_idx] = 1;  // mark nonzero row as 1
        const dim_t offset_out = col_idx * num_cols;
        for (dim_t l = 0; l < num_cols; ++l) {
          out[offset_out+l] += data_r[offset_r+l] * data_l[k];
        }
      }
    }
  }
};

/*!
 * \brief CPU Impl of dot(csr, dns1) = dns2 and dot(csr.T, dns1) = dns2
 */
inline void DotCsrDnsDnsImpl(const OpContext& ctx,
                             const cpu& cpu_dev,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  if (!lhs.storage_initialized()) return;

  using nnvm::dim_t;

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob& data_r = rhs;
  const TBlob data_out = *ret;

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        dim_t num_threads;
        if (kWriteTo == req) {
          num_threads = data_out.Size();
          mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(
              s, num_threads, data_out.dptr<DType>());
        }
        num_threads = mxnet_op::get_num_threads<cpu>(data_out.shape_[0]);
        dim_t seg_len = (data_out.shape_[0] + num_threads - 1) / num_threads;
        if (trans_lhs) {
          mxnet_op::Kernel<DotCsrTransDnsDnsByRowBlocks, cpu>::Launch(s, num_threads,
              data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
              col_idx_l.dptr<CType>(), data_r.dptr<DType>(), seg_len,
              lhs.shape()[0], data_out.shape_[0], data_out.shape_[1]);
        } else {
          mxnet_op::Kernel<DotCsrDnsDnsByRowBlocks, cpu>::Launch(s, num_threads,
              data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
              col_idx_l.dptr<CType>(), data_r.dptr<DType>(), seg_len,
              data_out.shape_[0], data_out.shape_[1]);
        }
      });
    });
  });
}

/*!
 * \brief CPU Impl of dot(csr.T, dns) = rsp
 */
inline void DotCsrDnsRspImpl(const OpContext& ctx,
                             const cpu& cpu_dev,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(ret->storage_type(), kRowSparseStorage);
  if (!lhs.storage_initialized()) return;
  CHECK_EQ(req, kWriteTo);

  using mxnet_op::set_zero;
  using nnvm::dim_t;

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob& data_r = rhs;

  // pre-allocate spaces for ret using the dense dimension size
  ret->CheckAndAlloc({mshadow::Shape1(lhs.shape()[1])});
  const TBlob data_out = ret->data();
  const TBlob row_idx_out = ret->aux_data(rowsparse::kIdx);

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(row_idx_out.type_flag_, RType, {  // row idx type
          dim_t num_threads = data_out.Size();
          mxnet_op::Kernel<set_zero, cpu>::Launch(s, num_threads, data_out.dptr<DType>());
          RType* row_idx = row_idx_out.dptr<RType>();
          num_threads = row_idx_out.Size();
          mxnet_op::Kernel<set_zero, cpu>::Launch(s, num_threads, row_idx);
          num_threads = mxnet_op::get_num_threads<cpu>(data_out.shape_[0]);
          dim_t seg_len = (data_out.shape_[0] + num_threads - 1) / num_threads;
          if (trans_lhs) {
            mxnet_op::Kernel<DotCsrTransDnsRspByRowBlocks, cpu>::Launch(s, num_threads,
                data_out.dptr<DType>(), row_idx, data_l.dptr<DType>(),
                indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(), data_r.dptr<DType>(),
                seg_len, lhs.shape()[0], data_out.shape_[0], data_out.shape_[1]);
            dim_t nnr = 0;
            nnr = mxnet::common::ParallelAccumulate(row_idx, ret->shape()[0], nnr);
            ret->set_aux_shape(rowsparse::kIdx, mshadow::Shape1(nnr));
            if (0 == nnr) return;
            mshadow::Tensor<cpu, 2, DType> rsp_data = data_out.FlatTo2D<cpu, DType>(s);
            dim_t idx = 0;
            for (index_t i = 0; i < ret->shape()[0]; ++i) {
              if (row_idx[i] > 0) {
                row_idx[idx] = i;
                mshadow::Copy(rsp_data[idx], rsp_data[i], s);
                ++idx;
              }
            }
          } else {
            LOG(FATAL) << "DotCsrDnsRspImpl has not implemented dot(csr, dns)=rsp yet.";
          }
        });
      });
    });
  });
}

/*!
 * \brief CPU Impl of dot(csr, rsp) = dns
 */
inline void DotCsrRspDnsImpl(const OpContext& ctx,
                             const cpu& cpu_dev,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  if (kNullOp == req) return;
  // reuse csr dns implementation when storage_shape == shape for rhs
  if (rhs.storage_shape()[0] == rhs.shape()[0]) {  // if rsp is actually dense
    DotCsrDnsDnsImpl(ctx, cpu_dev, lhs, rhs.data(), req, trans_lhs, ret);
    return;
  }

  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(rhs.storage_type(), kRowSparseStorage);
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  if (!lhs.storage_initialized() || !rhs.storage_initialized()) {
    if (kWriteTo == req) {
      MSHADOW_SGL_DBL_TYPE_SWITCH(ret->type_flag_, DType, {  // data type
        mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(
            s, ret->Size(), ret->dptr<DType>());
      });
    }
    return;
  }
  using nnvm::dim_t;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob data_r = rhs.data();
  const TBlob row_idx_r = rhs.aux_data(rowsparse::kIdx);

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(row_idx_r.type_flag_, RType, {  // row idx type
          dim_t num_threads;
          if (kWriteTo == req) {
            num_threads = ret->Size();
            mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, num_threads,
                                                              ret->dptr<DType>());
          }
          num_threads = mxnet_op::get_num_threads<cpu>(ret->shape_[0]);
          dim_t seg_len = (ret->shape_[0] + num_threads - 1) / num_threads;
          if (trans_lhs) {
            LOG(FATAL) << "DotCsrRspDnsImpl has not implemented dot(csr.T, rsp) = dns yet";
          } else {
            mxnet_op::Kernel<DotCsrRspDnsByRowBlocks, cpu>::Launch(s, num_threads,
                ret->dptr<DType>(), data_l.dptr<DType>(),
                indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(), data_r.dptr<DType>(),
                row_idx_r.dptr<RType>(), rhs.storage_shape()[0],
                ret->shape_[0], ret->shape_[1], seg_len);
          }
        });
      });
    });
  });
}

/*!
 * \brief CPU Impl of dot(csr.T, rsp1) = rsp2
 */
inline void DotCsrRspRspImpl(const OpContext& ctx,
                             const cpu& cpu_dev,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  if (kNullOp == req) return;
  // reuse csr dns implementation when storage_shape == shape for rhs
  if (rhs.storage_shape()[0] == rhs.shape()[0]) {  // if rsp is actually dense
    DotCsrDnsRspImpl(ctx, cpu_dev, lhs, rhs.data(), req, trans_lhs, ret);
    return;
  }

  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(rhs.storage_type(), kRowSparseStorage);
  CHECK_EQ(ret->storage_type(), kRowSparseStorage);
  if (!lhs.storage_initialized() || !rhs.storage_initialized()) return;
  CHECK_EQ(req, kWriteTo);

  using mxnet_op::set_zero;
  using nnvm::dim_t;

  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob data_r = rhs.data();
  const TBlob row_idx_r = rhs.aux_data(rowsparse::kIdx);

  // pre-allocate spaces for ret using the dense dimension size
  if (ret->storage_type() == kRowSparseStorage) {
    ret->CheckAndAlloc({mshadow::Shape1(lhs.shape()[1])});
  }
  const TBlob data_out = ret->data();
  const TBlob row_idx_out = ret->aux_data(rowsparse::kIdx);

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(row_idx_r.type_flag_, RType, {  // row idx type
          dim_t num_threads = data_out.Size();
          mxnet_op::Kernel<set_zero, cpu>::Launch(s, num_threads, data_out.dptr<DType>());
          num_threads = mxnet_op::get_num_threads<cpu>(data_out.shape_[0]);
          dim_t seg_len = (data_out.shape_[0] + num_threads - 1) / num_threads;
          if (trans_lhs) {
            RType* row_idx = row_idx_out.dptr<RType>();
            num_threads = row_idx_out.Size();
            mxnet_op::Kernel<set_zero, cpu>::Launch(s, num_threads, row_idx);
            mxnet_op::Kernel<DotCsrTransRspRspByRowBlocks, cpu>::Launch(s, num_threads,
                data_out.dptr<DType>(), row_idx, data_l.dptr<DType>(),
                indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(), data_r.dptr<DType>(),
                row_idx_r.dptr<RType>(), lhs.shape()[0], rhs.storage_shape()[0],
                ret->shape()[0], ret->shape()[1], seg_len);
            dim_t nnr = 0;
            nnr = mxnet::common::ParallelAccumulate(row_idx, ret->shape()[0], nnr);
            ret->set_aux_shape(rowsparse::kIdx, mshadow::Shape1(nnr));
            if (0 == nnr) return;
            mshadow::Tensor<cpu, 2, DType> rsp_data = data_out.FlatTo2D<cpu, DType>(s);
            dim_t idx = 0;
            for (index_t i = 0; i < ret->shape()[0]; ++i) {
              if (row_idx[i] > 0) {
                row_idx[idx] = i;
                mshadow::Copy(rsp_data[idx], rsp_data[i], s);
                ++idx;
              }
            }
          } else {
            LOG(FATAL) << "DotCsrRspRspImpl has not implemented dot(csr, rsp) = rsp2 yet";
          }
        });
      });
    });
  });
}

inline bool DotShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_attrs,
                     std::vector<TShape> *out_attrs) {
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& lshape = (*in_attrs)[0];
  TShape& rshape = (*in_attrs)[1];
  if (lshape.ndim() == 1 && rshape.ndim() == 1) {
    CHECK(!param.transpose_a && !param.transpose_b) << "Cannot transpose vectors";
    CHECK_EQ(lshape[0], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
  } else {
    bool Ta = param.transpose_a, Tb = param.transpose_b;
    TShape L[2], R[2];
    if (Ta) {
      L[0] = mshadow::Shape1(lshape[0]);
      L[1] = lshape.ndim() > 1 ? TShape(&lshape[1], &lshape[lshape.ndim()]) : TShape(1);
    } else {
      L[0] = lshape.ndim() > 1 ? TShape(&lshape[0], &lshape[lshape.ndim()-1]) : TShape(1);
      L[1] = mshadow::Shape1(lshape[lshape.ndim()-1]);
    }
    if (Tb) {
      R[0] = rshape.ndim() > 1 ? TShape(&rshape[0], &rshape[rshape.ndim()-1]) : TShape(1);
      R[1] = mshadow::Shape1(rshape[rshape.ndim()-1]);
    } else {
      R[0] = mshadow::Shape1(rshape[0]);
      R[1] = rshape.ndim() > 1 ? TShape(&rshape[1], &rshape[rshape.ndim()]) : TShape(1);
    }

    if (L[!Ta].Size() != 0 && R[Tb].Size() != 0) {
      CHECK_EQ(L[!Ta].Size(), R[Tb].Size())
        << "dot shape error: " << lshape << " X " << rshape;
    }
    std::vector<index_t> buf;
    if (lshape.ndim() > 1) buf.insert(buf.end(), &L[Ta][0], &L[Ta][L[Ta].ndim()]);
    if (rshape.ndim() > 1) buf.insert(buf.end(), &R[!Tb][0], &R[!Tb][R[!Tb].ndim()]);
    TShape oshape(buf.begin(), buf.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  }
  return true;
}

template<typename xpu>
void DotForwardEx(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK(!param.transpose_b) << "transposing rhs of the sparse dot op is not supported";
  CHECK_EQ(inputs[0].shape().ndim(), 2) << "sparse dot only supports 2 dimensional lhs";
  CHECK_EQ(inputs[1].shape().ndim(), 2) << "sparse dot only supports 2 dimensional rhs";
  auto lhs_stype = inputs[0].storage_type();
  auto rhs_stype = inputs[1].storage_type();
  auto out_stype = outputs[0].storage_type();
  if (lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage && out_stype == kDefaultStorage) {
    TBlob ret = outputs[0].data();
    DotCsrDnsDnsImpl(ctx, xpu(), inputs[0], inputs[1].data(), req[0], param.transpose_a, &ret);
  } else if (lhs_stype == kCSRStorage && rhs_stype == kRowSparseStorage
      && out_stype == kDefaultStorage) {
    TBlob ret = outputs[0].data();
    DotCsrRspDnsImpl(ctx, xpu(), inputs[0], inputs[1], req[0], param.transpose_a, &ret);
  } else if (lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage
      && out_stype == kRowSparseStorage) {
    NDArray out = outputs[0];
    DotCsrDnsRspImpl(ctx, xpu(), inputs[0], inputs[1].data(), req[0], param.transpose_a, &out);
  } else if (lhs_stype == kCSRStorage && rhs_stype == kRowSparseStorage
      && out_stype == kRowSparseStorage) {
    NDArray ret = outputs[0];
    DotCsrRspRspImpl(ctx, xpu(), inputs[0], inputs[1], req[0], param.transpose_a, &ret);
  } else {
    FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs, DotForward_<xpu>, "DotForward_");
  }
}

template<typename xpu>
void DotBackwardEx(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<NDArray>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(kNullOp, req[0])
    << "sparse dot does not support computing the gradient of the csr/lhs";
  CHECK_NE(req[1], kWriteInplace) << "DotBackwardEx does not support WriteInplace";

  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK(!param.transpose_b) << "sparse dot only supports dot(A, X) and dot(A.T(), X)";
  CHECK_EQ(inputs[0].shape().ndim(), 2) << "sparse dot only supports 2 dimensional lhs";
  CHECK_EQ(inputs[1].shape().ndim(), 2) << "sparse dot only supports 2 dimensional rhs";
  const auto ograd_stype = inputs[0].storage_type();
  const auto lhs_stype = inputs[1].storage_type();
  const auto rhs_stype = inputs[2].storage_type();
  const auto grad_rhs_stype = outputs[1].storage_type();
  if (ograd_stype == kDefaultStorage  // ograd dns format
      && lhs_stype == kCSRStorage  // csr input lhs of the op
      && grad_rhs_stype == kDefaultStorage) {  // grad(rhs) dns format
    TBlob ret = outputs[1].data();
    DotCsrDnsDnsImpl(ctx, xpu(), inputs[1], inputs[0].data(), req[1], !param.transpose_a, &ret);
  } else if (ograd_stype == kDefaultStorage
      && lhs_stype == kCSRStorage
      && grad_rhs_stype == kRowSparseStorage) {
    NDArray ret = outputs[1];
    DotCsrDnsRspImpl(ctx, xpu(), inputs[1], inputs[0].data(), req[1], !param.transpose_a, &ret);
  } else {
    FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs, DotBackward_<xpu>, "DotBackward_");
  }
}

template<typename xpu>
void BatchDotForward_(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(outputs[0].type_flag_ == kFloat32 || outputs[0].type_flag_ == kFloat64)
      << "dot only supports float32 and float64";
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
    mshadow::Tensor<xpu, 3, DType> mlhs = inputs[0].get<xpu, 3, DType>(s);
    mshadow::Tensor<xpu, 3, DType> mrhs = inputs[1].get<xpu, 3, DType>(s);
    mshadow::Tensor<xpu, 1, DType*> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, DType*>(mshadow::Shape1(3 * out.size(0)), s);
    if (kNullOp != req[0]) {
      if (param.transpose_a && param.transpose_b) {
        mshadow::BatchGEMM<true, true>(out, mlhs, mrhs, (DType)1.0f,
                                       (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                       workspace);
      } else if (!param.transpose_a && param.transpose_b) {
        mshadow::BatchGEMM<false, true>(out, mlhs, mrhs, (DType)1.0f,
                                       (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                       workspace);
      } else if (param.transpose_a && !param.transpose_b) {
        mshadow::BatchGEMM<true, false>(out, mlhs, mrhs, (DType)1.0f,
                                       (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                       workspace);
      } else {
        mshadow::BatchGEMM<false, false>(out, mlhs, mrhs, (DType)1.0f,
                                       (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                       workspace);
      }
    }
  });
}

template<typename xpu>
void BatchDotBackward_(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK_NE(req[1], kWriteInplace);
  CHECK_NE(req[0], kWriteInplace);
  CHECK(outputs[0].type_flag_ == kFloat32 || outputs[0].type_flag_ == kFloat64)
      << "dot only supports float32 and float64";
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mshadow::Tensor<xpu, 3, DType> mout_grad = inputs[0].get<xpu, 3, DType>(s);
    mshadow::Tensor<xpu, 3, DType> mlhs_data = inputs[1].get<xpu, 3, DType>(s);
    mshadow::Tensor<xpu, 3, DType> mrhs_data = inputs[2].get<xpu, 3, DType>(s);
    mshadow::Tensor<xpu, 3, DType> mlhs_grad = outputs[0].get<xpu, 3, DType>(s);
    mshadow::Tensor<xpu, 3, DType> mrhs_grad = outputs[1].get<xpu, 3, DType>(s);
    mshadow::Tensor<xpu, 2, DType*> workspace =
      ctx.requested[0].get_space_typed<xpu, 2, DType*>(
        mshadow::Shape2(2, 3 * mout_grad.size(0)), s);
    mshadow::Tensor<xpu, 1, DType*> rhs_workspace = workspace[0];
    mshadow::Tensor<xpu, 1, DType*> lhs_workspace = workspace[1];
    if (param.transpose_a && param.transpose_b) {
      // Gradient of z = dot(x.T, y.T)
      // dy = dot(x, dz).T = dot(dz.T, x.T)
      // dx = dot(dz, y).T = dot(y.T, dz.T)
      if (kNullOp != req[1]) {
        mshadow::BatchGEMM<true, true>(mrhs_grad, mout_grad, mlhs_data, (DType)1.0f,
                                        (kAddTo == req[1]) ? (DType)1.0f :  (DType)0.0f,
                                        rhs_workspace);
      }
      if (kNullOp != req[0]) {
        mshadow::BatchGEMM<true, true>(mlhs_grad, mrhs_data, mout_grad, (DType)1.0f,
                                        (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                        lhs_workspace);
      }
    } else if (!param.transpose_a && param.transpose_b) {
      // Gradient of z = dot(x, y.T)
      // dy = dot(x.T, dz).T = dot(dz.T, x)
      // dx = dot(dz, y)
      if (kNullOp != req[1]) {
        mshadow::BatchGEMM<true, false>(mrhs_grad, mout_grad, mlhs_data, (DType)1.0f,
                                        (kAddTo == req[1]) ? (DType)1.0f : (DType)0.0f,
                                        rhs_workspace);
      }
      if (kNullOp != req[0]) {
        mshadow::BatchGEMM<false, false>(mlhs_grad, mout_grad, mrhs_data, (DType)1.0f,
                                        (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                        lhs_workspace);
      }
    } else if (param.transpose_a && !param.transpose_b) {
      // Gradient of z = dot(x.T, y)
      // dy = dot(x, dz)
      // dx = dot(dz, y.T).T = dot(y, dz.T)
      if (kNullOp != req[1]) {
        mshadow::BatchGEMM<false, false>(mrhs_grad, mlhs_data, mout_grad, (DType)1.0f,
                                        (kAddTo == req[1]) ? (DType)1.0f : (DType)0.0f,
                                        rhs_workspace);
      }
      if (kNullOp != req[0]) {
        mshadow::BatchGEMM<false, true>(mlhs_grad, mrhs_data, mout_grad, (DType)1.0f,
                                        (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                        lhs_workspace);
      }
    } else {
      // Gradient of z = dot(x, y)
      // dy = dot(x.T, dz)
      // dx = dot(dz, y.T)
      if (kNullOp != req[1]) {
        mshadow::BatchGEMM<true, false>(mrhs_grad, mlhs_data, mout_grad, (DType)1.0f,
                                        (kAddTo == req[1]) ? (DType)1.0f : (DType)0.0f,
                                        rhs_workspace);
      }
      if (kNullOp != req[0]) {
        mshadow::BatchGEMM<false, true>(mlhs_grad, mout_grad, mrhs_data, (DType)1.0f,
                                        (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                        lhs_workspace);
      }
    }
  });
}

inline bool BatchDotShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  TShape& lshape = (*in_attrs)[0];
  TShape& rshape = (*in_attrs)[1];
  if (lshape.ndim() == 3 && rshape.ndim() == 3) {
    CHECK(lshape[0] == rshape[0])
      << "batch_dot shape error(batch_size must be equal): " << lshape << " X " << rshape
      << " trans_a=" << param.transpose_a << " trans_b=" << param.transpose_b;
    index_t out_m = param.transpose_a ? lshape[2] : lshape[1];
    index_t lshape_k = param.transpose_a ? lshape[1] : lshape[2];
    index_t out_n = param.transpose_b ? rshape[1] : rshape[2];
    index_t rshape_k = param.transpose_b ? rshape[2] : rshape[1];
    CHECK(lshape_k == rshape_k)
      << "batch_dot shape error(shape mismatch): " << lshape << " X " << rshape
      << " trans_a=" << param.transpose_a << " trans_b=" << param.transpose_b;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape3(lshape[0], out_m, out_n));
  } else {
    LOG(FATAL) << "batch_dot currently only support 3D*3D array"
               << lshape << " v.s. " << rshape;
  }
  return true;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_DOT_INL_H_
