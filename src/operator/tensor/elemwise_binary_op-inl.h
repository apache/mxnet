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
 * \file elemwise_binary_op.h
 * \brief Function definition of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_H_

#include <vector>
#include <algorithm>
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {

/*! \brief binary op handling for the following row sparse inputs/outputs
  rsp, rsp -> rsp,
  dns, rsp -> rsp,
  rsp, dns -> rsp,
  dns, rsp -> dns,
  rsp, dns -> dns,
*/
template<typename DType, typename IType, typename OP>
void ElemwiseBinaryOp::RspRspOp(mshadow::Stream<cpu> *s,
                                const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const NDArray &lhs,
                                const NDArray &rhs,
                                const OpReqType req,
                                const NDArray &output,
                                const bool lhs_may_be_dense,
                                const bool rhs_may_be_dense,
                                const bool allow_inplace,
                                const bool scatter) {
  using namespace mshadow;
  using namespace mshadow::expr;

  const bool is_dense_result = output.storage_type() == kDefaultStorage;
  const bool lhs_is_dense = lhs.storage_type() == kDefaultStorage;
  const bool rhs_is_dense = rhs.storage_type() == kDefaultStorage;
  CHECK(!lhs_is_dense || lhs_may_be_dense) << "rvalue cannot be dense";
  CHECK(!rhs_is_dense || rhs_may_be_dense) << "rvalue cannot be dense";
  CHECK(!lhs_is_dense || !rhs_is_dense);
  // Only one item at most may be dense (lhs, rhs or result)
  if (rhs_is_dense) {
    // For right-side dense, in order to have sparse output, lhs input zero should
    // always output zero
    CHECK(fabs(static_cast<float>(OP::Map(DType(0), DType(99)))) < 1e-4f);
    CHECK(!is_dense_result);  // Currently not handled
  }
  if (lhs_is_dense) {
    // For left-side dense, in order to have sparse output, lhs input zero should
    // always output zero
    CHECK(fabs(static_cast<float>(OP::Map(DType(99), DType(0)))) < 1e-4f);
    CHECK(!is_dense_result);  // Currently not handled
  }

  // Memory Estimation: This is (roughly) the number of result rows. We may still
  // need to subtract the number of common rows
  bool lhs_in_place = false, rhs_in_place = false;
  const size_t num_rows_l = lhs_is_dense ? lhs.shape()[0] : lhs.aux_shape(rowsparse::kIdx).Size();
  const size_t num_rows_r = rhs_is_dense ? rhs.shape()[0] : rhs.aux_shape(rowsparse::kIdx).Size();
  if (is_dense_result) {
    output.CheckAndAlloc();
  } else {
    if (rhs_is_dense || scatter) {
      output.CheckAndAlloc({mshadow::Shape1(num_rows_l)});
    } else if (lhs_is_dense) {
      output.CheckAndAlloc({mshadow::Shape1(num_rows_r)});
    } else {
      lhs_in_place = IsSameArray(lhs, output);
      rhs_in_place = IsSameArray(rhs, output);
      if (!lhs_in_place && !rhs_in_place) {
        output.CheckAndAlloc({mshadow::Shape1(num_rows_l + num_rows_r)});
      } else {
        CHECK_EQ(allow_inplace, true);
        CHECK_EQ(is_dense_result, false);
        if (lhs_in_place) {
          // For in-place, zero L-value must always be zero output
          DCHECK(fabs(static_cast<float>(OP::Map(DType(0), DType(99)))) < DType(1e-3));
        } else {
          // For in-place, zero R-value must always be zero output
          DCHECK(fabs(static_cast<float>(OP::Map(DType(99), DType(0)))) < DType(1e-3));
        }
      }
    }
  }

  // Indices
  const Tensor<cpu, 1, IType> indices_l = lhs_is_dense
                                          ? Tensor<cpu, 1, IType>()
                                          : lhs.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);
  const Tensor<cpu, 1, IType> indices_r = rhs_is_dense
                                          ? Tensor<cpu, 1, IType>()
                                          : rhs.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);
  Tensor<cpu, 1, IType> indices_out = is_dense_result
                                      ? Tensor<cpu, 1, IType>()
                                      : output.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);

  // Data
  // TODO(cjolivier01): Change to get_with_shape() calls
  const Tensor<cpu, 2, DType> data_l = AsRowise2D<DType>(s, lhs.data());
  const Tensor<cpu, 2, DType> data_r = AsRowise2D<DType>(s, rhs.data());
  Tensor<cpu, 2, DType> out = AsRowise2D<DType>(s, output.data());

  size_t iter_l = 0;
  size_t iter_r = 0;
  size_t iter_out = 0;
  int32_t num_common_rows = 0;

  if (is_dense_result) {
    if (!num_rows_l && !num_rows_r) {
      const size_t all_rows = static_cast<size_t>(lhs.shape()[0]);
      iter_out = FillDense<DType, OP>(s, all_rows, all_rows, req, &out, iter_out);
    }
  }

  while (iter_l < num_rows_l && iter_r < num_rows_r) {
    IType idx_l = lhs_is_dense ? indices_r[iter_r] : indices_l[iter_l];
    IType idx_r = rhs_is_dense ? idx_l : indices_r[iter_r];
    if (lhs_in_place) {
      while (idx_r < idx_l && ++iter_r < num_rows_r) {
        idx_r = indices_r[iter_r];
      }
      if (iter_r >= num_rows_r) {
        break;
      }
    } else if (rhs_in_place) {
      while (idx_l < idx_r && ++iter_l < num_rows_l) {
        idx_l = indices_l[iter_l];
      }
      if (iter_l >= num_rows_l) {
        break;
      }
    }
    if (is_dense_result) {
      iter_out = FillDense<DType, OP>(s, idx_l, idx_r, req, &out, iter_out);
      DCHECK_EQ(iter_out, static_cast<size_t>(std::min(idx_l, idx_r)));
    }
    if (idx_l == idx_r) {
      // Same row
      if (!is_dense_result) {
        indices_out[iter_out] = idx_l;
      }
      Tensor<cpu, 1, DType> lvalue = !lhs_is_dense ? data_l[iter_l++] : data_l[idx_l];
      Tensor<cpu, 1, DType> rvalue = !rhs_is_dense ? data_r[iter_r++] : data_r[idx_r];
      DCHECK_EQ(lvalue.shape_.Size(), rvalue.shape_.Size());
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, cpu>::Launch(
          s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_, rvalue.dptr_);
      });
      num_common_rows++;
    } else if (idx_l < idx_r) {
      // Left only
      if (!is_dense_result) {
        indices_out[iter_out] = idx_l;
      }
      Tensor<cpu, 1, DType> lvalue = !lhs_is_dense ? data_l[iter_l++] : data_l[idx_l];
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        mxnet_op::Kernel<MissingRValueOp<OP, Req>, cpu>::Launch(
          s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_);
      });
    } else {
      // Right only
      if (scatter) {
        ++iter_r;
        continue;  // skip '++iter_out' below
      }
      if (!is_dense_result) {
        indices_out[iter_out] = idx_r;
      }
      Tensor<cpu, 1, DType> rvalue = !rhs_is_dense ? data_r[iter_r++] : data_r[idx_r];
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        mxnet_op::Kernel<MissingLValueOp<OP, Req>, cpu>::Launch(
          s, rvalue.shape_.Size(), out[iter_out].dptr_, rvalue.dptr_);
      });
    }
    ++iter_out;
  }
  // Evaluate the remaining rows beyond the l and r value row intersetion
  while (iter_l < num_rows_l && !lhs_is_dense && !rhs_in_place) {
    if (!is_dense_result) {
      indices_out[iter_out] = indices_l[iter_l];
    } else {
      const IType idx_l = indices_l[iter_l];
      iter_out = FillDense<DType, OP>(s, lhs.shape()[0], idx_l, req, &out, iter_out);
    }
    Tensor<cpu, 1, DType> lvalue = data_l[iter_l++];
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      mxnet_op::Kernel<MissingRValueOp<OP, Req>, cpu>::Launch(
        s, lvalue.shape_.Size(), out[iter_out++].dptr_, lvalue.dptr_);
    });
  }
  while (iter_r < num_rows_r && !rhs_is_dense && !lhs_in_place && !scatter) {
    if (!is_dense_result) {
      indices_out[iter_out] = indices_r[iter_r];
    } else {
      const IType idx_r = indices_r[iter_r];
      iter_out = FillDense<DType, OP>(s, lhs.shape()[0], idx_r, req, &out, iter_out);
    }
    Tensor<cpu, 1, DType> rvalue = data_r[iter_r++];
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      mxnet_op::Kernel<MissingLValueOp<OP, Req>, cpu>::Launch(
        s, rvalue.shape_.Size(), out[iter_out++].dptr_, rvalue.dptr_);
    });
  }
  if (is_dense_result) {
    const size_t all_rows = static_cast<size_t>(lhs.shape()[0]);
    iter_out = FillDense<DType, OP>(s, all_rows, all_rows, req, &out, iter_out);
  } else {
    if (lhs_in_place) {
      CHECK_LE(iter_out, num_rows_l);
    }
    if (rhs_in_place) {
      CHECK_LE(iter_out, num_rows_r);
    }
    DCHECK_LE(iter_out, num_rows_l + num_rows_r);  // Make sure that we didn't overrun
    nnvm::TShape new_shape = output.aux_shape(rowsparse::kIdx);
    CHECK_LE(iter_out, new_shape.Size());
    if (!rhs_is_dense && !lhs_is_dense && !lhs_in_place && !rhs_in_place && !scatter) {
      // Reduce the first-dimension size by the number of common rows
      new_shape[0] -= num_common_rows;
      output.set_aux_shape(rowsparse::kIdx, new_shape);
    }
  }
}

/*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
template<typename DType, typename IType, typename CType, typename OP>
void ElemwiseBinaryOp::CsrCsrOp(mshadow::Stream<cpu> *s,
                                const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const NDArray &lhs,
                                const NDArray &rhs,
                                const OpReqType req,
                                const NDArray &output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;

  const auto nr_rows = static_cast<size_t>(lhs.shape()[0]);
  if (!nr_rows) {
    return;
  }
  CHECK_EQ(lhs.aux_shape(csr::kIndPtr).Size(), nr_rows + 1);
  const size_t nr_cols = lhs.shape().Size() / nr_rows;

  CHECK_EQ(lhs.shape().Size(), rhs.shape().Size());

  const bool same_lhs_rhs = IsSameArray(lhs, rhs);

  const size_t lhs_nnz = lhs.storage_shape().Size();
  const size_t rhs_nnz = rhs.storage_shape().Size();

  const size_t output_nnz_guess = same_lhs_rhs ? lhs_nnz : lhs_nnz + rhs_nnz;

  output.CheckAndAlloc({mshadow::Shape1(lhs.shape()[0] + 1),
                        mshadow::Shape1(std::min(output_nnz_guess, lhs.shape().Size()))});
  DCHECK_EQ(output.aux_shape(csr::kIndPtr), lhs.aux_shape(csr::kIndPtr));

  const size_t alloc_size = nr_cols * sizeof(IType) + 2 * nr_cols * sizeof(DType);

  Tensor<cpu, 1, uint8_t> workspace =
    ctx.requested[ResourceRequestType::kTempSpace].get_space_typed<cpu, 1, uint8_t>(
      mshadow::Shape1(alloc_size), s);

  // Allocate temp space and partition into three tensors
  mshadow::Tensor<cpu, 1, IType> next(reinterpret_cast<IType *>(workspace.dptr_),
                                      Shape1(nr_cols));
  mshadow::Tensor<cpu, 1, DType> lhs_row(reinterpret_cast<DType *>(workspace.dptr_
                                                                   + nr_cols * sizeof(IType)),
                                         Shape1(nr_cols));
  mshadow::Tensor<cpu, 1, DType> rhs_row(lhs_row.dptr_ + nr_cols, Shape1(nr_cols));

  OpBase::FillDense<IType>(s, next.shape_.Size(), IType(-1), req, next.dptr_);
  OpBase::FillDense<DType>(s, lhs_row.shape_.Size(), DType(0),  req, lhs_row.dptr_);
  if (!same_lhs_rhs) {
    OpBase::FillDense<DType>(s, rhs_row.shape_.Size(), DType(0), req, rhs_row.dptr_);
  }

  // Column indices
  const Tensor<cpu, 1, IType> col_indices_l = lhs.aux_data(csr::kIdx).FlatTo1D<cpu, IType>(s);
  const Tensor<cpu, 1, IType> col_indices_r = rhs.aux_data(csr::kIdx).FlatTo1D<cpu, IType>(s);
  Tensor<cpu, 1, IType> col_indices_out = output.aux_data(csr::kIdx).FlatTo1D<cpu, IType>(s);

  // Row pointers
  const Tensor<cpu, 1, CType> row_ptr_l = lhs.aux_data(csr::kIndPtr).FlatTo1D<cpu, CType>(s);
  const Tensor<cpu, 1, CType> row_ptr_r = rhs.aux_data(csr::kIndPtr).FlatTo1D<cpu, CType>(s);
  Tensor<cpu, 1, CType> row_ptr_out = output.aux_data(csr::kIndPtr).FlatTo1D<cpu, CType>(s);

  Tensor<cpu, 1, DType>   data_l = lhs.data().FlatTo1D<cpu, DType>(s);
  Tensor<cpu, 1, DType>   data_r = rhs.data().FlatTo1D<cpu, DType>(s);
  Tensor<cpu, 1, DType> data_out = output.data().FlatTo1D<cpu, DType>(s);

  IType nnz = 0;
  row_ptr_out[0] = 0;

  for (IType i = 0; i < static_cast<IType>(nr_rows); i++) {
    IType head = -2;
    IType length = 0;

    // add a row of A to lhs_row
    const IType i_start_l = row_ptr_l[i];
    const IType i_end_l = row_ptr_l[i + 1];
    for (IType jj = i_start_l; jj < i_end_l; jj++) {
      IType col = col_indices_l[jj];
      lhs_row[col] += data_l[jj];

      if (next[col] == -1) {
        next[col] = head;
        head = col;
        ++length;
      }
    }

    // add a row of B to rhs_row
    const IType i_start_r = row_ptr_r[i];
    const IType i_end_r = row_ptr_r[i + 1];
    for (IType jj = i_start_r; jj < i_end_r; jj++) {
      const IType col = col_indices_r[jj];
      rhs_row[col] += data_r[jj];

      if (next[col] == -1) {
        next[col] = head;
        head = col;
        ++length;
      }
    }

    // scan through columns where A or B has
    // contributed a non-zero entry
    for (IType jj = 0; jj < length; jj++) {
      const DType result = OP::Map(lhs_row[head], rhs_row[head]);

      if (result != 0) {
        col_indices_out[nnz] = head;
        data_out[nnz] = result;
        ++nnz;
      }

      const IType temp = head;
      head = next[head];

      next[temp] = -1;
      lhs_row[temp] = 0;
      rhs_row[temp] = 0;
    }

    row_ptr_out[i + 1] = nnz;
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_H_
