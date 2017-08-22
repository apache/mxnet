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
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <vector>
#include <string>
#include <utility>
#include <typeinfo>
#include <algorithm>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "elemwise_unary_op.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op {

/*! Gather binary operator functions into BinaryOp class */
class ElemwiseBinaryOp : public OpBase {
 public:
  template<typename OP, int Req>
  struct BinaryOpBackwardUseNone {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *igrad, const DType *ograd) {
      KERNEL_ASSIGN(igrad[i], Req, OP::Map(ograd[i]));
    }
    template<typename DType>
    MSHADOW_XINLINE static DType Map(const DType ograd) {
      return OP::Map(ograd);
    }
  };

  template<typename OP, int Req>
  struct BinaryOpBackwardUseIn {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *igrad,
                                    const DType *ograd, const DType *lhs, const DType *rhs) {
      KERNEL_ASSIGN(igrad[i], Req, ograd[i] * OP::Map(lhs[i], rhs[i]));
    }
  };

  /*! \brief For sparse, assume missing rvalue is 0 */
  template<typename OP, int Req>
  struct BinaryOpMissingRValue {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(lhs[i], DType(0)));
    }
  };

  /*! \brief For sparse, assume missing lvalue is 0 */
  template<typename OP, int Req>
  struct BinaryOpMissingLValue {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *rhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(DType(0), rhs[i]));
    }
  };

 private:
  /*! \brief Fill dense output block with a single scalar value */
  template<typename xpu, typename DType>
  static inline void Fill(mshadow::Stream<xpu> *s,
                          const DType val,
                          const OpReqType req,
                          const TBlob& blob) {
    using namespace mxnet_op;
    using namespace mshadow::expr;
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      Kernel<MapSetToScalar<Req>, xpu>::Launch(s, blob.Size(), blob.dptr<DType>(), val);
    });
  }

  /*! \brief Fill contiguous dense output rows with value computed from 0 lhs and 0 rhs input */
  template<typename xpu, typename DType, typename OP>
  static inline size_t FillDense(mshadow::Stream<xpu> *s,
                                 const size_t idx_l,
                                 const size_t idx_r,
                                 const OpReqType req,
                                 mshadow::Tensor<xpu, 2, DType> *out,
                                 const size_t iter_out) {
    using namespace mxnet_op;
    using namespace mshadow::expr;
    const int index_out_min = std::min(idx_l, idx_r);
    if (index_out_min > iter_out) {
      const size_t size = (*out)[iter_out].shape_.Size();
      const DType zero_input_val = OP::Map(DType(0), DType(0));
      #pragma omp parallel for
      for (int i = static_cast<int>(iter_out); i < index_out_min; ++i) {
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          Kernel<MapSetToScalar<Req>, xpu>::Launch(s, size, (*out)[i].dptr_, zero_input_val);
        });
      }
    }
    return static_cast<size_t>(index_out_min);
  }

  template<typename DType>
  static inline bool IsSameArray(const NDArray& a1, const NDArray& a2) {
    return a1.var() == a2.var();
  }


  // TODO(cjolivier01) Optimize: change some bool parameters to template arguments
  //                   in order to remove runtime checks for these invariant vars
  //                   (i.e. lhs_is_dense, rhs_is_dense, is_dense_result, etc.)
  template<typename DType, typename IType, typename OP>
  static void RspRspElemwiseBinaryOp(mshadow::Stream<cpu> *s,
                                     const nnvm::NodeAttrs &attrs,
                                     const OpContext &ctx,
                                     const NDArray& lhs,
                                     const NDArray& rhs,
                                     const OpReqType req,
                                     const NDArray& output,
                                     const bool lhs_may_be_dense,
                                     const bool rhs_may_be_dense,
                                     const bool allow_inplace) {
    using namespace mshadow;
    using namespace mxnet_op;
    using namespace mshadow::expr;

    const bool is_dense_result = output.storage_type() == kDefaultStorage;
    const bool lhs_is_dense = lhs.storage_type() == kDefaultStorage;
    const bool rhs_is_dense = rhs.storage_type() == kDefaultStorage;
    CHECK(!lhs_is_dense || lhs_may_be_dense) << "rvalue cannot be dense";
    CHECK(!rhs_is_dense || rhs_may_be_dense) << "rvalue cannot be dense";
    CHECK(!lhs_is_dense || !rhs_is_dense);
    if (rhs_is_dense) {
      // For right-side dense, lhs input zero should always output zero
      CHECK(fabs(static_cast<float>(OP::Map(0, 99))) < 1e-4f);
      CHECK(!is_dense_result);  // Currently not handled
    }
    if (lhs_is_dense) {
      // For right-side dense, lhs input zero should always output zero
      CHECK(fabs(static_cast<float>(OP::Map(99, 0))) < 1e-4f);
      CHECK(!is_dense_result);  // Currently not handled
    }

    // Memory Estimation: This is (roughly) the number of result rows. We still
    // need to subtract the number of common rows
    bool lhs_in_place = false, rhs_in_place = false;
    const size_t num_rows_l = lhs_is_dense ? lhs.shape()[0] : lhs.aux_shape(rowsparse::kIdx).Size();
    const size_t num_rows_r = rhs_is_dense ? rhs.shape()[0] : rhs.aux_shape(rowsparse::kIdx).Size();
    if (is_dense_result) {
      output.CheckAndAlloc();
    } else {
      if (rhs_is_dense) {
        output.CheckAndAlloc({mshadow::Shape1(num_rows_l)});
      } else if (lhs_is_dense) {
        output.CheckAndAlloc({mshadow::Shape1(num_rows_r)});
      } else {
        lhs_in_place = IsSameArray<DType>(lhs, output);
        rhs_in_place = IsSameArray<DType>(rhs, output);
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
        iter_out = FillDense<cpu, DType, OP>(s, all_rows, all_rows, req, &out, iter_out);
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
        iter_out = FillDense<cpu, DType, OP>(s, idx_l, idx_r, req, &out, iter_out);
        DCHECK_EQ(iter_out, std::min(idx_l, idx_r));
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
          Kernel<BMap<OP, Req>, cpu>::Launch(
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
          Kernel<BinaryOpMissingRValue<OP, Req>, cpu>::Launch(
            s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_);
        });
      } else {
        // Right only
        if (!is_dense_result) {
          indices_out[iter_out] = idx_r;
        }
        Tensor<cpu, 1, DType> rvalue = !rhs_is_dense ? data_r[iter_r++] : data_r[idx_r];
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          Kernel<BinaryOpMissingLValue<OP, Req>, cpu>::Launch(
            s, rvalue.shape_.Size(), out[iter_out].dptr_, rvalue.dptr_);
        });
      }
      iter_out++;
    }
    // Evaluate the remaining rows beyond the l and r value row intersetion
    while (iter_l < num_rows_l && !lhs_is_dense && !rhs_in_place) {
      if (!is_dense_result) {
        indices_out[iter_out] = indices_l[iter_l];
      } else {
        const IType idx_l = indices_l[iter_l];
        iter_out = FillDense<cpu, DType, OP>(s, lhs.shape()[0], idx_l, req, &out, iter_out);
      }
      Tensor<cpu, 1, DType> lvalue = data_l[iter_l++];
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        Kernel<BinaryOpMissingRValue<OP, Req>, cpu>::Launch(
          s, lvalue.shape_.Size(), out[iter_out++].dptr_, lvalue.dptr_);
      });
    }
    while (iter_r < num_rows_r && !rhs_is_dense && !lhs_in_place) {
      if (!is_dense_result) {
        indices_out[iter_out] = indices_r[iter_r];
      } else {
        const IType idx_r = indices_r[iter_r];
        iter_out = FillDense<cpu, DType, OP>(s, lhs.shape()[0], idx_r, req, &out, iter_out);
      }
      Tensor<cpu, 1, DType> rvalue = data_r[iter_r++];
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        Kernel<BinaryOpMissingLValue<OP, Req>, cpu>::Launch(
          s, rvalue.shape_.Size(), out[iter_out++].dptr_, rvalue.dptr_);
      });
    }
    if (is_dense_result) {
      const size_t all_rows = static_cast<size_t>(lhs.shape()[0]);
      iter_out = FillDense<cpu, DType, OP>(s, all_rows, all_rows, req, &out, iter_out);
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
      if (!rhs_is_dense && !lhs_is_dense && !lhs_in_place && !rhs_in_place) {
        // Reduce the first-dimension size by the number of common rows
        new_shape[0] -= num_common_rows;
        output.set_aux_shape(rowsparse::kIdx, new_shape);
      }
    }
  }

  /*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
  template<typename DType, typename IType, typename CType, typename OP>
  static inline void CsrCsrElemwiseBinaryOp(mshadow::Stream<cpu> *s,
                                            const nnvm::NodeAttrs &attrs,
                                            const OpContext &ctx,
                                            const NDArray& lhs,
                                            const NDArray& rhs,
                                            const OpReqType req,
                                            const NDArray& output) {
    using namespace mshadow;
    using namespace mxnet_op;
    using namespace mshadow::expr;

    const auto nr_rows = static_cast<size_t>(lhs.shape()[0]);
    if (!nr_rows) {
      return;
    }
    const size_t nr_cols = lhs.shape().Size() / nr_rows;

    CHECK_EQ(lhs.shape().Size(), rhs.shape().Size());

    const size_t lhs_nnz = lhs.storage_shape().Size();
    const size_t rhs_nnz = rhs.storage_shape().Size();

    output.CheckAndAlloc({mshadow::Shape1(lhs.shape()[0] + 1),
                          mshadow::Shape1(std::min(lhs_nnz + rhs_nnz, lhs.shape().Size()))});
    DCHECK_EQ(output.aux_shape(csr::kIndPtr), lhs.aux_shape(csr::kIndPtr));

    const size_t alloc_size = nr_cols * sizeof(IType) + 2 * nr_cols * sizeof(DType);

    mshadow::Tensor<cpu, 1, uint8_t> workspace =
      AllocateTempDataForSparseHandling<cpu, 1, uint8_t>(ctx, mshadow::Shape1(alloc_size));

    // Allocate temp space and partition into three tensors
    mshadow::Tensor<cpu, 1, IType> next(reinterpret_cast<IType *>(workspace.dptr_),
                                        Shape1(nr_cols));
    mshadow::Tensor<cpu, 1, DType> lhs_row(reinterpret_cast<DType *>(workspace.dptr_
                                                                     + nr_cols * sizeof(IType)),
                                           Shape1(nr_cols));
    mshadow::Tensor<cpu, 1, DType> rhs_row(lhs_row.dptr_ + nr_cols, Shape1(nr_cols));

    Fill<cpu, IType>(s, IType(-1), req, next);
    Fill<cpu, DType>(s, DType(0),  req, lhs_row);
    Fill<cpu, DType>(s, DType(0),  req, rhs_row);

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

    for (IType i = 0; i < nr_rows; i++) {
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

  /*! \brief Minimum of three */
  static MSHADOW_XINLINE size_t minthree(const size_t a, const size_t b, const size_t c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
  }

  /*! \brief Maximum of three */
  static MSHADOW_XINLINE size_t maxthree(const size_t a, const size_t b, const size_t c) {
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
  }

  /*! \brief LaunchEx allowing dense lvalue and/or rvalue */
  template<typename xpu, typename OP, typename DType,
    bool lhs_may_be_dense, bool rhs_may_be_dense, typename BackupCompute>
  static void LaunchExDenseLRValue_(const nnvm::NodeAttrs &attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs,
                                    BackupCompute backup_compute) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if (req[0] != kNullOp) {
      const NDArray *sparse = &inputs[0];
      if (sparse->storage_type() == kDefaultStorage) {
        sparse = &inputs[1];
        if (sparse->storage_type() == kDefaultStorage) {
          // Do we need to worry about sparse result here?
          CHECK_EQ(outputs[0].storage_type(), kDefaultStorage);
          MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Launch<xpu, OP>);
          return;
        }
      }
      bool allowed = false;
      if (lhs_may_be_dense && rhs_may_be_dense) {
        allowed = common::ContainsNonDefaultStorage(inputs);
      } else if (lhs_may_be_dense) {
        allowed = inputs[1].storage_type() != kDefaultStorage;
      } else if (rhs_may_be_dense) {
        allowed = inputs[0].storage_type() != kDefaultStorage;
      } else {
        allowed = !common::ContainsNonDefaultStorage(inputs);
      }
      if (allowed) {
        allowed = !common::ContainsStorage(inputs, kCSRStorage);
      }
      // If any input or output is dense, fallback to FCompute
      if (allowed) {
        mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
        MSHADOW_IDX_TYPE_SWITCH(sparse->aux_type(rowsparse::kIdx), IType, {
          RspRspElemwiseBinaryOp<DType, IType, OP>(
            s, attrs, ctx, inputs[0], inputs[1],
            req[0], outputs[0],
            lhs_may_be_dense, rhs_may_be_dense, false);
        });
      } else {
        // May be lhs=dense, rhs=sparse
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             backup_compute,
                             "LaunchExDenseLRValue_");
      }
    }
  }

  template<typename xpu, typename LOP, typename ROP, typename DType>
  static void BinaryBackwardUseNone_(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
                                / DataType<DType>::kLanes);
    const DType *ograd_dptr = inputs[0].dptr<DType>();
    if (std::is_same<LOP, mshadow_op::identity>::value && req[0] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, outputs[0].dptr<DType>());
    } else if (req[0] != kNullOp) {
      DType *lgrad_dptr = outputs[0].dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<BinaryOpBackwardUseNone<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr, ograd_dptr);
      });
    }
    if (std::is_same<ROP, mshadow_op::identity>::value && req[1] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, outputs[1].dptr<DType>());
    } else if (req[1] != kNullOp) {
      DType *rgrad_dptr = outputs[1].dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
        Kernel<BinaryOpBackwardUseNone<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr, ograd_dptr);
      });
    }
  }

  template<typename xpu, typename LOP, typename ROP, typename DType>
  static void BinaryBackwardUseIn_(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
    DCHECK_EQ(outputs.size(), 2U);
    DCHECK_EQ(inputs.size(), 3U);
    mxnet_op::Stream<xpu> *s = ctx.get_stream<xpu>();
    const DType *ograd_dptr = inputs[0].dptr<DType>();
    const DType *lhs_dptr = inputs[1].dptr<DType>();
    const DType *rhs_dptr = inputs[2].dptr<DType>();
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      const int size = static_cast<int>(
        (outputs[0].Size() + mxnet_op::DataType<DType>::kLanes - 1)
        / mxnet_op::DataType<DType>::kLanes);
      DType * lgrad_dptr = outputs[0].dptr<DType>();
      mxnet_op::Kernel<BinaryOpBackwardUseIn<LOP, Req>, xpu>::Launch(
        s, size, lgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);});
    MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
      const int size = static_cast<int>(
        (outputs[1].Size() + mxnet_op::DataType<DType>::kLanes - 1)
        / mxnet_op::DataType<DType>::kLanes);
      DType * rgrad_dptr = outputs[1].dptr<DType>();
      mxnet_op::Kernel<BinaryOpBackwardUseIn<ROP, Req>, xpu>::Launch(
        s, size, rgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);});
  }

  template<
    typename xpu,
    typename LOP,
    typename ROP,
    typename DType,
    bool in0_ok_dense = false,
    bool in1_ok_dense = false,
    bool in2_ok_dense = false,
    typename BackupCompute>
  static inline void BinaryBackwardUseInEx_(const nnvm::NodeAttrs &attrs,
                                           const OpContext &ctx,
                                           const std::vector<NDArray> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<NDArray> &outputs,
                                           BackupCompute backup_compute) {
    CHECK_EQ(inputs.size(), 3U);  // output grad,
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
    if (req[0] != kNullOp) {
      // If any input is dense, fallback to FCompute
      if (common::ContainsOnlyStorage(inputs, kRowSparseStorage)) {
        mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
        // ComputeRspRsp can handle dense outputs so long as OP(0, 0) == 0
        MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            RspRspElemwiseBinaryOp<DType, IType, LOP>(
              s, attrs, ctx, inputs[1], inputs[2], req[0], outputs[0],
              false, false, false);
        });
        // LHS in-place
        MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            RspRspElemwiseBinaryOp<DType, IType, mshadow::op::mul>(
              s, attrs, ctx, outputs[0], inputs[0], req[0], outputs[0],
              false, false, true);
        });
        MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            RspRspElemwiseBinaryOp<DType, IType, ROP>(
              s, attrs, ctx, inputs[1], inputs[2], req[1], outputs[1],
              false, false, false);
        });
        // RHS in-place
        MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            RspRspElemwiseBinaryOp<DType, IType, mshadow::op::mul>(
              s, attrs, ctx, inputs[0], outputs[1], req[1], outputs[1],
              false, false, true);
        });
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             backup_compute,
                             "BinaryBackwardUseInEx_");
      }
    }
  }

 public:
  template<typename xpu, typename OP>
  static void Launch(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    if (req[0] != kNullOp) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      CHECK_EQ(inputs.size(), 2U);
      CHECK_EQ(outputs.size(), 1U);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
          const size_t size = (minthree(outputs[0].Size(), inputs[0].Size(), inputs[1].Size())
          + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
          Kernel<BMap<OP, Req>, xpu>::Launch(s, size,
          outputs[0].dptr<DType>(),
          inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
        });
      });
    }
  }

  template<typename xpu, typename OP>
  static void LaunchWithHalf2(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    if (req[0] != kNullOp) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      CHECK_EQ(inputs.size(), 2U);
      CHECK_EQ(outputs.size(), 1U);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
          const size_t size = (minthree(outputs[0].Size(), inputs[0].Size(), inputs[1].Size())
          + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
          Kernel<BMap<OP, Req>, xpu>::Launch(s, size,
          outputs[0].dptr<DType>(),
          inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
        });
      });
    }
  }

  template<typename xpu, typename OP>
  static void LaunchEx(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<NDArray> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if (req[0] != kNullOp) {
      // If any input or output is dense, fallback to FCompute
      if (!common::ContainsDefaultStorage(inputs)
          && inputs[0].storage_type() == inputs[1].storage_type()) {
        mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
        switch (inputs[0].storage_type()) {
          case kRowSparseStorage:
            MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
              MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
                RspRspElemwiseBinaryOp<DType, IType, OP>(
                  s, attrs, ctx, inputs[0], inputs[1],
                  req[0], outputs[0],
                  false, false, false);
              });
            });
            break;
          case kCSRStorage:
            MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(csr::kIdx), IType, {
              MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(csr::kIndPtr), CType, {
                MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
                  CsrCsrElemwiseBinaryOp<DType, IType, CType, OP>(
                    s, attrs, ctx, inputs[0], inputs[1],
                    req[0], outputs[0]);
                });
              });
            });
            break;
          default:
            CHECK(false) << "Unsupported storage type for LaunchEx" << inputs[0].storage_type();
            break;
        }
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             Launch<xpu, OP>, "LaunchEx");
      }
    }
  }

  /*! \brief LaunchEx allowing dense lvalue and/or rvalue */
  template<typename xpu, typename OP, bool lhs_may_be_dense, bool rhs_may_be_dense>
  static void LaunchExDenseLRValue(const nnvm::NodeAttrs &attrs,
                                  const OpContext &ctx,
                                  const std::vector<NDArray> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<NDArray> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      LaunchExDenseLRValue_<xpu, OP, DType, lhs_may_be_dense, rhs_may_be_dense>(
        attrs, ctx, inputs, req, outputs, Launch<xpu, OP>);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNone(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BinaryBackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNoneWithHalf2(const nnvm::NodeAttrs &attrs,
                                                    const OpContext &ctx,
                                                    const std::vector<TBlob> &inputs,
                                                    const std::vector<OpReqType> &req,
                                                    const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      BinaryBackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNoneEx(const nnvm::NodeAttrs &attrs,
                                             const OpContext &ctx,
                                             const std::vector<NDArray> &inputs,
                                             const std::vector<OpReqType> &req,
                                             const std::vector<NDArray> &outputs) {
    CHECK_EQ(inputs.size(), 1U);   // output grad,
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[0] != kNullOp) {
      // If any input is dense, fallback to FCompute
      if (!common::ContainsDefaultStorage(inputs)) {
        CHECK_EQ(inputs[0].storage_type(), kRowSparseStorage);
        DCHECK_LT(fabs(static_cast<float>(LOP::Map(0))), 1e-5f);  // op requires 0-input
                                                                  // returns 0-output
        DCHECK_LT(fabs(static_cast<float>(ROP::Map(0))), 1e-5f);  // op requires 0-input
                                                                  // returns 0-output
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          UnaryOp::LaunchEx<xpu, BinaryOpBackwardUseNone<LOP, Req>>(attrs, ctx, inputs,
                                                                    req, {outputs[0]});
        });
        MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
          UnaryOp::LaunchEx<xpu, BinaryOpBackwardUseNone<ROP, Req>>(attrs, ctx, inputs,
                                                                    req, {outputs[1]});
        });
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             BinaryBackwardUseNone<xpu, LOP, ROP>,
                             "BinaryBackwardUseNoneEx");
      }
    }
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseIn(const nnvm::NodeAttrs &attrs,
                                         const OpContext &ctx,
                                         const std::vector<TBlob> &inputs,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BinaryBackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseInWithHalf2(const nnvm::NodeAttrs &attrs,
                                                  const OpContext &ctx,
                                                  const std::vector<TBlob> &inputs,
                                                  const std::vector<OpReqType> &req,
                                                  const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      BinaryBackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<
    typename xpu, typename LOP, typename ROP,
    bool in0_ok_dense = false, bool in1_ok_dense = false, bool in2_ok_dense = false>
  static inline void BinaryBackwardUseInEx(const nnvm::NodeAttrs &attrs,
                                           const OpContext &ctx,
                                           const std::vector<NDArray> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<NDArray> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      BinaryBackwardUseInEx_<xpu, LOP, ROP, DType, in0_ok_dense, in1_ok_dense, in2_ok_dense>(
        attrs, ctx, inputs, req, outputs, BinaryBackwardUseIn<xpu, LOP, ROP>);
    });
  }
};  // class ElemwiseBinaryOp

#define MXNET_OPERATOR_REGISTER_BINARY(name)                        \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FListInputNames>("FListInputNames",               \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::string>{"lhs", "rhs"};                \
    })                                                              \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
    })                                                              \
  .add_argument("lhs", "NDArray-or-Symbol", "first input")          \
  .add_argument("rhs", "NDArray-or-Symbol", "second input")

/*! \brief Binary launch */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(__name$, __kernel$)                \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                            \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>)       \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)     \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::LaunchEx<cpu, __kernel$>)

/*! \brief Binary launch, dense result */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(__name$, __kernel$)               \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                              \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageTypeDenseOutput<1>) \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)       \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::LaunchEx<cpu, __kernel$>)

/*! \brief Binary launch, dense rvalue */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DENSE_LRVALUE(__name$, __kernel$)           \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                                     \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageTypeLeastDense<2, 1>)      \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)              \
  .set_attr<FComputeEx>("FComputeEx<cpu>",                                                    \
    ElemwiseBinaryOp::LaunchExDenseLRValue<cpu, __kernel$, true, true>)

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
