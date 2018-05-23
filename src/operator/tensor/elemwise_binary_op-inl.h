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

template<typename OP>
void ElemwiseBinaryOp::RspRspOp(mshadow::Stream<gpu> *s,
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
  LOG(FATAL) << "GPU not supported for RspRspOp";
}


/*! \brief binary op handling for the following row sparse inputs/outputs
  rsp, rsp -> rsp,
  dns, rsp -> rsp,
  rsp, dns -> rsp,
  dns, rsp -> dns,
  rsp, dns -> dns,
*/
template<typename OP>
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
  const NDArray& rsp = lhs.storage_type() == kRowSparseStorage ? lhs : rhs;
  const bool is_dense_result = output.storage_type() == kDefaultStorage;
  const bool lhs_is_dense = lhs.storage_type() == kDefaultStorage;
  const bool rhs_is_dense = rhs.storage_type() == kDefaultStorage;
  CHECK(!lhs_is_dense || lhs_may_be_dense) << "rvalue cannot be dense";
  CHECK(!rhs_is_dense || rhs_may_be_dense) << "rvalue cannot be dense";
  CHECK(!lhs_is_dense || !rhs_is_dense);
  MSHADOW_IDX_TYPE_SWITCH(rsp.aux_type(rowsparse::kIdx), IType, {
    MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
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
      const size_t num_rows_l = lhs_is_dense ? lhs.shape()[0] :
                                               lhs.aux_shape(rowsparse::kIdx).Size();
      const size_t num_rows_r = rhs_is_dense ? rhs.shape()[0] :
                                               rhs.aux_shape(rowsparse::kIdx).Size();
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
      const Tensor<cpu, 1, IType> indices_l = lhs_is_dense ?
                                              Tensor<cpu, 1, IType>() :
                                              lhs.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);
      const Tensor<cpu, 1, IType> indices_r = rhs_is_dense ?
                                              Tensor<cpu, 1, IType>() :
                                              rhs.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);
      Tensor<cpu, 1, IType> indices_out = is_dense_result ?
                                          Tensor<cpu, 1, IType>() :
                                          output.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);

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
    });
  });
}

/*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
template<typename OP>
void ElemwiseBinaryOp::CsrCsrOp(mshadow::Stream<gpu> *s,
                                const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const NDArray &lhs,
                                const NDArray &rhs,
                                const OpReqType req,
                                const NDArray &output) {
  LOG(FATAL) << "GPU not supported for CsrCsrOp";
}

/*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
template<typename OP>
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

  MSHADOW_IDX_TYPE_SWITCH(lhs.aux_type(csr::kIdx), IType, {
    MSHADOW_IDX_TYPE_SWITCH(lhs.aux_type(csr::kIndPtr), CType, {
      MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
        const size_t alloc_size = nr_cols * sizeof(IType) + 2 * nr_cols * sizeof(DType);

        Tensor<cpu, 1, uint8_t> workspace =
          ctx.requested[ResourceRequestType::kTempSpace].get_space_typed<cpu, 1, uint8_t>(
            mshadow::Shape1(alloc_size), s);

        // Allocate temp space and partition into three tensors
        mshadow::Tensor<cpu, 1, IType> next(reinterpret_cast<IType *>(workspace.dptr_),
                                            Shape1(nr_cols));
        mshadow::Tensor<cpu, 1, DType> lhs_row(reinterpret_cast<DType *>(
                                                 workspace.dptr_ + nr_cols * sizeof(IType)),
                                               Shape1(nr_cols));
        mshadow::Tensor<cpu, 1, DType> rhs_row;

        OpBase::FillDense<IType>(s, next.shape_.Size(), IType(-1), req, next.dptr_);
        OpBase::FillDense<DType>(s, lhs_row.shape_.Size(), DType(0),  req, lhs_row.dptr_);

        if (!same_lhs_rhs) {
          rhs_row = Tensor<cpu, 1, DType>(lhs_row.dptr_ + nr_cols, Shape1(nr_cols));
          OpBase::FillDense<DType>(s, rhs_row.shape_.Size(), DType(0), req, rhs_row.dptr_);
        } else {
          rhs_row = lhs_row;
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

          if (!same_lhs_rhs) {
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
            if (!same_lhs_rhs) rhs_row[temp] = 0;
          }

          row_ptr_out[i + 1] = nnz;
        }
      });
    });
  });
}

/*!
 * \brief Kernel for performing elemwise op between dense and csr matrix
 * \param i            global thread id
 * \param req          type of request
 * \param out          output array
 * \param dns_data     data array of dense input
 * \param csr_data     data array of csr input
 * \param csr_indices  indices array of csr input
 * \param csr_indptr   indptr array of csr input
 * \param num_rows     number of rows of both inputs
 * \param num_cols     number of columns of both inputs
 */
template<int req, typename OP>
struct ElemwiseDnsCsrDnsKernel {
  template<typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out, DType* dns_data,
                                  const DType* csr_data, const IType* csr_indices,
                                  const CType* csr_indptr, const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    if (i < num_rows) {
      for (int j = csr_indptr[i]; j < csr_indptr[i+1]; ++j) {
        KERNEL_ASSIGN(out[i * num_cols + csr_indices[j]], req,
                      OP::Map(dns_data[i * num_cols + csr_indices[j]], csr_data[j]));
      }
    }
  }
};

/*! \brief DNS -op- CSR binary operator for non-canonical NDArray */
template<typename xpu, typename OP>
void ElemwiseBinaryOp::DnsCsrDnsOp(mshadow::Stream<xpu> *s,
                                   const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const NDArray &dns,
                                   const NDArray &csr,
                                   const OpReqType req,
                                   const NDArray &output,
                                   const bool reverse) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(dns.storage_type(), kDefaultStorage);
  CHECK_EQ(csr.storage_type(), kCSRStorage);
  CHECK(req != kAddTo);
  CHECK(req != kNullOp);
  const bool supported_op = std::is_same<OP, mshadow_op::minus>::value ||
                            std::is_same<OP, mshadow_op::plus>::value;
  CHECK(supported_op == true);
  const nnvm::dim_t num_csr_rows = csr.shape()[0];
  const nnvm::dim_t num_csr_cols = csr.shape()[1];
  TBlob csr_data = csr.data();
  TBlob csr_indices = csr.aux_data(csr::kIdx);
  TBlob csr_indptr = csr.aux_data(csr::kIndPtr);
  MSHADOW_SGL_DBL_TYPE_SWITCH(csr_data.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(csr_indices.type_flag_, IType, {
      MSHADOW_IDX_TYPE_SWITCH(csr_indptr.type_flag_, CType, {
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          if (reverse && std::is_same<OP, mshadow_op::minus>::value) {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::negation, Req>, xpu>::Launch(
              s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
            if (!csr.storage_initialized()) { return; }
            mxnet_op::Kernel<ElemwiseDnsCsrDnsKernel<Req, mshadow_op::plus>, xpu>::Launch(
              s, num_csr_rows, output.data().dptr<DType>(),
              output.data().dptr<DType>(), csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
              csr_indptr.dptr<CType>(), num_csr_rows, num_csr_cols);
          } else {
            if (req == kWriteTo) {
              mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
                s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
            }
            if (!csr.storage_initialized()) { return; }
            mxnet_op::Kernel<ElemwiseDnsCsrDnsKernel<Req, OP>, xpu>::Launch(
              s, num_csr_rows, output.data().dptr<DType>(),
              output.data().dptr<DType>(), csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
              csr_indptr.dptr<CType>(), num_csr_rows, num_csr_cols);
          }
        });
      });
    });
  });
}

/*!
 * \brief Kernel for performing elemwise op between dense and rsp tensor
 * \param i            global thread id
 * \param req          type of request
 * \param out          output array
 * \param dns_data     data array of dense input
 * \param rsp_data     data array of rsp input
 * \param rsp_indices  indices array of rsp input
 * \param num_rows     number of rows of both inputs
 * \param nz_rows      number of non-zero rows of rsp tensor
 * \param num_cols     number of columns of both inputs
 */
template<int req, typename OP>
struct ElemwiseDnsRspDnsKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out, DType* dns_data,
                                  const DType* rsp_data, const IType* rsp_indices,
                                  const nnvm::dim_t num_rows, const nnvm::dim_t nz_rows,
                                  const nnvm::dim_t num_cols) {
    if (i < nz_rows * num_cols) {
      const nnvm::dim_t rsp_idx = i / num_cols;
      const nnvm::dim_t dns_row = rsp_indices[rsp_idx];
      const nnvm::dim_t col = i % num_cols;
      KERNEL_ASSIGN(out[dns_row * num_cols + col], req,
                    OP::Map(dns_data[dns_row * num_cols + col],
                            rsp_data[rsp_idx * num_cols + col]));
    }
  }
};

/*! \brief DNS -op- RSP binary operator for non-canonical NDArray */
template<typename xpu, typename OP>
void ElemwiseBinaryOp::DnsRspDnsOp(mshadow::Stream<xpu> *s,
                                   const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const NDArray &dns,
                                   const NDArray &rsp,
                                   const OpReqType req,
                                   const NDArray &output,
                                   const bool reverse) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(dns.storage_type(), kDefaultStorage);
  CHECK_EQ(rsp.storage_type(), kRowSparseStorage);
  CHECK_EQ(output.data().Size(), dns.data().Size());
  CHECK(req != kAddTo);
  if (req == kNullOp) return;
  const bool supported_op = std::is_same<OP, mshadow_op::minus>::value ||
                            std::is_same<OP, mshadow_op::plus>::value;
  CHECK(supported_op == true) <<
    "Only plus and minus supported now for elemwise operation between default and rsp matrices";
  const nnvm::dim_t num_rows = dns.shape()[0];
  const nnvm::dim_t num_cols = dns.data().Size() / num_rows;
  const nnvm::dim_t nz_rows = rsp.aux_shape(rowsparse::kIdx).Size();
  TBlob rsp_data = rsp.data();
  TBlob rsp_indices = rsp.aux_data(rowsparse::kIdx);

  MSHADOW_SGL_DBL_TYPE_SWITCH(rsp_data.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(rsp_indices.type_flag_, IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        if (reverse && std::is_same<OP, mshadow_op::minus>::value) {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::negation, Req>, xpu>::Launch(
            s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
          if (rsp.storage_initialized()) {
            mxnet_op::Kernel<ElemwiseDnsRspDnsKernel<Req, mshadow_op::plus>, xpu>::Launch(
              s, nz_rows * num_cols, output.data().dptr<DType>(),
              output.data().dptr<DType>(), rsp_data.dptr<DType>(), rsp_indices.dptr<IType>(),
              num_rows, nz_rows, num_cols);
          }
        } else {
          if (req == kWriteTo) {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
              s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
          }
          if (rsp.storage_initialized()) {
            mxnet_op::Kernel<ElemwiseDnsRspDnsKernel<Req, OP>, xpu>::Launch(
              s, nz_rows * num_cols, output.data().dptr<DType>(),
              output.data().dptr<DType>(), rsp_data.dptr<DType>(), rsp_indices.dptr<IType>(),
              num_rows, nz_rows, num_cols);
          }
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_H_
