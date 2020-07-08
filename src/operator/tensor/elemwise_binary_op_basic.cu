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
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_op_basic.cu
 * \brief GPU Implementation of basic elementwise binary broadcast operators
 */
#include <cub/cub.cuh>
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_op-inl.h"
#include "./indexing_op.h"

namespace mxnet {
namespace op {

template<typename OP>
struct RspElemwiseKernel {
  template<typename DType, typename IType>
  static MSHADOW_XINLINE void Map(int i, DType* out, const IType* lookup_table,
                                  const DType* data, const IType* indices,
                                  const nnvm::dim_t nz_rows, const nnvm::dim_t num_cols) {
    if (i < nz_rows * num_cols) {
      const nnvm::dim_t row = i / num_cols;
      const nnvm::dim_t col = i % num_cols;
      const nnvm::dim_t out_row = lookup_table[indices[row]] - 1;
      const nnvm::dim_t out_idx = out_row * num_cols + col;
      out[out_idx] = OP::Map(out[out_idx], data[i]);
    }
  }
};

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
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  using namespace rowsparse;

  if (req == kNullOp) return;

  CHECK(!scatter) << "scatter is not supported in RspRspOp on GPU yet...";
  CHECK(lhs.storage_type() == kRowSparseStorage && rhs.storage_type() == kRowSparseStorage);
  CHECK(output.storage_type() == kRowSparseStorage);
  CHECK(req != kAddTo);

  const nnvm::dim_t num_rows = output.shape()[0];
  MSHADOW_TYPE_SWITCH(lhs.data().type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(lhs.aux_data(kIdx).type_flag_, IType, {
      if (lhs.storage_initialized() && rhs.storage_initialized()) {
        const nnvm::dim_t lhs_nz_rows = lhs.storage_shape()[0];
        const nnvm::dim_t rhs_nz_rows = rhs.storage_shape()[0];
        const nnvm::dim_t num_cols = lhs.data().Size() / lhs_nz_rows;
        // Optimize for the case where one of the rsps is actually dense
        if ((lhs_nz_rows == num_rows || rhs_nz_rows == num_rows) && req == kWriteInplace) {
          const NDArray& dns = (output.IsSame(lhs)) ? lhs : rhs;
          const NDArray& rsp = (output.IsSame(lhs)) ? rhs : lhs;
          const bool reverse = !(lhs_nz_rows == num_rows);
          ElemwiseBinaryOp::DnsRspDnsOp<gpu, OP>(s, attrs, ctx, dns, rsp, req, output, reverse);
          return;
        }
        CHECK(req == kWriteTo) << "Should be kWriteTo but got " << req;
        const TBlob& lhs_indices = lhs.aux_data(kIdx);
        const TBlob& rhs_indices = rhs.aux_data(kIdx);
        size_t common_row_table_bytes = num_rows * sizeof(IType);
        IType* common_row_table = nullptr;
        void* temp_storage_ptr = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(temp_storage_ptr,
                                      temp_storage_bytes,
                                      common_row_table,
                                      common_row_table,
                                      num_rows,
                                      mshadow::Stream<gpu>::GetStream(s));
        size_t workspace_bytes = common_row_table_bytes + temp_storage_bytes;
        Tensor<gpu, 1, char> workspace =
          ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_bytes), s);
        common_row_table = reinterpret_cast<IType*>(workspace.dptr_);
        temp_storage_ptr = workspace.dptr_ + common_row_table_bytes;
        mxnet_op::Kernel<set_zero, gpu>::Launch(s, num_rows, common_row_table);
        Kernel<MarkRspRowFlgKernel, gpu>::Launch(
          s, lhs_nz_rows, common_row_table, lhs_indices.dptr<IType>(), lhs_nz_rows);
        Kernel<MarkRspRowFlgKernel, gpu>::Launch(
          s, rhs_nz_rows, common_row_table, rhs_indices.dptr<IType>(), rhs_nz_rows);
        cub::DeviceScan::InclusiveSum(temp_storage_ptr,
                                      temp_storage_bytes,
                                      common_row_table,
                                      common_row_table,
                                      num_rows,
                                      mshadow::Stream<gpu>::GetStream(s));
        nnvm::dim_t nnr_out = 0;
        CUDA_CALL(cudaMemcpyAsync(&nnr_out, &common_row_table[num_rows-1], sizeof(nnvm::dim_t),
                                  cudaMemcpyDeviceToHost, mshadow::Stream<gpu>::GetStream(s)));
        CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)))
        output.CheckAndAlloc({mshadow::Shape1(nnr_out)});
        Kernel<FillRspRowIdxKernel, gpu>::Launch(
          s, num_rows, output.aux_data(kIdx).dptr<IType>(), common_row_table, num_rows);
        Kernel<set_zero, gpu>::Launch(s, nnr_out * num_cols, output.data().dptr<DType>());
        Kernel<RspElemwiseKernel<mshadow_op::plus>, gpu>::Launch(
          s, lhs_nz_rows * num_cols, output.data().dptr<DType>(), common_row_table,
          lhs.data().dptr<DType>(), lhs_indices.dptr<IType>(), lhs_nz_rows, num_cols);
        Kernel<RspElemwiseKernel<OP>, gpu>::Launch(
          s, rhs_nz_rows * num_cols, output.data().dptr<DType>(), common_row_table,
          rhs.data().dptr<DType>(), rhs_indices.dptr<IType>(), rhs_nz_rows, num_cols);
      } else {
        if (lhs.storage_initialized()) {
          if (req == kWriteTo) {
            output.CheckAndAlloc({lhs.aux_shape(kIdx)});
            Copy(output.data().FlatTo1D<gpu, DType>(),
                 lhs.data().FlatTo1D<gpu, DType>(), s);
            Copy(output.aux_data(kIdx).FlatTo1D<gpu, IType>(),
                 lhs.aux_data(kIdx).FlatTo1D<gpu, IType>(), s);
          } else if (req == kWriteInplace && rhs.IsSame(output)) {
            LOG(FATAL) << "Inplace on an empty rhs is not supported";
          }
        } else if (rhs.storage_initialized()) {
          if (req == kWriteTo) {
            output.CheckAndAlloc({rhs.aux_shape(kIdx)});
          } else if (req == kWriteInplace && lhs.IsSame(output)) {
            LOG(FATAL) << "Inplace on an empty lhs is not supported";
          }
          if (std::is_same<OP, mshadow_op::minus>::value) {
            Kernel<op_with_req<mshadow_op::negation, kWriteTo>, gpu>::Launch(
              s, rhs.data().Size(), output.data().dptr<DType>(), rhs.data().dptr<DType>());
          } else if (req == kWriteTo) {
            Copy(output.data().FlatTo1D<gpu, DType>(),
                 rhs.data().FlatTo1D<gpu, DType>(), s);
          }
          if (req == kWriteTo) {
            Copy(output.aux_data(kIdx).FlatTo1D<gpu, IType>(),
                 rhs.aux_data(kIdx).FlatTo1D<gpu, IType>(), s);
          }
        } else {
          FillZerosRspImpl(s, output);
        }
      }
    });
  });
}

/*! \brief DNS -op- CSR binary operator for non-canonical NDArray */
template<typename OP>
void ElemwiseBinaryOp::DnsCsrDnsOp(mshadow::Stream<gpu> *s,
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
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::negation, Req>, gpu>::Launch(
              s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
            if (!csr.storage_initialized()) { return; }
            mxnet_op::Kernel<ElemwiseDnsCsrDnsWarpKernel<Req, mshadow_op::plus>, gpu>::Launch(
              s, kWarpSize * num_csr_rows, output.data().dptr<DType>(),
              output.data().dptr<DType>(), csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
              csr_indptr.dptr<CType>(), num_csr_rows, num_csr_cols);
          } else {
            if (req == kWriteTo) {
              mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, gpu>::Launch(
                s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
            }
            if (!csr.storage_initialized()) { return; }
            mxnet_op::Kernel<ElemwiseDnsCsrDnsWarpKernel<Req, OP>, gpu>::Launch(
              s, kWarpSize * num_csr_rows, output.data().dptr<DType>(),
              output.data().dptr<DType>(), csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
              csr_indptr.dptr<CType>(), num_csr_rows, num_csr_cols);
          }
        });
      });
    });
  });
}

NNVM_REGISTER_OP(elemwise_add)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, op::mshadow_op::plus>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::ComputeEx<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_grad_add)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, op::mshadow_op::plus>);

NNVM_REGISTER_OP(_backward_add)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseNoneWithHalf2<gpu, mshadow_op::identity,
                    mshadow_op::identity>);

NNVM_REGISTER_OP(elemwise_sub)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<
  gpu, op::mshadow_op::minus>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::ComputeEx<gpu, op::mshadow_op::minus>);

NNVM_REGISTER_OP(_backward_sub)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseNoneWithHalf2<gpu, mshadow_op::identity,
                    mshadow_op::negation>);

NNVM_REGISTER_OP(elemwise_mul)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, op::mshadow_op::mul>)
.set_attr<FComputeEx>("FComputeEx<gpu>",
  ElemwiseBinaryOp::ComputeDnsLRValueEx<gpu, op::mshadow_op::mul, true, true>);

NNVM_REGISTER_OP(_backward_mul)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::right,
                    mshadow_op::left>);

NNVM_REGISTER_OP(elemwise_div)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::ElemwiseBinaryOp::ComputeWithHalf2<gpu, op::mshadow_op::div>);

NNVM_REGISTER_OP(_backward_div)
.set_attr<FCompute>("FCompute<gpu>",
                    ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::div_grad,
                    mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(_mod)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::ComputeWithHalf2<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_mod)
.set_attr<FCompute>("FCompute<gpu>",
  ElemwiseBinaryOp::BackwardUseInWithHalf2<gpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
