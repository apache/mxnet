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
 * \file dot.cc
 * \brief CPU Implementation of matrix dot
 */

#include "./dot-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DotParam);

/*
 * \brief CPU Impl of dot(dns, csr) = csr
 */
template<typename cpu>
inline void DotDnsCsrCsrImpl(const OpContext& ctx,
                             const TBlob& lhs, const NDArray& rhs,
                             const OpReqType req, NDArray* ret) {
  if (kNullOp == req) return;

  CHECK_EQ(req, kWriteTo);
  CHECK_EQ(rhs.storage_type(), kCSRStorage);

  using namespace mshadow;
  using namespace mshadow::expr;
  using nnvm::dim_t;

  /* Initialize data structures */
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const NDArray& out = *ret;
  const TBlob data_l = lhs;
  const TBlob data_r = rhs.data();
  const TBlob indptr_r = rhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_r = rhs.aux_data(csr::kIdx);
  if (!rhs.storage_initialized()) {
    FillZerosCsrImpl(s, *ret);
    return;
  }

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_r.type_flag_, DType, {     // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_r.type_flag_, IType, {     // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_r.type_flag_, CType, {  // colidx type
        /* Allocate workspace */
        CType num_cols_out = out.shape()[1];
        CType rhs_data_size = static_cast<CType>(col_idx_r.shape_.Size());
        size_t workspace_size = 2 * num_cols_out * sizeof(CType);
        Tensor<cpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<cpu, 1, char>(
                Shape1(workspace_size), s);
        CType* col_flg = reinterpret_cast<dim_t*>(workspace.dptr_);

        CType* prefix_sum = col_flg;
        CType* nnc_idx = prefix_sum + num_cols_out;

        /* Set the column flags for nnz columns */
        mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, num_cols_out,
                                                          col_flg);
        mxnet_op::Kernel<MarkRowFlgKernel, cpu>::Launch(
            s, rhs_data_size, col_flg, col_idx_r.dptr<CType>());

        /* 1. Calculate prefix sum from col flgs
         * 2. Storage all non zero column indexes in nnc_idx
         */
        CType cur = 0;
        prefix_sum[0] = col_flg[0];
        if (prefix_sum[0]) nnc_idx[cur++] = 0;
        for (CType i = 1; i < num_cols_out; i++) {
          prefix_sum[i] = prefix_sum[i - 1] + col_flg[i];
          if (prefix_sum[i] > prefix_sum[i - 1]) nnc_idx[cur++] = i;
        }

        /* Allocate aux data for out */
        IType num_rows_l = lhs.shape_[0];
        dim_t nnc = prefix_sum[num_cols_out - 1];
        dim_t nnz = nnc * num_rows_l;
        out.CheckAndAllocAuxData(csr::kIndPtr, Shape1(num_rows_l + 1));
        out.CheckAndAllocAuxData(csr::kIdx, Shape1(nnz));
        out.CheckAndAllocData(Shape1(nnz));

        /* Set csr indptr and index according to nnc_idx*/
        IType* indptr_out = out.aux_data(csr::kIndPtr).dptr<IType>();
        CType* col_idx_out = out.aux_data(csr::kIdx).dptr<CType>();
        DType* data_out = out.data().dptr<DType>();
        mxnet_op::Kernel<PopulateCsrForNNC, cpu>::Launch(
            s, num_rows_l, nnc_idx, indptr_out, col_idx_out, nnc, num_rows_l);
        mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, nnz, data_out);

        const dim_t num_threads = mxnet_op::get_num_threads<cpu>(num_rows_l);
        const dim_t seg_len = (num_rows_l + num_threads - 1) / num_threads;

        IType num_rows_r = rhs.shape()[0];
        mxnet_op::Kernel<DotDnsCsrCsrByRowBlocks, cpu>::Launch(
            s, num_threads, data_out, data_l.dptr<DType>(),
            indptr_r.dptr<IType>(), col_idx_r.dptr<CType>(),
            data_r.dptr<DType>(), seg_len, num_rows_r, num_rows_l, num_cols_out,
            nnc, prefix_sum);
      });
    });
  });
}


template<typename cpu>
inline void DotDnsCsrDnsImpl(const OpContext& ctx,
                             const TBlob& dns, const NDArray& rhs,
                             const OpReqType req, NDArray* ret,
                             const bool transpose_b) {
  LOG(FATAL) << "dot(dense, csr) = dense is not implemented on CPU";
}

NNVM_REGISTER_OP(dot)
.add_alias("_sparse_dot")  // alias for op registration under mxnet.ndarray.sparse
.describe(R"doc(Dot product of two arrays.

``dot``'s behavior depends on the input array dimensions:

- 1-D arrays: inner product of vectors
- 2-D arrays: matrix multiplication
- N-D arrays: a sum product over the last axis of the first input and the first
  axis of the second input

  For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
  result array will have shape `(n,m,r,s)`. It is computed by::

    dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])

  Example::

    x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
    y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
    dot(x,y)[0,0,1,1] = 0
    sum(x[0,0,:]*y[:,1,1]) = 0

The storage type of ``dot`` output depends on storage types of inputs, transpose options and given
hint for output storage type:

Implemented sprase operations include:
- dot(csr, default) = default
- dot(csr.T, default) = row_sparse
- dot(csr, row_sparse) = default
- dot(default, csr) = csr on CPU only
- dot(default, csr) = dense on GPU only
- dot(default, csr.T) = dense on GPU only
- otherwise, ``dot`` generates output with default storage

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", DotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", DotForwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", DotForward_<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", DotForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_dot"})
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", DotBackwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", DotBackward_<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", DotBackwardEx<cpu>)
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(batch_dot)
.describe(R"doc(Batchwise dot product.

``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.

For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
`(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
which is computed by::

   batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", BatchDotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BatchDotForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_batch_dot"})
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_batch_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", BatchDotBackward_<cpu>);

}  // namespace op
}  // namespace mxnet
