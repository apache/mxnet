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
 * Copyright (c) 2015 by Contributors
 * \file concat-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_NN_CONCAT_INL_H_
#define MXNET_OPERATOR_NN_CONCAT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../channel_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

namespace concat_enum {
enum ConcatOpInputs {kData0, kData1, kData2, kData3, kData4};
enum ConcatOpResource {kTempSpace};
enum ConcatOpOutputs {kOut};
}  // namespace concat_enum

struct ConcatParam : public dmlc::Parameter<ConcatParam> {
  int num_args;
  int dim;
  DMLC_DECLARE_PARAMETER(ConcatParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be concated.");
    DMLC_DECLARE_FIELD(dim).set_default(1)
    .describe("the dimension to be concated.");
  }
};  // struct ConcatParam

template<typename xpu, typename DType>
class ConcatOp {
 public:
  void Init(const ConcatParam &param) {
    this->size_ = param.num_args;
    this->dimension_ = param.dim;
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1U);
    int axis = CheckAxis(dimension_, in_data[concat_enum::kData0].ndim());
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3, DType> > data(size_);
    Tensor<xpu, 3, DType> out;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < axis; ++i) {
      leading *= out_data[concat_enum::kOut].shape_[i];
    }
    for (int i = axis + 1; i < out_data[concat_enum::kOut].ndim(); ++i) {
      trailing *= out_data[concat_enum::kOut].shape_[i];
    }
    size_t mid = out_data[concat_enum::kOut].shape_[axis];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    out = out_data[concat_enum::kOut].get_with_shape<xpu, 3, DType>(oshape, s);

    for (int i = 0; i < size_; ++i) {
      Shape<3> dshape = Shape3(leading, in_data[i].shape_[axis], trailing);
      data[i] = in_data[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Concatenate(data, &out, 1, req[concat_enum::kOut]);
  }

  void Backward(const OpContext &ctx, const TBlob &out_grad,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    int axis = CheckAxis(dimension_, out_grad.ndim());
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3, DType> > grad_in(size_);
    Tensor<xpu, 3, DType> grad;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < axis; ++i) {
      leading *= out_grad.shape_[i];
    }
    for (int i = axis + 1; i < out_grad.ndim(); ++i) {
      trailing *= out_grad.shape_[i];
    }
    size_t mid = out_grad.shape_[axis];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    grad = out_grad.get_with_shape<xpu, 3, DType>(oshape, s);

    for (int i = 0; i < size_; ++i) {
      Shape<3> dshape = Shape3(leading, in_grad[i].shape_[axis], trailing);
      grad_in[i] = in_grad[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Split(grad, &grad_in, 1, req);
  }

 private:
  int size_;
  int dimension_;
};  // class ConcatOp

template<typename xpu>
void ConcatCompute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(inputs[concat_enum::kData0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void HStackCompute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  ConcatParam param = nnvm::get<ConcatParam>(attrs.parsed);
  param.dim = inputs[0].shape_.ndim() > 1 ? 1 : 0;
  std::vector<TBlob> modified_inputs(inputs.size());
  for (int i = 0; i < param.num_args; ++i) {
    if (inputs[i].shape_.ndim() == 0) {
      modified_inputs[i] = inputs[i].reshape(TShape(1, 1));
    } else {
      modified_inputs[i] = inputs[i];
    }
  }
  MSHADOW_TYPE_SWITCH(inputs[concat_enum::kData0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, modified_inputs, req, outputs);
  });
}

template<typename xpu>
void ConcatGradCompute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(inputs[concat_enum::kOut].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, inputs[concat_enum::kOut], req, outputs);
  });
}

template<typename xpu>
void HStackGradCompute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  ConcatParam param = nnvm::get<ConcatParam>(attrs.parsed);
  param.dim = inputs[0].shape_.ndim() > 1 ? 1 : 0;
  std::vector<TBlob> modified_outputs(outputs.size());
  for (int i = 0; i < param.num_args; ++i) {
    if (outputs[i].shape_.ndim() == 0) {
      modified_outputs[i] = outputs[i].reshape(TShape(1, 1));
    } else {
      modified_outputs[i] = outputs[i];
    }
  }
  MSHADOW_TYPE_SWITCH(inputs[concat_enum::kOut].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, inputs[concat_enum::kOut], req, modified_outputs);
  });
}

/*!
 * \brief concat CSRNDArray on the first dimension.
 */
struct concat_csr_first_dim {
  /*!
   * \param i              the i-th row of the input ndarray
   * \param out_idx        output csr ndarray column indices
   * \param out_data       output csr ndarray data
   * \param out_indptr     output csr ndarray row index pointer
   * \param in_idx         input csr ndarray column indices
   * \param in_data        input csr ndarray data
   * \param in_indptr      input csr ndarray row index pointer
   * \param indptr_offset  offset for ouput ndarray row index pointer
   * \param idx_offset     offset for ouput ndarray column indices
   */
  template<typename DType, typename RType, typename IType>
  MSHADOW_XINLINE static void Map(int i, const OpReqType req,
                                  DType* out_data, const DType* in_data,
                                  RType* out_indptr, const RType* in_indptr,
                                  IType* out_idx, const IType* in_idx,
                                  const nnvm::dim_t indptr_offset,
                                  const nnvm::dim_t idx_offset) {
    if (i == 0) out_indptr[0] = 0;
    out_indptr[i+1+indptr_offset] = in_indptr[i+1] + idx_offset;
    for (nnvm::dim_t j = in_indptr[i]; j < in_indptr[i+1]; ++j) {
      KERNEL_ASSIGN(out_idx[j+idx_offset], req, in_idx[j]);
      KERNEL_ASSIGN(out_data[j+idx_offset], req, in_data[j]);
    }
  }
};

template<typename xpu>
void ConcatCSRImpl(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<NDArray>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  int num_args = param.num_args;
  int concat_dim = param.dim;
  CHECK_EQ(inputs.size(), num_args);
  CHECK_EQ(outputs.size(), 1);
  int axis = CheckAxis(concat_dim, inputs[0].shape().ndim());
  CHECK_EQ(axis, 0) << "concat of csr ndarrays on axis 1 is not supported.";
  if (req[0] == kNullOp) return;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  nnvm::dim_t nnz = 0;
  for (int i=0; i < num_args; i++) {
    nnz += inputs[i].aux_shape(kIdx)[0];
  }
  const NDArray& out = outputs[0];
  if (nnz == 0) {
    FillZerosCsrImpl(s, out);
    return;
  }
  const nnvm::dim_t num_rows = out.shape()[0];
  out.CheckAndAllocAuxData(kIndPtr, Shape1(num_rows+1));

  MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(kIdx), IType, {
      MSHADOW_TYPE_SWITCH(inputs[0].dtype(), DType, {
        RType* out_indptr = out.aux_data(kIndPtr).dptr<RType>();
        out.CheckAndAllocAuxData(kIdx, Shape1(nnz));
        out.CheckAndAllocData(Shape1(nnz));
        IType* out_idx = out.aux_data(kIdx).dptr<IType>();
        DType* out_data = out.data().dptr<DType>();
        nnvm::dim_t indptr_offset = 0;
        nnvm::dim_t idx_offset = 0;
        for (const auto& in : inputs) {
          const RType* in_indptr = in.aux_data(kIndPtr).dptr<RType>();
          const IType* in_idx = in.aux_data(kIdx).dptr<IType>();
          const DType* in_data = in.data().dptr<DType>();
          Kernel<concat_csr_first_dim, xpu>::Launch(s, in.shape()[0], req[0], out_data,
            in_data, out_indptr, in_indptr, out_idx, in_idx, indptr_offset, idx_offset);
          indptr_offset += in.shape()[0];
          idx_offset += in.aux_shape(kIdx)[0];
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CONCAT_INL_H_
