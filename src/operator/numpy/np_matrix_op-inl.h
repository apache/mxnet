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
 *  Copyright (c) 2019 by Contributors
 * \file np_matrix_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_

#include <vector>
#include "../tensor/matrix_op-inl.h"
#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

struct NumpyTransposeParam : public dmlc::Parameter<NumpyTransposeParam> {
  mxnet::TShape axes;
  DMLC_DECLARE_PARAMETER(NumpyTransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(mxnet::TShape(-1, 0))
    .describe("By default, reverse the dimensions, otherwise permute "
              "the axes according to the values given.");
  }
};

template<typename xpu>
void NumpyTranspose(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "Transpose does not support inplace";
  if (ndim_is_known(param.axes)) {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], param.axes);
  } else {
    mxnet::TShape axes(inputs[0].ndim(), -1);
    for (int i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  }
}

template<typename xpu>
inline void HSplitOpForward(const nnvm::NodeAttrs &attrs,
                            const OpContext &ctx,
                            const std::vector<TBlob> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam &param = nnvm::get<SplitParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), param.sections > 0 ? param.sections : param.indices.ndim());
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob &input_data = inputs[split_enum::kData];
  size_t leading = 1, trailing = 1;
  int real_axis = 1;

  if (input_data.ndim() > 1) {
    real_axis = 1;
  } else {
    real_axis = 0;
  }

  CHECK_LT(real_axis, input_data.ndim());

  size_t mid = input_data.shape_[real_axis];
  for (int i = 0; i < real_axis; ++i) {
    leading *= input_data.shape_[i];
  }
  for (int i = real_axis + 1; i < input_data.ndim(); ++i) {
    trailing *= input_data.shape_[i];
  }

  size_t workspace_size = 0;
  const mxnet::TShape &ishape = input_data.shape_;
  const mxnet::TShape split_pts =
           param.sections > 0 ? GetSplitIndices(ishape, real_axis, param.sections) : param.indices;
  std::vector<size_t> indices;
  for (const auto &section : split_pts) {
    indices.push_back(section);
  }
  if (param.sections == 0) {
    indices.push_back(ishape[real_axis]);
  }
  workspace_size += indices.size() * sizeof(size_t);
  MSHADOW_TYPE_SWITCH(input_data.type_flag_, DType, {
    std::vector<DType *> output_data;
    for (const TBlob &data : outputs) {
      output_data.push_back(data.dptr<DType>());
    }
    workspace_size += output_data.size() * sizeof(DType * );
    Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size),
                                                       s);
    Tensor<cpu, 1, size_t>
        indices_cpu_tensor(indices.data(), Shape1(indices.size()));
    Tensor<xpu, 1, size_t> indices_xpu_tensor(
        reinterpret_cast<size_t *>(workspace.dptr_), Shape1(indices.size()));
    Tensor<cpu, 1, DType *>
        ptrs_cpu_tensor(output_data.data(), Shape1(output_data.size()));
    Tensor<xpu, 1, DType *> ptrs_xpu_tensor(
        reinterpret_cast<DType **>(workspace.dptr_
            + indices.size() * sizeof(size_t)),
        Shape1(output_data.size()));
    mshadow::Copy(indices_xpu_tensor, indices_cpu_tensor, s);
    mshadow::Copy(ptrs_xpu_tensor, ptrs_cpu_tensor, s);
    Kernel<SplitKernel, xpu>::Launch(
        s, input_data.Size(), input_data.dptr<DType>(), ptrs_xpu_tensor.dptr_,
        indices_xpu_tensor.dptr_, indices.size() - 1, mid, trailing);
  });
}

template<typename xpu>
inline void HSplitOpBackward(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam &param = nnvm::get<SplitParam>(attrs.parsed);
  CHECK_EQ(inputs.size(),
           (param.sections > 0) ? param.sections : param.indices.ndim())
    << "out grad vector size mush match the output size";
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  TBlob input_grad = outputs[split_enum::kData];
  size_t leading = 1, trailing = 1;

  int real_axis = 1;
  if (input_grad.ndim() > 1) {
    real_axis = 1;
  } else {
    real_axis = 0;
  }
  CHECK_LT(real_axis, input_grad.ndim());

  size_t mid = input_grad.shape_[real_axis];
  for (int i = 0; i < real_axis; ++i) {
    leading *= input_grad.shape_[i];
  }
  for (int i = real_axis + 1; i < input_grad.ndim(); ++i) {
    trailing *= input_grad.shape_[i];
  }

  size_t workspace_size = 0;
  const mxnet::TShape &ishape = input_grad.shape_;
  const mxnet::TShape split_pts =
      (param.sections > 0) ? GetSplitIndices(ishape, real_axis, param.sections)
                           : param.indices;
  std::vector<size_t> indices;
  for (const auto &section : split_pts) {
    indices.push_back(section);
  }
  if (param.sections == 0) {
    indices.push_back(ishape[real_axis]);
  }
  workspace_size += indices.size() * sizeof(size_t);
  MSHADOW_TYPE_SWITCH(input_grad.type_flag_, DType, {
    std::vector<DType *> out_grads;
    for (const TBlob &output_grad : inputs) {
      out_grads.push_back(output_grad.dptr<DType>());
    }
    workspace_size += out_grads.size() * sizeof(DType * );
    Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size),
                                                       s);
    Tensor<cpu, 1, size_t>
        indices_cpu_tensor(indices.data(), Shape1(indices.size()));
    Tensor<xpu, 1, size_t> indices_xpu_tensor(
        reinterpret_cast<size_t *>(workspace.dptr_), Shape1(indices.size()));
    Tensor<cpu, 1, DType *>
        ptrs_cpu_tensor(out_grads.data(), Shape1(inputs.size()));
    Tensor<xpu, 1, DType *> ptrs_xpu_tensor(
        reinterpret_cast<DType **>(workspace.dptr_
            + indices.size() * sizeof(size_t)),
        Shape1(inputs.size()));
    mshadow::Copy(indices_xpu_tensor, indices_cpu_tensor, s);
    mshadow::Copy(ptrs_xpu_tensor, ptrs_cpu_tensor, s);
    Kernel<ConcatenateKernel, xpu>::Launch(
        s, input_grad.Size(), ptrs_xpu_tensor.dptr_, input_grad.dptr<DType>(),
        indices_xpu_tensor.dptr_, indices.size() - 1, mid, trailing);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
