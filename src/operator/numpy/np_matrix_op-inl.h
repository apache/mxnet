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

struct NumpyRot90Param : public dmlc::Parameter<NumpyRot90Param> {
  int k;
  dmlc::optional<mxnet::TShape> axes;
  DMLC_DECLARE_PARAMETER(NumpyRot90Param) {
    DMLC_DECLARE_FIELD(k)
    .set_default(1)
    .describe("Number of times the array is rotated by 90 degrees.");
    DMLC_DECLARE_FIELD(axes)
    .set_default(dmlc::optional<mxnet::TShape>())
    .describe(" The array is rotated in the plane defined by the axes. Axes must be different.");
  }
};

struct rot90reverse {
  MSHADOW_XINLINE static index_t ReverseIndex(index_t idx,
                                              index_t nreversedim,
                                              const index_t * stride_,
                                              const index_t * trailing_) {
    index_t outputIndex = idx;
    for (index_t i = 0; i < nreversedim; ++i) {
      const index_t low = outputIndex % trailing_[i];
      index_t high = outputIndex / trailing_[i];
      const index_t x = high % stride_[i];
      high /= stride_[i];
      outputIndex = (high * stride_[i] + stride_[i] - 1 - x) * trailing_[i] + low;
    }
    return outputIndex;
  }
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t index, index_t nreversedim, const DType *src, DType *dst,
                                  const index_t * stride_,
                                  const index_t * trailing_) {
    index_t new_idx = ReverseIndex(index, nreversedim, stride_, trailing_);
    dst[new_idx] = src[index];
  }
};

template<typename xpu>
void NumpyRot90ComputeFlipIml(const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs,
                              const index_t axis0, const index_t axis1) {
  using namespace mshadow;
  using namespace mxnet_op;

  const mxnet::TShape& ishape = inputs[0].shape_;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  std::vector<index_t> stride_(2);
  std::vector<index_t>  trailing_(2);
  index_t reverse_index = 0;
  std::vector<index_t> temp{axis0, axis1};
  for (int axis : temp) {
    stride_[reverse_index] = ishape[axis];
    trailing_[reverse_index] = 1;
    for (int i2 = axis + 1; i2 < ishape.ndim(); ++i2) {
      trailing_[reverse_index] *= ishape[i2];
    }
    reverse_index++;
  }

  index_t workspace_size = 2 * sizeof(index_t);
  Tensor<xpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(2 * workspace_size), s);
  Tensor<cpu, 1, index_t> stride_cpu_tensor(stride_.data(), Shape1(stride_.size()));
  Tensor<xpu, 1, index_t> stride_xpu_tensor(
      reinterpret_cast<index_t*>(workspace.dptr_), Shape1(stride_.size()));
  Tensor<cpu, 1, index_t> trailing_cpu_tensor(trailing_.data(), Shape1(trailing_.size()));
  Tensor<xpu, 1, index_t> trailing_xpu_tensor(
      reinterpret_cast<index_t*>(workspace.dptr_ + workspace_size), Shape1(trailing_.size()));

  mshadow::Copy(stride_xpu_tensor, stride_cpu_tensor, s);
  mshadow::Copy(trailing_xpu_tensor, trailing_cpu_tensor, s);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<rot90reverse, xpu>::Launch(s, inputs[0].Size(), reverse_index,
                                      inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
                                      stride_xpu_tensor.dptr_, trailing_xpu_tensor.dptr_);
  });
}

struct rot90Transreverse {
  MSHADOW_XINLINE static index_t ReverseIndex(index_t idx,
                                              const index_t stride_,
                                              const index_t trailing_) {
    index_t outputIndex = idx;
    const index_t low = outputIndex % trailing_;
    index_t high = outputIndex / trailing_;
    const index_t x = high % stride_;
    high /= stride_;
    outputIndex = (high * stride_ + stride_ - 1 - x) * trailing_ + low;

    return outputIndex;
  }
  template<typename DType>
  MSHADOW_XINLINE  static void Map(index_t index, const DType *src, DType *dst,
                                   const index_t  stride_,
                                   const index_t  trailing_) {
    index_t new_idx = ReverseIndex(index, stride_, trailing_);
    dst[new_idx] = src[index];
  }
};

template<typename xpu>
void NumpyRot90ComputeFlipTransposeIml(const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs,
                                       const mxnet::TShape axes_list,
                                       const index_t axis) {
  using namespace mshadow;
  using namespace mxnet_op;

  const mxnet::TShape& ishape = inputs[0].shape_;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  index_t stride_;
  index_t trailing_;

  stride_ = ishape[axis];
  trailing_ = 1;
  for (int i2 = axis + 1; i2 < ishape.ndim(); ++i2) {
    trailing_ *= ishape[i2];
  }

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    index_t workspace_size = inputs[0].Size() * sizeof(DType);
    Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
    DType* data_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    TBlob mid_data = TBlob(data_ptr, inputs[0].shape_, xpu::kDevMask);
    Kernel<rot90Transreverse, xpu>::Launch(s, inputs[0].Size(), inputs[0].dptr<DType>(),
                                           mid_data.dptr<DType>(),
                                           stride_, trailing_);
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, mid_data, outputs[0], axes_list);
  });
}


template<typename xpu>
void NumpyRot90ComputeTransposeFlipIml(const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs,
                                       const mxnet::TShape axes_list,
                                       const index_t axis) {
  using namespace mshadow;
  using namespace mxnet_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    index_t workspace_size = inputs[0].Size() * sizeof(DType);
    Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
    DType* data_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    mxnet::TShape mid_shape(outputs[0].shape_);
    TBlob mid_data = TBlob(data_ptr, mid_shape, xpu::kDevMask);
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, inputs[0], mid_data, axes_list);

    index_t stride_;
    index_t trailing_;
    stride_ = mid_shape[axis];
    trailing_ = 1;
    for (int i2 = axis + 1; i2 < mid_shape.ndim(); ++i2) {
      trailing_ *= mid_shape[i2];
    }
    Kernel<rot90Transreverse, xpu>::Launch(s, mid_data.Size(), mid_data.dptr<DType>(),
                                           outputs[0].dptr<DType>(),
                                           stride_, trailing_);
  });
}

template<int req>
struct rot90 {
  template<typename DType>
  MSHADOW_XINLINE  static void Map(index_t i, const DType *in_data, DType *out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

template<typename xpu>
void NumpyRot90Compute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyRot90Param& param = nnvm::get<NumpyRot90Param>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  if (outputs[0].Size() == 0) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  int real_k(param.k);
  real_k = real_k % 4;
  if (real_k < 0) {
    real_k += 4;
  }

  // axis has value
  mxnet::TShape real_axes(param.axes.value());
  for (index_t i = 0; i < real_axes.ndim(); i++) {
    if (real_axes[i] < 0) {
      real_axes[i] += inputs[0].shape_.ndim();
    }
  }
  if (real_k == 0) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<rot90<req_type>, xpu>::Launch(s, inputs[0].Size(), inputs[0].dptr<DType>(),
                                             outputs[0].dptr<DType>());
      });
    });
  } else if (real_k == 2) {
    NumpyRot90ComputeFlipIml<xpu>(ctx, inputs, req, outputs, real_axes[0], real_axes[1]);
  } else if (real_k == 1) {
    mxnet::TShape axes_list(inputs[0].shape_.ndim(), -1);
    for (int i = 0; i < inputs[0].shape_.ndim(); ++i) {
      axes_list[i] = i;
    }
    axes_list[real_axes[0]] += axes_list[real_axes[1]];
    axes_list[real_axes[1]] = axes_list[real_axes[0]] - axes_list[real_axes[1]];
    axes_list[real_axes[0]] -= axes_list[real_axes[1]];
    NumpyRot90ComputeFlipTransposeIml<xpu>(ctx, inputs, req, outputs, axes_list, real_axes[1]);
  } else if (real_k == 3) {
    mxnet::TShape axes_list(inputs[0].shape_.ndim(), -1);
    for (int i = 0; i < inputs[0].shape_.ndim(); ++i) {
      axes_list[i] = i;
    }
    axes_list[real_axes[0]] += axes_list[real_axes[1]];
    axes_list[real_axes[1]] = axes_list[real_axes[0]] - axes_list[real_axes[1]];
    axes_list[real_axes[0]] -= axes_list[real_axes[1]];
    NumpyRot90ComputeTransposeFlipIml<xpu>(ctx, inputs, req, outputs, axes_list, real_axes[1]);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
