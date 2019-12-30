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
 * Copyright (c) 2019 by Contributors
 * \file np_normal_op.h
 * \brief Operator for numpy sampling from normal distributions
 */
#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_NORMAL_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_NORMAL_OP_H_

#include <mxnet/operator_util.h>
#include <cstdio>
#include <algorithm>
#include <string>
#include <vector>
#include "../../elemwise_op_common.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "./dist_common.h"

namespace mxnet {
namespace op {

struct NumpyNormalParam : public dmlc::Parameter<NumpyNormalParam> {
  dmlc::optional<float> loc;
  dmlc::optional<float> scale;
  std::string ctx;
  int dtype;
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyNormalParam) {
    DMLC_DECLARE_FIELD(loc);
    DMLC_DECLARE_FIELD(scale);
    DMLC_DECLARE_FIELD(size)
        .set_default(dmlc::optional<mxnet::Tuple<int>>())
        .describe(
            "Output shape. If the given shape is, "
            "e.g., (m, n, k), then m * n * k samples are drawn. "
            "Default is None, in which case a single value is returned.");
    DMLC_DECLARE_FIELD(ctx).set_default("cpu").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .set_default(mshadow::kFloat32)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to float32 if not defined (dtype=None).");
  }
};

inline bool NumpyNormalOpType(const nnvm::NodeAttrs &attrs,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  const NumpyNormalParam &param = nnvm::get<NumpyNormalParam>(attrs.parsed);
  int otype = param.dtype;
  if (otype != -1) {
    (*out_attrs)[0] = otype;
  } else {
    (*out_attrs)[0] = mshadow::kFloat32;
  }
  (*out_attrs)[1] = mshadow::kFloat32;
  return true;
}

namespace mxnet_op {
template <int ndim, typename IType, typename OType>
struct normal_kernel {
  MSHADOW_XINLINE static void Map(index_t i, const Shape<ndim> &lstride,
                                  const Shape<ndim> &hstride,
                                  const Shape<ndim> &oshape, IType *loc,
                                  IType *scale, float *normals, OType *out) {
    Shape<ndim> coord = unravel(i, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto hidx = static_cast<index_t>(dot(coord, hstride));
    IType loc_value = loc[lidx];
    IType scale_value = scale[hidx];
    out[i] = loc_value + normals[i] * scale_value;
  }
};

template <int ndim, typename IType, typename OType>
struct normal_one_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, int scalar_pos,
                                  const Shape<ndim> &stride,
                                  const Shape<ndim> &oshape, IType *array,
                                  float scalar, float *normals, OType *out) {
    Shape<ndim> coord = unravel(i, oshape);
    auto idx = static_cast<index_t>(dot(coord, stride));
    IType loc_value;
    IType scale_value;
    if (scalar_pos == 0) {
      loc_value = scalar;
      scale_value = array[idx];
    } else {
      loc_value = array[idx];
      scale_value = scalar;
    }
    out[i] = loc_value + normals[i] * scale_value;
  }
};

template <typename OType>
struct normal_two_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float loc, float scale,
                                  float *normals, OType *out) {
    out[i] = loc + normals[i] * scale;
  }
};

template <typename IType>
struct check_legal_scale_kernel {
  MSHADOW_XINLINE static void Map(index_t i, IType *scalar, float* flag) {
    if (scalar[i] < 0) {
      *flag = -1.0;
    }
  }
};

}  // namespace mxnet_op

template <typename xpu>
void NumpyNormalForward(const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyNormalParam &param = nnvm::get<NumpyNormalParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Generate base random number.
  Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  Tensor<xpu, 1, float> workspace =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(1), s);
  Tensor<xpu, 1, float> normal_tensor = outputs[1].FlatTo1D<xpu, float>(s);
  Tensor<xpu, 1, float> indicator_device = workspace;
  float indicator_host = 1.0;
  float *indicator_device_ptr = indicator_device.dptr_;
  Kernel<set_zero, xpu>::Launch(s, 1, indicator_device_ptr);
  prnd->SampleGaussian(&normal_tensor, 0.0, 1.0);
  mxnet::TShape new_lshape, new_hshape, new_oshape;
  // [scalar scalar] case
  if (inputs.size() == 0U) {
    CHECK_GE(param.scale.value(), 0.0) << "ValueError: scale < 0";
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      Kernel<normal_two_scalar_kernel<OType>, xpu>::Launch(
          s, outputs[0].Size(), param.loc.value(), param.scale.value(),
          normal_tensor.dptr_, outputs[0].dptr<OType>());
    });
  } else if (inputs.size() == 1U) {
    // [scalar tensor], [tensor scalar] case
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    int scalar_pos;
    float scalar_value;
    if (param.loc.has_value()) {
      scalar_pos = 0;
      scalar_value = param.loc.value();
      MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
        Kernel<check_legal_scale_kernel<IType>, xpu>::Launch(
            s, inputs[0].Size(), inputs[0].dptr<IType>(), indicator_device_ptr);
      });
      _copy<xpu>(s, &indicator_host, indicator_device_ptr);
      CHECK_GE(indicator_host, 0.0) << "ValueError: scale < 0";
    } else {
      scalar_pos = 1;
      scalar_value = param.scale.value();
      CHECK_GE(scalar_value, 0.0) << "ValueError: scale < 0";
    }
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> oshape = new_oshape.get<NDim>();
          Shape<NDim> stride = calc_stride(new_lshape.get<NDim>());
          Kernel<normal_one_scalar_kernel<NDim, IType, OType>, xpu>::Launch(
              s, outputs[0].Size(), scalar_pos, stride, oshape,
              inputs[0].dptr<IType>(), scalar_value, normal_tensor.dptr_,
              outputs[0].dptr<OType>());
        });
      });
    });
  } else if (inputs.size() == 2U) {
    // [tensor tensor] case
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      Kernel<check_legal_scale_kernel<IType>, xpu>::Launch(
          s, inputs[1].Size(), inputs[1].dptr<IType>(), indicator_device_ptr);
    });
    _copy<xpu>(s, &indicator_host, indicator_device_ptr);
    CHECK_GE(indicator_host, 0.0) << "ValueError: scale < 0";
    int ndim = FillShape(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                         &new_lshape, &new_hshape, &new_oshape);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> oshape = new_oshape.get<NDim>();
          Shape<NDim> lstride = calc_stride(new_lshape.get<NDim>());
          Shape<NDim> hstride = calc_stride(new_hshape.get<NDim>());
          Kernel<normal_kernel<NDim, IType, OType>, xpu>::Launch(
              s, outputs[0].Size(), lstride, hstride, oshape,
              inputs[0].dptr<IType>(), inputs[1].dptr<IType>(),
              normal_tensor.dptr_, outputs[0].dptr<OType>());
        });
      });
    });
  }
}

template<typename xpu, int ndim, typename DType>
inline void NormalReparamBackwardImpl(const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs,
                                      const mxnet::TShape& new_lshape,
                                      const mxnet::TShape& new_rshape,
                                      const mxnet::TShape& new_oshape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob lgrad = outputs[0].reshape(new_lshape);
  const TBlob rgrad = outputs[1].reshape(new_rshape);
  const TBlob ograd = inputs[0].reshape(new_oshape);
  // Mean
  const TBlob lhs = inputs[2].reshape(new_lshape);
  // Variance
  const TBlob rhs = inputs[3].reshape(new_rshape);
  const TBlob samples = inputs[4].reshape(new_oshape);
  const TBlob noise = inputs[5].reshape(new_oshape);
  size_t workspace_size_l = ReduceWorkspaceSize<ndim, DType>(
      s, lgrad.shape_, req[0], ograd.shape_, lhs.shape_, rhs.shape_);
  size_t workspace_size_r = ReduceWorkspaceSize<ndim, DType>(
      s, rgrad.shape_, req[1], ograd.shape_, lhs.shape_, rhs.shape_);
  size_t workspace_size = std::max(workspace_size_l, workspace_size_r);
  Tensor<xpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  Reduce<red::sum, ndim, DType, op::mshadow_op::identity>(s,
          lgrad, req[0], workspace, ograd);
  Reduce<red::sum, ndim, DType, op::mshadow_op::mul, op::mshadow_op::left>(
      s, rgrad, req[1], workspace, ograd, noise, rhs);
}

template<typename xpu, int ndim, typename DType>
inline void ScalarNormalReparamBackwardImpl(const OpContext& ctx,
                                            const std::vector<TBlob>& inputs,
                                            const std::vector<OpReqType>& req,
                                            const std::vector<TBlob>& outputs,
                                            const mxnet::TShape& new_ishape,
                                            const mxnet::TShape& new_oshape,
                                            const bool loc_is_tensor) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob igrad = outputs[0].reshape(new_ishape);
  // inputs: [grad_from_samples, grad_from_noise(invisible), input_tensor,
  //          samples, noise]
  const TBlob ograd = inputs[0].reshape(new_oshape);
  const TBlob itensor = inputs[2].reshape(new_ishape);
  const TBlob samples = inputs[3].reshape(new_oshape);
  const TBlob noise = inputs[4].reshape(new_oshape);
  size_t workspace_size =
      ReduceWorkspaceSize<ndim, DType>(s, igrad.shape_, req[0], ograd.shape_);
  Tensor<xpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  if (loc_is_tensor) {
    Reduce<red::sum, ndim, DType, op::mshadow_op::identity>(s, igrad, req[0],
                                                            workspace, ograd);
  } else {
    Reduce<red::sum, ndim, DType, op::mshadow_op::mul, op::mshadow_op::left>(
        s, igrad, req[0], workspace, ograd, noise, noise);
  }
}

// Allow normal sampling to be differentiable,
// using reparameterization trick described in:
// Auto-encoding variational bayes.
// Kingma, D. P., & Welling, M. (2013).
template<typename xpu>
void NormalReparamBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  // skip kernel launch for zero-size tensors
  if (inputs[0].shape_.Size() == 0U) {
    return;
  }
  // [scalar scalar] case
  if (outputs.size() == 0U) {
    return;
  }
  const NumpyNormalParam &param = nnvm::get<NumpyNormalParam>(attrs.parsed);
  // [tensor tensor] case
  if (inputs.size() == 6U) {
    mxnet::TShape new_lshape, new_rshape, new_oshape;
    int ndim = FillShape(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                         &new_lshape, &new_rshape, &new_oshape);
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        NormalReparamBackwardImpl<xpu, NDim, DType>(
          ctx, inputs, req, outputs, new_lshape, new_rshape, new_oshape);
      });
    });
  }
  // [tensor scalar], [scalar tensor] case
  if (inputs.size() == 5U) {
    mxnet::TShape new_ishape, new_oshape;
    int ndim = FillShape(outputs[0].shape_, outputs[0].shape_, inputs[0].shape_,
                         &new_ishape, &new_ishape, &new_oshape);
    bool loc_is_tensor = !param.loc.has_value();
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        ScalarNormalReparamBackwardImpl<xpu, NDim, DType>(
          ctx, inputs, req, outputs, new_ishape, new_oshape, loc_is_tensor);
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_NORMAL_OP_H_
