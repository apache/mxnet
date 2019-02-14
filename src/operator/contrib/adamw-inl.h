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
 * \file optimizer_op-inl.h
 * \brief Optimizer operators
 * \author Haibin Lin
 */
#ifndef MXNET_OPERATOR_CONTRIB_ADAMW_INL_H_
#define MXNET_OPERATOR_CONTRIB_ADAMW_INL_H_
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include <cmath>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

struct AdamWParam : public dmlc::Parameter<AdamWParam> {
  float lr;
  float beta1;
  float beta2;
  float epsilon;
  float wd;
  float eta;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(AdamWParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(beta1)
    .set_default(0.9f)
    .describe("The decay rate for the 1st moment estimates.");
    DMLC_DECLARE_FIELD(beta2)
    .set_default(0.999f)
    .describe("The decay rate for the 2nd moment estimates.");
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1e-8f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(eta)
    .describe("Learning rate schedule multiplier");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};

// rescale_grad is a reserved argument at position -1. Example:
// n_in = 2: weight, grad (fp16)
// n_out = 1: weight (fp16)
// total_in = 6: weight, grad, mean, var, weight32, rescale_grad (fp32)
template<int n_in, int n_out, int total_in>
inline bool MPUpdateInferShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape> *in_attrs,
                               std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(total_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  // rescale_grad.shape = (1,)
  SHAPE_ASSIGN_CHECK(*in_attrs, total_in - 1, mshadow::Shape1(1));
  return ElemwiseAttr<TShape, shape_is_none, shape_assign, true, shape_string, n_in, n_out>(
      attrs, in_attrs, out_attrs, TShape());
}

// rescale_grad is a reserved argument at position -1. Example:
// n_in = 2: weight, grad (fp16)
// n_out = 1: weight (fp16)
// total_in = 6: weight, grad, mean, var, weight32, rescale_grad (fp32)
template<int n_in, int n_out, int total_in>
inline bool MPUpdateInferType(const nnvm::NodeAttrs& attrs,
                              std::vector<int> *in_attrs,
                              std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(total_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  for (int i = n_in; i < total_in; ++i) {
    TYPE_ASSIGN_CHECK(*in_attrs, i, mshadow::kFloat32);
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string, n_in, n_out>(
      attrs, in_attrs, out_attrs, -1);
}

template<int req>
struct MPAdamWKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, float* mean_data,
    float* var_data, const DType* weight_data, const DType* grad_data, float* weight32,
    const float param_clip_gradient, const float param_beta1, const float param_beta2,
    const float param_eta, const float param_lr, const float param_wd,
    const float param_rescale_grad, const float param_epsilon) {
    float w = weight32[i];
    float mean = mean_data[i];
    float var = var_data[i];
    float scaled_grad = param_rescale_grad*static_cast<float>(grad_data[i]);
    if (param_clip_gradient >= 0.0f) {
      mean = param_beta1 * mean +
             (1 - param_beta1) * mshadow_op::clip::Map(scaled_grad, param_clip_gradient);
      var = param_beta2 * var + (1 - param_beta2) *
            mshadow_op::square::Map(mshadow_op::clip::Map(scaled_grad, param_clip_gradient));
    } else {
      mean = param_beta1 * mean + (1 - param_beta1) * scaled_grad;
      var = param_beta2 * var + (1 - param_beta2) * mshadow_op::square::Map(scaled_grad);
    }
    mean_data[i] = mean;
    var_data[i] = var;
    w = w - param_eta * (param_lr * mean / (mshadow_op::square_root::Map(var) + param_epsilon)
                         + param_wd * w);
    weight32[i] = w;
    KERNEL_ASSIGN(out_data[i], req, w);
  }
};


template<typename xpu>
struct MPAdamWUpdate {
  static inline void Forward(const nnvm::NodeAttrs& attrs,
               const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &outputs,
               const float rescale_grad) {
    using namespace mxnet_op;
    AdamWParam param = nnvm::get<AdamWParam>(attrs.parsed);
    Stream<xpu>* s = ctx.get_stream<xpu>();
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, float> mean = inputs[2].FlatTo2D<xpu, float>(s);
      Tensor<xpu, 2, float> var = inputs[3].FlatTo2D<xpu, float>(s);
      Tensor<xpu, 2, float> weight32 = inputs[4].FlatTo2D<xpu, float>(s);
      Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<MPAdamWKernel<req_type>, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, mean.dptr_,
          var.dptr_, weight.dptr_, grad.dptr_, weight32.dptr_, param.clip_gradient, param.beta1,
          param.beta2, param.eta, param.lr, param.wd, rescale_grad, param.epsilon);
      });
    });
  }
};

/*
 * \brief adam_w update.
 */
template<typename xpu>
struct AdamWUpdate {
  static inline void Forward(const nnvm::NodeAttrs& attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs,
                             const float rescale_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace mshadow_op;
    const AdamWParam& param = nnvm::get<AdamWParam>(attrs.parsed);
    Stream<xpu>* s = ctx.get_stream<xpu>();
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> mean = inputs[2].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> var = inputs[3].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);

      grad = scalar<DType>(rescale_grad) * grad;
      if (param.clip_gradient >= 0.0f) {
        mean = scalar<DType>(param.beta1)*mean + scalar<DType>(1.f-param.beta1) *
            F<clip>(grad, DType(param.clip_gradient));
        var = scalar<DType>(param.beta2)*var + scalar<DType>(1.f-param.beta2)*F<square>(
            F<clip>(grad, DType(param.clip_gradient)));
      } else {
        mean = scalar<DType>(param.beta1)*mean + scalar<DType>(1.f-param.beta1) * grad;
        var = scalar<DType>(param.beta2)*var + scalar<DType>(1.f-param.beta2) * F<square>(grad);
      }
      Assign(out, req[0],
             weight -
             scalar<DType>(param.eta) * (scalar<DType>(param.lr) *
             mean / (F<square_root>(var) + scalar<DType>(param.epsilon)) +
             (scalar<DType>(param.wd) * weight)));
    });
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ADAMW_INL_H_
