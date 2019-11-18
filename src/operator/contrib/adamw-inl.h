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
 *  Copyright (c) 2018 by Contributors
 * \file adamw-inl.h
 * \brief Optimizer operators
 * \author Haibin Lin, Moises Hernandez, Andrei Ivanov
 */
#ifndef MXNET_OPERATOR_CONTRIB_ADAMW_INL_H_
#define MXNET_OPERATOR_CONTRIB_ADAMW_INL_H_
#include <mxnet/operator.h>
#include <vector>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

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
                               mxnet::ShapeVector *in_attrs,
                               mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(total_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  SHAPE_ASSIGN_CHECK(*in_attrs, total_in - 1, mxnet::TShape());
  // TODO(@reminisce): change "none" behavior in ElemwiseAttr
  return ElemwiseAttr<mxnet::TShape, shape_is_none, shape_assign, true, shape_string, n_in, n_out>(
      attrs, in_attrs, out_attrs, mxnet::TShape());
}

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
    float scaled_grad = param_rescale_grad*static_cast<float>(grad_data[i]);
    if (param_clip_gradient >= 0.0f)
      scaled_grad = mshadow_op::clip::Map(scaled_grad, param_clip_gradient);

    float mean = mean_data[i] = param_beta1 * mean_data[i] + (1.0f - param_beta1) * scaled_grad;
    float var = var_data[i] = param_beta2 * var_data[i] +
                  (1.0f - param_beta2) * mshadow_op::square::Map(scaled_grad);

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
    const auto& param = nnvm::get<AdamWParam>(attrs.parsed);
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
    const auto &param = nnvm::get<AdamWParam>(attrs.parsed);
    Stream<xpu>* s = ctx.get_stream<xpu>();
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      const Tensor<xpu, 2, DType> &weight = inputs[0].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> mean = inputs[2].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> var = inputs[3].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);

      grad = scalar<DType>(rescale_grad) * grad;
      if (param.clip_gradient >= 0.0f)
        grad = F<clip>(grad, DType(param.clip_gradient));

      mean = scalar<DType>(param.beta1) * mean + scalar<DType>(1.f-param.beta1) * grad;
      var = scalar<DType>(param.beta2) * var + scalar<DType>(1.f-param.beta2) * F<square>(grad);

      Assign(out, req[0],
             weight -
             scalar<DType>(param.eta) * (scalar<DType>(param.lr) *
             mean / (F<square_root>(var) + scalar<DType>(param.epsilon)) +
             (scalar<DType>(param.wd) * weight)));
    });
  }
};

////
// Multiple gradients in single kernel
////
struct MultiAdamWParam : public dmlc::Parameter<MultiAdamWParam> {
  mxnet::Tuple<float> lrs;
  mxnet::Tuple<float> wds;
  mxnet::Tuple<float> etas;
  float beta1;
  float beta2;
  float epsilon;
  float clip_gradient;
  int num_weights;
  DMLC_DECLARE_PARAMETER(MultiAdamWParam) {
    DMLC_DECLARE_FIELD(lrs)
    .describe("Learning rates");
    DMLC_DECLARE_FIELD(beta1)
    .set_default(0.9f)
    .describe("The decay rate for the 1st moment estimates.");
    DMLC_DECLARE_FIELD(beta2)
    .set_default(0.999f)
    .describe("The decay rate for the 2nd moment estimates.");
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1e-8f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(wds)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(etas)
    .describe("Learning rates schedule multiplier");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(num_weights)
    .set_default(1)
    .describe("Number of updated weights.");
  }
};


template<typename ParamType, int input_stride>
inline bool MP_MultiAdamW_InferShape(const nnvm::NodeAttrs& attrs,
                                          mxnet::ShapeVector *in_attrs,
                                          mxnet::ShapeVector *out_attrs) {
  const ParamType& param = dmlc::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), input_stride * param.num_weights +1);
  CHECK_EQ(out_attrs->size(), param.num_weights);

  bool all_inferred = true;
  auto& input_shapes = *in_attrs;
  auto& output_shapes = *out_attrs;

  // Learning rates
  CHECK_EQ(param.lrs.ndim(), param.num_weights)
    << "Number of learning rates is inconsistent with num_weights "
    << "parameter passed. Expected number of learning rates: "
    << param.num_weights << ", and got " << param.lrs.ndim();
  // Weight decays
  CHECK_EQ(param.wds.ndim(), param.num_weights)
    << "Number of weight decays is inconsistent with num_weights "
    << "parameter passed. Expected number of weight decays: "
    << param.num_weights << ", and got " << param.wds.ndim();
  // Learning rates schedule multiplier
  CHECK_EQ(param.etas.ndim(), param.num_weights)
    << "Number of learning rates schedule multiplier is inconsistent with num_weights "
    << "parameter passed. Expected number of learning rates schedule multiplier: "
    << param.num_weights << ", and got " << param.lrs.ndim();

  // Weights, gradients, mean and variance
  for (int i = 0; i < param.num_weights; ++i) {
    mxnet::ShapeVector input_vec;
    mxnet::ShapeVector output_vec({output_shapes[i]});
    for (int j = 0; j < input_stride; ++j) {
      input_vec.push_back(input_shapes[i * input_stride + j]);
    }
    all_inferred = all_inferred && ElemwiseShape<input_stride, 1>(attrs, &input_vec, &output_vec);
  }

  SHAPE_ASSIGN_CHECK(*in_attrs, param.num_weights*input_stride, mxnet::TShape());
  return all_inferred;
}

template <typename ParamType, int input_stride, int num_fp32_inputs>
inline bool MP_MultiAdamW_InferType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int> *in_attrs,
                                    std::vector<int> *out_attrs) {
  const ParamType& param = dmlc::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), input_stride * param.num_weights +1);
  CHECK_EQ(out_attrs->size(), param.num_weights);

  bool all_inferred = true;
  auto& input_types = *in_attrs;
  auto& output_types = *out_attrs;

  // Weights, gradients,
  for (int i = 0; i < param.num_weights; ++i) {
    std::vector<int> input_vec;
    std::vector<int> output_vec({output_types[i]});
    for (int j = 0; j < input_stride - 2 - num_fp32_inputs; ++j) {
      input_vec.push_back(input_types[i * input_stride + j]);
    }
    all_inferred = all_inferred &&
            ElemwiseType<input_stride - 2 - num_fp32_inputs, 1>(attrs, &input_vec, &output_vec);
  }
  // mean, var
  for (int i = 0; i < param.num_weights; ++i) {
    TYPE_ASSIGN_CHECK(input_types, input_stride * i +2, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(input_types, input_stride * i +3, mshadow::kFloat32);
  }

  // master copies of weights
  for (int i = 0; i < param.num_weights; ++i) {
    for (int j = 0; j < num_fp32_inputs; ++j) {
      TYPE_ASSIGN_CHECK(input_types, input_stride * i + input_stride - 1 - j, mshadow::kFloat32);
    }
  }

  TYPE_ASSIGN_CHECK(input_types, param.num_weights*input_stride, mshadow::kFloat32);
  return all_inferred;
}


template<typename T>
class Adam_type_identity {
 public:
  using type = T;
};


template<typename T>
class Adam_single_precision {
 public:
  using type = float;
};

template<typename DType, typename MPDType>
struct MultiAdamKernelParam {
  static const int N = 50;
  int count;
  size_t max_size;
  size_t sizes[N];
  DType* weights[N];
  DType* grad_data[N];
  MPDType* mean_data[N];
  MPDType* var_data[N];
  MPDType* weights32[N];
  DType* out_data[N];
  MPDType clip_gradient;
  MPDType beta1;
  MPDType beta2;
  MPDType etas[N];
  MPDType lrs[N];
  MPDType wds[N];
  MPDType epsilon;
};

template<typename MPDType, bool has_mixed_precision>
struct MultiMPAdamWKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const MultiAdamKernelParam<DType, MPDType>& param,
                                  const OpReqType req, const float rescale_grad){
    for (int index = 0; index < param.count; ++index) {
      if ((size_t)i < param.sizes[index]) {
        MPDType w = has_mixed_precision ? param.weights32[index][i]:
                                          MPDType(param.weights[index][i]);
        MPDType scaled_grad = static_cast<MPDType>(rescale_grad)*
                              static_cast<MPDType>(param.grad_data[index][i]);

        if (param.clip_gradient >= 0.0f)
          scaled_grad = mshadow_op::clip::Map(scaled_grad, param.clip_gradient);

        const auto mean = param.beta1 * (param.mean_data[index][i]- scaled_grad) + scaled_grad;
        const auto adj = mshadow_op::square::Map(scaled_grad);
        const auto var = param.beta2 * (param.var_data[index][i] - adj) + adj;

        param.mean_data[index][i] = mean;
        param.var_data[index][i] = var;
        w = w - param.etas[index] * (param.lrs[index] *
            mean / (mshadow_op::square_root::Map(var) + param.epsilon)
            + param.wds[index] * w);
        if (has_mixed_precision)
          param.weights32[index][i] = w;

        KERNEL_ASSIGN(param.out_data[index][i], req, w);
      }
    }
  }
};

template<typename xpu,
         typename DType,
         typename MPDType,
         typename ParamType = MultiAdamWParam,
         int input_stride = 4>
void FillMultiAdamKernelParam(const nnvm::NodeAttrs& attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<TBlob> &outputs,
                              MultiAdamKernelParam<DType, MPDType> *pParam) {
  const ParamType& p = nnvm::get<ParamType>(attrs.parsed);
  mxnet_op::Stream<xpu>* s = ctx.get_stream<xpu>();
  pParam->clip_gradient = p.clip_gradient;
  pParam->beta1 = p.beta1;
  pParam->beta2 = p.beta2;

  pParam->epsilon = p.epsilon;

  pParam->count = p.num_weights;
  pParam->max_size = 0;
  constexpr bool isSame = std::is_same<DType, MPDType>::value;
  for (int i = 0; i < pParam->count; ++i) {
    const auto idx = i * input_stride;
    pParam->sizes[i] = inputs[idx].shape_.Size();
    if (pParam->max_size < pParam->sizes[i])
      pParam->max_size = pParam->sizes[i];

    pParam->weights[i] = inputs[idx].FlatTo2D<xpu, DType>(s).dptr_;
    pParam->grad_data[i] = inputs[idx + 1].FlatTo2D<xpu, DType>(s).dptr_;
    pParam->mean_data[i] = inputs[idx + 2].FlatTo2D<xpu, MPDType>(s).dptr_;
    pParam->var_data[i]  = inputs[idx + 3].FlatTo2D<xpu, MPDType>(s).dptr_;
    // if mixed precision, then the last input in a set
    // is 32-bit master copy of the weights
    if (!isSame)
      pParam->weights32[i] = inputs[idx + input_stride - 1].FlatTo2D<xpu, MPDType>(s).dptr_;

    pParam->out_data[i] = outputs[i].FlatTo2D<xpu, DType>(s).dptr_;
  }
  memcpy(pParam->etas, p.etas.begin(), pParam->count * sizeof(p.etas[0]));
  memcpy(pParam->lrs, p.lrs.begin(), pParam->count * sizeof(p.lrs[0]));
  memcpy(pParam->wds, p.wds.begin(), pParam->count * sizeof(p.wds[0]));
}

template<typename xpu, template<typename> class MPTypeChooser, int input_stride>
static inline void MultiAdamWUpdate(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<TBlob> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<TBlob> &outputs,
                                    const float rescale_grad) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    using MPDType = typename MPTypeChooser<DType>::type;
    MultiAdamKernelParam<DType, MPDType> param;
    FillMultiAdamKernelParam<xpu, DType, MPDType, MultiAdamWParam, input_stride>
            (attrs, ctx, inputs, outputs, &param);

    Kernel<MultiMPAdamWKernel<MPDType, !std::is_same<DType, MPDType>::value>, xpu>::
                              Launch(s, param.max_size, param, req[0], rescale_grad);
  });
}

template<typename xpu>
void GetScaleFloat(mshadow::Stream<xpu> *s, const TBlob &scale_blob, float *pScalef);

template<typename xpu>
bool PrepareInputBlobs(const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       std::vector<TBlob> *inputs_wo_scale,
                       float *pScalef) {
  const size_t num_in = inputs.size() - 1;
  GetScaleFloat<xpu>(ctx.get_stream<xpu>(), inputs[num_in], pScalef);
  if (!std::isfinite(*pScalef) || *pScalef == 0)
    return false;

  inputs_wo_scale->reserve(num_in);
  for (size_t i = 0; i < num_in; i++)
    inputs_wo_scale->emplace_back(inputs[i]);

  return true;
}

template<typename xpu, class F>
inline void MPUpdate(const nnvm::NodeAttrs& attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
  std::vector<TBlob> inputs_wo_scale;
  float scalef;
  if (!PrepareInputBlobs<xpu>(ctx, inputs, &inputs_wo_scale, &scalef))
    return;

  F::Forward(attrs, ctx, inputs_wo_scale, req, outputs, scalef);
}

template<typename xpu, bool MP>
inline void multiMPUpdate(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  std::vector<TBlob> inputs_wo_scale;
  float scalef;
  if (!PrepareInputBlobs<xpu>(ctx, inputs, &inputs_wo_scale, &scalef))
    return;

  if (!MP)
    MultiAdamWUpdate<xpu, Adam_type_identity, 4>
      (attrs, ctx, inputs_wo_scale, req, outputs, scalef);
  else
    MultiAdamWUpdate<xpu, Adam_single_precision, 5>
      (attrs, ctx, inputs_wo_scale, req, outputs, scalef);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_ADAMW_INL_H_
