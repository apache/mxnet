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
 * \file preloaded_multi_sgd-inl.h
 * \brief Multi-sgd optimizers with lrs and wds as mxnet inputs
 * \author Clement Fuji Tsang
 */
#ifndef MXNET_OPERATOR_CONTRIB_PRELOADED_MULTI_SGD_INL_H_
#define MXNET_OPERATOR_CONTRIB_PRELOADED_MULTI_SGD_INL_H_
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"
#include "../tensor/init_op.h"
#include "../tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

struct PreloadedMultiSGDParam : public dmlc::Parameter<PreloadedMultiSGDParam> {
  float rescale_grad;
  float clip_gradient;
  int num_weights;
  DMLC_DECLARE_PARAMETER(PreloadedMultiSGDParam) {
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
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

struct PreloadedMultiSGDMomParam : public dmlc::Parameter<PreloadedMultiSGDMomParam> {
  float momentum;
  float rescale_grad;
  float clip_gradient;
  int num_weights;
  DMLC_DECLARE_PARAMETER(PreloadedMultiSGDMomParam) {
    DMLC_DECLARE_FIELD(momentum)
    .set_default(0.0f)
    .describe("The decay rate of momentum estimates at each epoch.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
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
inline bool PreloadedMultiSGDShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<mxnet::TShape> *in_attrs,
                                   std::vector<mxnet::TShape> *out_attrs) {
  const ParamType& param = dmlc::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), input_stride * param.num_weights + 2);
  CHECK_EQ(out_attrs->size(), param.num_weights);

  bool all_inferred = true;
  auto& input_shapes = *in_attrs;
  auto& output_shapes = *out_attrs;
  // Learning rates
  CHECK_EQ(in_attrs->at(param.num_weights * input_stride).Size(), param.num_weights)
    << "Number of learning rates is inconsistent with num_weights "
    << "parameter passed. Expected number of learning rates: "
    << param.num_weights << ", and got " << in_attrs->at(param.num_weights * input_stride).Size();
  // Weight decays
  CHECK_EQ(in_attrs->at(param.num_weights * input_stride + 1).Size(), param.num_weights)
    << "Number of weight decays is inconsistent with num_weights "
    << "parameter passed. Expected number of weight decays: "
    << param.num_weights << ", and got "
    << in_attrs->at(param.num_weights * input_stride + 1).Size();
  // Weights and gradients
  for (int i = 0; i < param.num_weights; ++i) {
    std::vector<mxnet::TShape> input_vec;
    std::vector<mxnet::TShape> output_vec({output_shapes[i]});
    for (int j = 0; j < input_stride; ++j) {
      input_vec.push_back(input_shapes[i * input_stride + j]);
    }
    all_inferred = all_inferred && ElemwiseShape<input_stride, 1>(attrs, &input_vec, &output_vec);
  }
  return all_inferred;
}

template <typename ParamType, int input_stride, int num_fp32_inputs>
inline bool MP_PreloadedMultiSGD_InferType(const nnvm::NodeAttrs& attrs,
                                           std::vector<int> *in_attrs,
                                           std::vector<int> *out_attrs) {
  const ParamType& param = dmlc::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), input_stride * param.num_weights + 2);
  CHECK_EQ(out_attrs->size(), param.num_weights);

  bool all_inferred = true;
  auto& input_types = *in_attrs;
  auto& output_types = *out_attrs;
  // Weights and gradients
  for (int i = 0; i < param.num_weights; ++i) {
    std::vector<int> input_vec;
    std::vector<int> output_vec({output_types[i]});
    for (int j = 0; j < input_stride - num_fp32_inputs; ++j) {
      input_vec.push_back(input_types[i * input_stride + j]);
    }
    all_inferred = all_inferred &&
                   ElemwiseType<input_stride - num_fp32_inputs, 1>(attrs, &input_vec, &output_vec);
  }
  // master copies of weights
  for (int i = 0; i < param.num_weights; ++i) {
    for (int j = 0; j < num_fp32_inputs; ++j) {
      TYPE_ASSIGN_CHECK(input_types, input_stride * i + input_stride - 1 - j, mshadow::kFloat32);
    }
  }
  TYPE_ASSIGN_CHECK(input_types, input_stride * param.num_weights, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(input_types, input_stride * param.num_weights + 1, mshadow::kFloat32);
  return all_inferred;
}

template<typename DType, typename MPDType>
struct PreloadedMultiSGDKernelParam {
  static const int N = 60;
  int count;
  size_t max_size;
  size_t sizes[N];
  DType * weights[N];
  DType * grads[N];
  MPDType * mom[N];
  MPDType * weights32[N];
  DType * out_data[N];
  float * lrs;
  float * wds;
  MPDType clip_gradient;
  MPDType rescale_grad;
  MPDType momentum;
};

template <typename MPDType, bool has_momentum, bool has_mixed_precision>
struct PreloadedMultiSGDKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const PreloadedMultiSGDKernelParam<DType, MPDType>& param,
                                  const OpReqType req) {
    for (int index = 0; index < param.count; ++index) {
      if ((size_t)i < param.sizes[index]) {
        MPDType w = has_mixed_precision ? param.weights32[index][i] :
                                          MPDType(param.weights[index][i]);
        MPDType mom = has_momentum ? param.mom[index][i] : MPDType(0);
        if (param.clip_gradient >= 0.0f) {
          mom = param.momentum*mom
                - param.lrs[index]*param.wds[index]*w
                - param.lrs[index]
                *mshadow_op::clip::Map(param.rescale_grad *
                                       static_cast<MPDType>(param.grads[index][i]),
                                     param.clip_gradient);
        } else {
          mom = param.momentum*mom
                - param.lrs[index]*param.wds[index]*w
                - param.lrs[index]*param.rescale_grad*static_cast<MPDType>(param.grads[index][i]);
        }
        if (has_momentum) {
          param.mom[index][i] = mom;
        }
        w = w + mom;
        if (has_mixed_precision) {
          param.weights32[index][i] = w;
        }
        KERNEL_ASSIGN(param.out_data[index][i], req, w);
      }
    }
  }
};

template<typename xpu,
         typename DType,
         typename MPDType,
         typename ParamType = PreloadedMultiSGDParam,
         int input_stride = 2>
PreloadedMultiSGDKernelParam<DType, MPDType> FillPreloadedMultiSGDKernelParam(
    const nnvm::NodeAttrs& attrs, const OpContext &ctx, const std::vector<TBlob> &inputs,
    const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  const ParamType& p = nnvm::get<ParamType>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  PreloadedMultiSGDKernelParam<DType, MPDType> param;
  param.clip_gradient = p.clip_gradient;
  param.rescale_grad = p.rescale_grad;
  param.momentum = 0;
  param.count = p.num_weights;
  param.max_size = 0;
  for (int i = 0; i < param.count; ++i) {
    param.sizes[i] = inputs[i * input_stride].shape_.Size();
    if (param.max_size < param.sizes[i]) {
      param.max_size = param.sizes[i];
    }
    param.weights[i] = inputs[i * input_stride].FlatTo2D<xpu, DType>(s).dptr_;
    param.grads[i] = inputs[i * input_stride + 1].FlatTo2D<xpu, DType>(s).dptr_;
    // if mixed precision, then the last input in a set
    // is 32-bit master copy of the weights
    if (!std::is_same<DType, MPDType>::value) {
      param.weights32[i] = inputs[i * input_stride + input_stride - 1]
                           .FlatTo2D<xpu, MPDType>(s).dptr_;
    }
    param.out_data[i] = outputs[i].FlatTo2D<xpu, DType>(s).dptr_;
  }
  const int lrs_idx = param.count * input_stride;
  const int wds_idx = param.count * input_stride + 1;
  param.lrs = inputs[lrs_idx].FlatTo2D<xpu, float>(s).dptr_;
  param.wds = inputs[wds_idx].FlatTo2D<xpu, float>(s).dptr_;
  return param;
}


template<typename xpu,
         typename DType,
         typename MPDType,
         int input_stride = 3>
PreloadedMultiSGDKernelParam<DType, MPDType> FillPreloadedMultiSGDMomKernelParam(
    const nnvm::NodeAttrs& attrs, const OpContext &ctx, const std::vector<TBlob> &inputs,
    const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  const PreloadedMultiSGDMomParam& p = nnvm::get<PreloadedMultiSGDMomParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  PreloadedMultiSGDKernelParam<DType, MPDType> param =
    FillPreloadedMultiSGDKernelParam<xpu,
                                     DType,
                                     MPDType,
                                     PreloadedMultiSGDMomParam,
                                     input_stride>(attrs, ctx, inputs, outputs);
  param.momentum = p.momentum;
  for (int i = 0; i < param.count; ++i) {
    param.mom[i] = inputs[i * input_stride + 2].FlatTo2D<xpu, MPDType>(s).dptr_;
  }

  return param;
}

template<typename T>
class preloaded_type_identity {
 public:
  using type = T;
};

template<typename T>
class preloaded_single_precision {
 public:
  using type = float;
};

template<typename xpu, template<typename> class MPTypeChooser, int input_stride>
inline void PreloadedMultiSGDUpdate(const nnvm::NodeAttrs& attrs,
                                    const OpContext &ctx,
                                    const std::vector<TBlob> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    using MPDType = typename MPTypeChooser<DType>::type;
    PreloadedMultiSGDKernelParam<DType, MPDType> param =
      FillPreloadedMultiSGDKernelParam<xpu,
                                       DType,
                                       MPDType,
                                       PreloadedMultiSGDParam,
                                       input_stride>(attrs, ctx, inputs, outputs);
    Kernel<PreloadedMultiSGDKernel<MPDType,
                                   false,
                                   !std::is_same<DType, MPDType>::value>,
                                   xpu>::Launch(s, param.max_size, param, req[0]);
  });
}

template<typename xpu, template<typename> class MPTypeChooser, int input_stride>
inline void PreloadedMultiSGDMomUpdate(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    using MPDType = typename MPTypeChooser<DType>::type;
    PreloadedMultiSGDKernelParam<DType, MPDType> param =
      FillPreloadedMultiSGDMomKernelParam<xpu,
                                          DType,
                                          MPDType,
                                          input_stride>(attrs, ctx, inputs, outputs);
    Kernel<PreloadedMultiSGDKernel<MPDType,
                                   true,
                                   !std::is_same<DType, MPDType>::value>,
                                   xpu>::Launch(s, param.max_size, param, req[0]);
  });
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_CONTRIB_PRELOADED_MULTI_SGD_INL_H_
