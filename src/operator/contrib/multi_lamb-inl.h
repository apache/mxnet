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
 * \file multi_lamb-inl.h
 * \brief multi-tensor LAMB optimizer
 * \author Moises Hernandez
 */
#ifndef MXNET_OPERATOR_CONTRIB_MULTI_LAMB_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTI_LAMB_INL_H_

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
#include "../mxnet_op.h"
#include "../tensor/init_op.h"
#include "../tensor/util/tensor_util-inl.h"
#include "multi_sum_sq-inl.h"

namespace mxnet {
namespace op {

namespace multilamb {
enum MultiLambUpdateResource {kTempSpace};
}  // namespace multilamb

struct MultiLAMBParam : public dmlc::Parameter<MultiLAMBParam> {
  mxnet::Tuple<float> learning_rates;
  mxnet::Tuple<float> wds;
  float beta1;
  float beta2;
  float epsilon;
  float rescale_grad;
  float lower_bound;
  float upper_bound;
  float clip_gradient;
  bool bias_correction;
  int num_tensors;
  mxnet::Tuple<int> step_count;

  DMLC_DECLARE_PARAMETER(MultiLAMBParam) {
    DMLC_DECLARE_FIELD(learning_rates)
    .describe("List of learning rates");
    DMLC_DECLARE_FIELD(beta1)
    .set_default(0.9f)
    .describe("Exponential decay rate for the first moment estimates.");
    DMLC_DECLARE_FIELD(beta2)
    .set_default(0.999f)
    .describe("Exponential decay rate for the second moment estimates.");
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1e-6f)
    .describe("Small value to avoid division by 0.");
    DMLC_DECLARE_FIELD(wds)
    .describe("List of Weight decays."
              "Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
               "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Gradient rescaling factor");
    DMLC_DECLARE_FIELD(lower_bound)
    .set_default(-1.0f)
    .describe("Lower limit of norm of weight. If lower_bound <= 0, Lower limit is not set");
    DMLC_DECLARE_FIELD(upper_bound)
    .set_default(-1.0f)
    .describe("Upper limit of norm of weight. If upper_bound <= 0, Upper limit is not set");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(bias_correction)
    .set_default(true)
    .describe("Whether to use bias correction.");
    DMLC_DECLARE_FIELD(step_count)
    .describe("Step count for each tensor");
    DMLC_DECLARE_FIELD(num_tensors)
    .set_default(1)
    .describe("Number of tensors");
  }
};

template<typename ParamType, int input_stride>
inline bool MultiLAMBInferShape(const nnvm::NodeAttrs& attrs,
                                mxnet::ShapeVector *in_attrs,
                                mxnet::ShapeVector *out_attrs) {
  const ParamType& param = dmlc::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), input_stride * param.num_tensors);
  CHECK_EQ(out_attrs->size(), param.num_tensors);

  bool all_inferred = true;
  auto& input_shapes = *in_attrs;
  auto& output_shapes = *out_attrs;

  CHECK_LE(param.num_tensors, 45)
    << "Invalid number of tensors, the maximum value is 45, and got "
    << param.num_tensors;
  CHECK_EQ(param.learning_rates.ndim(), param.num_tensors)
    << "Number of learning rates is inconsistent with num_tensors "
    << "parameter passed. Expected number of learning rates: "
    << param.num_tensors << ", and got " << param.learning_rates.ndim();
  CHECK_EQ(param.wds.ndim(), param.num_tensors)
    << "Number of weight decays is inconsistent with num_tensors "
    << "parameter passed. Expected number of weight decays: "
    << param.num_tensors << ", and got " << param.wds.ndim();
  CHECK_EQ(param.step_count.ndim(), param.num_tensors)
    << "Number of step counts is inconsistent with num_tensors."
    << "Expected number of step counts: "
    << param.num_tensors << ", and got " << param.step_count.ndim();

  // Weights, gradients, mean and variance
  for (int i = 0; i < param.num_tensors; ++i) {
    mxnet::ShapeVector input_vec;
    mxnet::ShapeVector output_vec({output_shapes[i]});
    for (int j = 0; j < input_stride; ++j) {
      input_vec.push_back(input_shapes[i * input_stride + j]);
    }
    all_inferred = all_inferred && ElemwiseShape<input_stride, 1>(attrs, &input_vec, &output_vec);
  }
  return all_inferred;
}

template <typename ParamType, int input_stride>
inline bool MPMultiLAMBInferType(const nnvm::NodeAttrs& attrs,
                                 std::vector<int> *in_attrs,
                                 std::vector<int> *out_attrs) {
  const ParamType& param = dmlc::get<ParamType>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), input_stride * param.num_tensors);
  CHECK_EQ(out_attrs->size(), param.num_tensors);

  bool all_inferred = true;
  auto& input_types = *in_attrs;
  auto& output_types = *out_attrs;

  // weights, gradients
  for (int i = 0; i < param.num_tensors; ++i) {
    std::vector<int> input_vec;
    std::vector<int> output_vec({output_types[i]});
    for (int j = 0; j < 2; ++j) {
      input_vec.push_back(input_types[i * input_stride + j]);
    }
    all_inferred = all_inferred &&
            ElemwiseType<2, 1>(attrs, &input_vec, &output_vec);
  }

  // mean, var, weights32 (master copies of weights)
  for (int i = 0; i < param.num_tensors; ++i) {
    TYPE_ASSIGN_CHECK(input_types, input_stride * i + 2, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(input_types, input_stride * i + 3, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(input_types, input_stride * i + input_stride - 1, mshadow::kFloat32);
  }
  return all_inferred;
}

template<typename T>
class LAMBTypeIdentity {
 public:
  using type = T;
};

template<typename T>
class LAMBSinglePrecision {
 public:
  using type = float;
};

template<typename DType, typename MPDType>
struct MultiLAMBKernelParam {
  static const int N = 45;
  size_t ntensors;
  size_t max_size;
  size_t total_size;
  size_t sizes[N];
  size_t tensor2temp_g[N];
  DType* weights[N];
  DType* grads[N];
  MPDType* mean[N];
  MPDType* var[N];
  MPDType* weights32[N];
  DType* out_data[N];
  int step_count[N];
  MPDType learning_rates[N];
  MPDType wds[N];

  // gpu
  int chunk_size = 65536;
  int nchunks;
};

template<typename xpu,
         typename DType,
         typename MPDType,
         typename ParamType = MultiLAMBParam,
         int input_stride = 5>
void FillMultiLAMBKernelParam(const nnvm::NodeAttrs& attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<TBlob> &outputs,
                              MultiLAMBKernelParam<DType, MPDType> *multi_param) {
  const ParamType& p = nnvm::get<ParamType>(attrs.parsed);
  mxnet_op::Stream<xpu>* s = ctx.get_stream<xpu>();

  multi_param->ntensors = p.num_tensors;
  multi_param->total_size = 0;
  multi_param->max_size = 0;
  multi_param->nchunks = 0;

  constexpr bool is_same = std::is_same<DType, MPDType>::value;
  for (size_t i = 0; i < multi_param->ntensors; ++i) {
    const auto idx = i * input_stride;
    multi_param->sizes[i] = inputs[idx].shape_.Size();
    multi_param->tensor2temp_g[i] = multi_param->total_size;
    multi_param->total_size += multi_param->sizes[i];
    if (multi_param->max_size < multi_param->sizes[i])
      multi_param->max_size = multi_param->sizes[i];

    multi_param->weights[i] = inputs[idx].FlatTo2D<xpu, DType>(s).dptr_;
    multi_param->grads[i] = inputs[idx + 1].FlatTo2D<xpu, DType>(s).dptr_;
    multi_param->mean[i] = inputs[idx + 2].FlatTo2D<xpu, MPDType>(s).dptr_;
    multi_param->var[i]  = inputs[idx + 3].FlatTo2D<xpu, MPDType>(s).dptr_;

    // if mixed precision, then the last input in a set
    // is 32-bit master copy of the weights
    if (!is_same)
      multi_param->weights32[i] = inputs[idx + input_stride - 1].FlatTo2D<xpu, MPDType>(s).dptr_;
    multi_param->out_data[i] = outputs[i].FlatTo2D<xpu, DType>(s).dptr_;
    multi_param->nchunks += (multi_param->sizes[i] + multi_param->chunk_size - 1)
                            / multi_param->chunk_size;
    multi_param->learning_rates[i] = static_cast<MPDType>(p.learning_rates[i]);
    multi_param->wds[i] = static_cast<MPDType>(p.wds[i]);
  }
  memcpy(multi_param->step_count, p.step_count.begin(), multi_param->ntensors * sizeof(int));
}

using namespace mxnet_op;
template<typename MPDType, typename DType>
void CallKernel1(Stream<cpu>* s);
template<typename MPDType, typename DType>
void CallKernel1(Stream<gpu>* s);

template<typename MPDType, typename DType>
void CallKernel2(Stream<cpu>* s);
template<typename MPDType, typename DType>
void CallKernel2(Stream<gpu>* s);

template<typename xpu, template<typename> class MPTypeChooser, int input_stride>
inline void MultiLAMB(const nnvm::NodeAttrs& attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  auto param = nnvm::get<MultiLAMBParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    using MPDType = typename MPTypeChooser<DType>::type;
    MultiLAMBKernelParam<DType, MPDType> kernel_params;
    FillMultiLAMBKernelParam<xpu, DType, MPDType, MultiLAMBParam, input_stride>
            (attrs, ctx, inputs, outputs, &kernel_params);

    // create vector of TBlob with all the weights contiguous
    std::vector<TBlob> weights;
    for (size_t index = 0; index < kernel_params.ntensors; ++index) {
        weights.emplace_back(inputs[index*input_stride]);
    }

    // Calculate amount of temporary storage (temp_g, r1, r2, block_to_tensor, block_to_chunk)
    size_t workspace_size = kernel_params.total_size * sizeof(float) +
        2 * kernel_params.ntensors * sizeof(float) +
        2 * kernel_params.nchunks * sizeof(int);
    // take into account the required storage required within MultiSumSqRun
    size_t required_storage_multi_sum_sq = 0;
    required_storage_multi_sum_sq = GetRequiredStorageMultiSumSq<xpu>(inputs);
    workspace_size += required_storage_multi_sum_sq;

    // Request temporary storage
    Tensor<xpu, 1, char> workspace =
    ctx.requested[multilamb::kTempSpace].get_space_typed<xpu, 1, char>(
      Shape1(workspace_size), s);

    // Create tensors
    size_t pos_wspace = required_storage_multi_sum_sq;
    Tensor<xpu, 1, float> temp_g(reinterpret_cast<float*>(&workspace[pos_wspace]),
      Shape1(kernel_params.total_size), s);
    // create vector of TBlob with all the temp_g contiguous
    std::vector<TBlob> temp_g_tblobs;
    for (size_t index = 0; index < kernel_params.ntensors; ++index) {
      Tensor<xpu, 1, float> aux(reinterpret_cast<float*>(&workspace[pos_wspace]),
        Shape1(kernel_params.sizes[index]), s);
      TBlob newtblob(aux);
      temp_g_tblobs.emplace_back(newtblob);
      pos_wspace += kernel_params.sizes[index] * sizeof(float);
    }
    Tensor<xpu, 1, float> r1(reinterpret_cast<float*>(&workspace[pos_wspace]),
      Shape1(kernel_params.ntensors), s);
    pos_wspace += kernel_params.ntensors * sizeof(float);
    Tensor<xpu, 1, float> r2(reinterpret_cast<float*>(&workspace[pos_wspace]),
      Shape1(kernel_params.ntensors), s);
    pos_wspace += kernel_params.ntensors * sizeof(float);
    Tensor<xpu, 1, int> block_to_tensor(reinterpret_cast<int*>(&workspace[pos_wspace]),
      Shape1(kernel_params.nchunks), s);
    pos_wspace += kernel_params.nchunks * sizeof(int);
    Tensor<xpu, 1, int> block_to_chunk(reinterpret_cast<int*>(&workspace[pos_wspace]),
      Shape1(kernel_params.nchunks), s);

    MultiSumSqRun<xpu>(weights, kernel_params.ntensors, r1.dptr_, ctx);
    CallKernel1<MPDType, DType>(s, kernel_params, param, temp_g.dptr_,
                                block_to_tensor.dptr_,
                                block_to_chunk.dptr_);
    MultiSumSqRun<xpu>(temp_g_tblobs, kernel_params.ntensors, r2.dptr_, ctx);
    CallKernel2<MPDType, DType>(s, kernel_params, param, r1.dptr_, r2.dptr_,
                                temp_g.dptr_,
                                block_to_tensor.dptr_, block_to_chunk.dptr_,
                                req[0]);
  });
}

template<typename xpu, bool MP>
inline void MultiLAMBUpdate(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const std::vector<TBlob> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<TBlob> &outputs) {
  if (!MP) {
    MultiLAMB<xpu, LAMBTypeIdentity, 4>
      (attrs, ctx, inputs, req, outputs);
  } else {
    MultiLAMB<xpu, LAMBSinglePrecision, 5>
      (attrs, ctx, inputs, req, outputs);
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_MULTI_LAMB_INL_H_
