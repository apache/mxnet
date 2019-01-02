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

#if MXNET_USE_MKLDNN == 1

#include <utility>
#include <vector>
#include <string>
#include "../common.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../quantization/quantization_utils.h"
#include "mkldnn_conv-inl.h"

namespace mxnet {
namespace op {

template <typename DType>
static void UpdateConvWeightBias(NDArray *weight, NDArray *bias, bool no_bias,
                                 const NDArray &gamma, const NDArray &beta,
                                 const NDArray &mean, const NDArray &variance,
                                 const BatchNormParam *param) {
  // TODO(Zhennan): Handle the case weight is not in dims 4.
  NDArray update_weight = NDArray(weight->storage_type(), weight->shape(),
                                  weight->ctx(), true, weight->dtype());
  NDArray update_bias = NDArray(beta.storage_type(), beta.shape(), beta.ctx(),
                                true, beta.dtype());
  const DType *weight_ptr = weight->data().dptr<DType>();
  const DType *bias_ptr = no_bias ? nullptr : bias->data().dptr<DType>();
  const DType *gamma_ptr = gamma.Reorder2Default().data().dptr<DType>();
  const DType *beta_ptr = beta.Reorder2Default().data().dptr<DType>();
  const DType *mean_ptr = mean.Reorder2Default().data().dptr<DType>();
  const DType *var_ptr = variance.Reorder2Default().data().dptr<DType>();
  DType *update_weight_ptr = update_weight.data().dptr<DType>();
  DType *update_bias_ptr = update_bias.data().dptr<DType>();
  size_t channel = gamma.shape()[0];
  size_t offset = weight->shape()[1] * weight->shape()[2] * weight->shape()[3];
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int c = 0; c < static_cast<int>(channel); ++c) {
    const DType *p1 = weight_ptr + c * offset;
    DType *p2 = update_weight_ptr + c * offset;
    DType alpha = (param->fix_gamma ? static_cast<DType>(1.0f) : gamma_ptr[c]) /
                  sqrt(var_ptr[c] + param->eps);

    if (bias_ptr)
      update_bias_ptr[c] = beta_ptr[c] + alpha * (bias_ptr[c] - mean_ptr[c]);
    else
      update_bias_ptr[c] = beta_ptr[c] - alpha * mean_ptr[c];

    for (size_t k = 0; k < offset; ++k) {
      p2[k] = p1[k] * alpha;
    }
  }
  *weight = update_weight;
  *bias = update_bias;
}

static inline size_t GetInSumIndex(const MKLDNNConvFusionParam &param) {
  return 2 + (param.full_conv_param.conv_param.no_bias ? 0 : 1) +
         (param.full_conv_param.mkldnn_param.with_bn ? 4 : 0);
}

template <typename DType>
static void QuantizeConvWeightBias(NDArray *weight, NDArray *bias,
                                   bool has_bias, float data_scale,
                                   bool weight_channelwise_scale,
                                   std::vector<float> *weight_scales) {
  using red::limits::MaxValue;
  using red::limits::MinValue;
  const DType *weight_ptr = weight->data().dptr<DType>();
  NDArray quantized_weight = NDArray(weight->storage_type(), weight->shape(),
                                     weight->ctx(), true, mshadow::kInt8);
  int8_t *quan_weight_ptr = quantized_weight.data().dptr<int8_t>();
  size_t channel = weight->shape()[0];

  // TODO(Zhennan): Handle the case weight is not in dims 4.
  size_t offset = weight->shape()[1] * weight->shape()[2] * weight->shape()[3];
  std::vector<DType> weight_c_min(channel, MaxValue<DType>());
  std::vector<DType> weight_c_max(channel, MinValue<DType>());
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int c = 0; c < static_cast<int>(channel); ++c) {
    const DType *p1 = weight_ptr + c * offset;
    for (size_t k = 0; k < offset; ++k) {
      if (weight_c_min[c] > p1[k])
        weight_c_min[c] = p1[k];
      if (weight_c_max[c] < p1[k])
        weight_c_max[c] = p1[k];
    }
  }

  if (weight_channelwise_scale) {
    weight_scales->resize(channel);
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int c = 0; c < static_cast<int>(channel); ++c) {
      DType weight_range = MaxAbs(weight_c_min[c], weight_c_max[c]);
      weight_scales->at(c) = kInt8Range / weight_range;
      const DType *fp_ptr = weight_ptr + c * offset;
      int8_t *quan_ptr = quan_weight_ptr + c * offset;
      for (size_t k = 0; k < offset; ++k) {
        quan_ptr[k] = std::round(weight_scales->at(c) * fp_ptr[k]);
      }
    }
  } else {
    DType total_min = weight_c_min[0];
    DType total_max = weight_c_max[0];
    for (size_t c = 0; c < channel; ++c) {
      if (total_min > weight_c_min[c]) total_min = weight_c_min[c];
      if (total_max < weight_c_max[c]) total_max = weight_c_max[c];
    }
    weight_scales->resize(1);
    DType weight_range = MaxAbs(total_min, total_max);
    weight_scales->at(0) = kInt8Range / weight_range;
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
    for (int c = 0; c < static_cast<int>(channel); ++c) {
      const DType *fp_ptr = weight_ptr + c * offset;
      int8_t *quan_ptr = quan_weight_ptr + c * offset;
      for (size_t k = 0; k < offset; ++k) {
        quan_ptr[k] = std::round(weight_scales->at(0) * fp_ptr[k]);
      }
    }
  }

  *weight = quantized_weight;
  if (has_bias) {
    const DType *bias_ptr = bias->data().dptr<DType>();
    NDArray quantized_bias = NDArray(bias->storage_type(), bias->shape(),
                                     bias->ctx(), true, mshadow::kInt32);
    int32_t *quan_bias_ptr = quantized_bias.data().dptr<int32_t>();
    for (size_t c = 0; c < channel; ++c) {
      auto weight_scale =
          weight_channelwise_scale ? weight_scales->at(c) : weight_scales->at(0);
      float bias_scale = weight_scale * data_scale;
      quan_bias_ptr[c] = std::round(bias_scale * bias_ptr[c]);
    }
    *bias = quantized_bias;
  }
}

static void ConvFusionFallBackCompute() {
  LOG(FATAL) << "Don't know how to do ConvFusionFallBackCompute!";
}

static void ConvolutionFusionComputeExCPU(const MKLDNNConvFullParam &full_param,
                                          const OpContext &ctx,
                                          MKLDNNConvForward *fwd,
                                          const std::vector<NDArray> &inputs,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &outputs) {
  if (SupportMKLDNNConv(full_param.conv_param, inputs[0])) {
    MKLDNNConvolutionForwardFullFeature(full_param, ctx, fwd, inputs, req, outputs);
    return;
  }
  ConvFusionFallBackCompute();
}

class SgMKLDNNConvOperator {
 public:
  explicit SgMKLDNNConvOperator(const nnvm::NodeAttrs &attrs)
      : initalized_(false),
        subgraph_sym_(*attrs.subgraphs[0]),
        param_(nnvm::get<MKLDNNConvFusionParam>(attrs.parsed)),
        inplace_(false) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx, const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn Conv only supports "
                  "inference computation.";
  }

 private:
  bool initalized_;
  nnvm::Symbol subgraph_sym_;
  MKLDNNConvFusionParam param_;
  std::shared_ptr<MKLDNNConvForward> fwd_;
  NDArray cached_weight_;
  NDArray cached_bias_;
  float cached_data_min_;
  float cached_data_max_;
  float cached_sum_min_;
  float cached_sum_max_;
  size_t weight_ver_;
  size_t bias_ver_;
  std::vector<float> weight_scales_;
  bool inplace_;
};

void SgMKLDNNConvOperator::Forward(const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs) {
  auto &full_conv_param = param_.full_conv_param;
  auto &mkldnn_param = param_.full_conv_param.mkldnn_param;
  auto &conv_param = param_.full_conv_param.conv_param;
  auto bn_param = param_.bn_param.get();
  size_t input_size =
      2 + (conv_param.no_bias ? 0 : 1) + (mkldnn_param.with_bn ? 4 : 0) +
      (mkldnn_param.with_sum ? 1 : 0) +
      (mkldnn_param.quantized ? 2 + (full_conv_param.mkldnn_param.with_sum ? 2 : 0) : 0);
  CHECK_EQ(inputs.size(), input_size);
  size_t idx = 0;

  auto in_data = idx++;
  auto in_weight = idx++;
  auto in_bias = conv_param.no_bias ? 0 : (idx++);
  auto in_gamma = mkldnn_param.with_bn ? (idx++) : 0;
  auto in_beta = mkldnn_param.with_bn ? (idx++) : 0;
  auto in_mean = mkldnn_param.with_bn ? (idx++) : 0;
  auto in_var = mkldnn_param.with_bn ? (idx++) : 0;
  auto in_sum = mkldnn_param.with_sum ? (idx++) : 0;
  float data_min =
      mkldnn_param.quantized ? inputs[idx++].data().dptr<float>()[0] : 0.0;
  float data_max =
      mkldnn_param.quantized ? inputs[idx++].data().dptr<float>()[0] : 0.0;
  float sum_min = (mkldnn_param.with_sum && mkldnn_param.quantized)
                      ? inputs[idx++].data().dptr<float>()[0]
                      : 0.0;
  float sum_max = (mkldnn_param.with_sum && mkldnn_param.quantized)
                      ? inputs[idx++].data().dptr<float>()[0]
                      : 0.0;
  float *out_min_ptr =
      mkldnn_param.quantized ? outputs[kMin].data().dptr<float>() : nullptr;
  float *out_max_ptr =
      mkldnn_param.quantized ? outputs[kMax].data().dptr<float>() : nullptr;
  CHECK_EQ(input_size, idx);
  bool has_bias = mkldnn_param.with_bn || !conv_param.no_bias;
  NDArray data = inputs[in_data];
  NDArray output = mkldnn_param.with_sum ? inputs[in_sum] : outputs[kOut];

  // Copy inputs[in_sum] into outputs[kOut] in case inplace optimization failed.
  if (mkldnn_param.with_sum) {
    if (!initalized_) {
      auto in_mkl_mem = inputs[in_sum].GetMKLDNNData();
      auto out_mkl_mem = outputs[kOut].GetMKLDNNData();
      // TODO(zhennan): Currently, mkldnn fallback mechanism will break inplace option,
      // which make check (req[kOut] == kWriteInplace) useless.
      if (in_mkl_mem->get_data_handle() == out_mkl_mem->get_data_handle()) {
        inplace_ = true;
      }
    }
    if (!inplace_) {
      auto in_mkl_mem = inputs[in_sum].GetMKLDNNData();
      const_cast<NDArray &>(outputs[kOut]).CopyFrom(*in_mkl_mem);
      output = NDArray(outputs[kOut].GetMKLDNNData());
    }
  }

  // Check input change
  // TODO(zhennan): Only update cached_* changed.
  if (initalized_) {
    if (mkldnn_param.with_bn) {
      if (weight_ver_ != inputs[in_weight].version() ||
          ((!conv_param.no_bias) && bias_ver_ != inputs[in_bias].version())) {
        initalized_ = false;
      }
    }
    if (initalized_ && mkldnn_param.quantized) {
      if (cached_data_min_ != data_min || cached_data_max_ != data_max ||
          cached_sum_min_ != sum_min || cached_sum_max_ != sum_max ||
          weight_ver_ != inputs[in_weight].version() ||
          ((!conv_param.no_bias) && bias_ver_ != inputs[in_bias].version())) {
        initalized_ = false;
      }
    }
  }
  bool post_requantize = false;
  if (mkldnn_param.quantized) {
    if (mkldnn_param.min_calib_range.has_value() &&
        mkldnn_param.max_calib_range.has_value()) {
      post_requantize = true;
      mkldnn_param.weight_channelwise_scale = true;
      *out_min_ptr = mkldnn_param.min_calib_range.value();
      *out_max_ptr = mkldnn_param.max_calib_range.value();
    } else {
      mkldnn_param.weight_channelwise_scale = false;
    }
  }

  if (!initalized_) {
    cached_data_min_ = data_min;
    cached_data_max_ = data_max;
    cached_sum_min_ = sum_min;
    cached_sum_max_ = sum_max;
    full_conv_param.sum_scale = 1.0;
    cached_weight_ = inputs[in_weight].Reorder2Default();
    weight_ver_ = inputs[in_weight].version();
    if (!conv_param.no_bias) {
      cached_bias_ = inputs[in_bias].Reorder2Default();
      bias_ver_ = inputs[in_bias].version();
    } else {
      cached_bias_ = NDArray();
    }

    // Update weight and bias after bn fusion.
    if (mkldnn_param.with_bn) {
      CHECK_EQ(inputs[in_weight].dtype(), inputs[in_gamma].dtype());
      CHECK_EQ(inputs[in_weight].dtype(), inputs[in_beta].dtype());
      CHECK_EQ(inputs[in_weight].dtype(), inputs[in_var].dtype());
      MSHADOW_REAL_TYPE_SWITCH(inputs[in_weight].dtype(), DType, {
        UpdateConvWeightBias<DType>(&cached_weight_, &cached_bias_,
                                    conv_param.no_bias, inputs[in_gamma],
                                    inputs[in_beta], inputs[in_mean],
                                    inputs[in_var], bn_param);
      });
    }
    // Quantize weight and bias.
    if (mkldnn_param.quantized) {
      CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
      auto data_range = (data.dtype() == mshadow::kInt8) ? kInt8Range : kUint8Range;
      float data_scale = data_range / MaxAbs(cached_data_min_, cached_data_max_);
      MSHADOW_REAL_TYPE_SWITCH(cached_weight_.dtype(), DType, {
        QuantizeConvWeightBias<DType>(&cached_weight_, &cached_bias_,
                                      has_bias, data_scale,
                                      mkldnn_param.weight_channelwise_scale,
                                      &weight_scales_);
      });
      // Collect scale.
      size_t channel = cached_weight_.shape()[0];
      float sum_in_scale = 1.0;
      float out_range;
      float quantized_out_range;
      float output_scale;
      if (cached_data_min_ < 0.0) {
        // TODO(zhennan): Support int8 input when mkldnn supports.
        LOG(FATAL) << "Can't handle negetive value for QuantizeData";
      }
      if (mkldnn_param.with_sum) {
        auto quantized_sum_range = cached_sum_min_ < 0 ? kInt8Range : kUint8Range;
        sum_in_scale = quantized_sum_range / MaxAbs(cached_sum_min_, cached_sum_max_);
      }
      if (post_requantize) {
        quantized_out_range =
            IsOutputUInt8(mkldnn_param) ? kUint8Range : kInt8Range;
        out_range = MaxAbs(*out_min_ptr, *out_max_ptr);
        output_scale = quantized_out_range / out_range;
        full_conv_param.requantize_scales.resize(channel);
        for (size_t c = 0; c < channel; c++) {
          auto weight_scale = mkldnn_param.weight_channelwise_scale
                                  ? weight_scales_[c]
                                  : weight_scales_[0];
          full_conv_param.requantize_scales[c] =
              output_scale / data_scale / weight_scale;
        }
      } else {
        output_scale = data_scale * weight_scales_[0];
        full_conv_param.requantize_scales.resize(0);
      }
      if (mkldnn_param.with_sum)
        full_conv_param.sum_scale = output_scale / sum_in_scale;
    }
    fwd_.reset(new MKLDNNConvForward(
        full_conv_param, ctx.is_train, data, cached_weight_,
        has_bias ? &cached_bias_ : nullptr, output));
  }
  initalized_ = true;
  std::vector<NDArray> new_inputs;
  std::vector<OpReqType> new_req;
  if (has_bias) {
    new_inputs = {data, cached_weight_, cached_bias_};
    new_req = {req[in_data], req[in_weight], req[in_bias]};
  } else {
    new_inputs = {data, cached_weight_};
    new_req = {req[in_data], req[in_weight]};
  }
  ConvolutionFusionComputeExCPU(full_conv_param, ctx, fwd_.get(), new_inputs,
                                new_req, {output});

  if (mkldnn_param.with_sum) {
    auto out = const_cast<NDArray &>(outputs[kOut]);
    out.UpdateMKLDNNMemDesc();
  }
}

static void SgMKLDNNConvOpForward(const OpStatePtr &state_ptr,
                                  const OpContext &ctx,
                                  const std::vector<NDArray> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<NDArray> &outputs) {
  SgMKLDNNConvOperator &op = state_ptr.get_state<SgMKLDNNConvOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

static uint32_t SgMKLDNNConvNumInputs(const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  auto num_input = DefaultSubgraphOpNumInputs(attrs);
  if (param.full_conv_param.mkldnn_param.quantized)
    return num_input + 2 + param.full_conv_param.mkldnn_param.with_sum ? 2 : 0;
  else
    return num_input;
}

static void SgMKLDNNConvParamParser(nnvm::NodeAttrs *attrs) {
  MKLDNNConvFusionParam param_;
  try {
    param_.full_conv_param.mkldnn_param.Init(attrs->dict);
  } catch (const dmlc::ParamError &e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto &k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  auto subgraph_sym = attrs->subgraphs[0];
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::NodePtr &node) {
    if (node->is_variable()) return;
    auto &node_name = node->op()->name;
    if (node_name == "BatchNorm") {
      CHECK_EQ(param_.full_conv_param.mkldnn_param.with_bn, true);
      CHECK(param_.bn_param.get() == nullptr);
      param_.bn_param = std::make_shared<BatchNormParam>(
          nnvm::get<BatchNormParam>(node->attrs.parsed));
    } else if (node_name == "Convolution") {
      param_.full_conv_param.conv_param =
          nnvm::get<ConvolutionParam>(node->attrs.parsed);
    }
  });
  attrs->parsed = std::move(param_);
}

static std::vector<std::string> SgMKLDNNConvListInputNames(
    const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  std::vector<std::string> input_names = DefaultSubgraphOpListInputs(attrs);
  if (param.full_conv_param.mkldnn_param.quantized) {
    input_names.emplace_back("data_min");
    input_names.emplace_back("data_max");
    if (param.full_conv_param.mkldnn_param.with_sum) {
      input_names.emplace_back("sum_min");
      input_names.emplace_back("sum_max");
    }
  }
  return input_names;
}

static std::vector<std::string> SgMKLDNNConvListOutputNames(
    const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized)
    return std::vector<std::string>{"output", "output_min", "output_max"};
  else
    return std::vector<std::string>{"output"};
}

static OpStatePtr CreateSgMKLDNNConvState(const nnvm::NodeAttrs &attrs,
                                          Context ctx,
                                          const std::vector<TShape> &in_shapes,
                                          const std::vector<int> &in_types) {
  return OpStatePtr::Create<SgMKLDNNConvOperator>(attrs);
}

template <typename DType>
static void FilterMinMaxIndice(const MKLDNNConvParam &mkldnn_param,
                               std::vector<DType> *in_shapes,
                               std::vector<DType> *out_shapes,
                               std::vector<DType> *base_in_shapes,
                               std::vector<DType> *base_out_shapes,
                               std::unordered_set<size_t> *minmax_indice) {
  base_out_shapes->push_back(out_shapes->at(0));
  size_t last = in_shapes->size() - 1;
  if (mkldnn_param.with_sum) {
    minmax_indice->insert(last);
    minmax_indice->insert(last - 1);
    minmax_indice->insert(last - 2);
    minmax_indice->insert(last - 3);
    *base_in_shapes =
        std::vector<DType>(in_shapes->begin(), in_shapes->end() - 4);
  } else {
    minmax_indice->insert(last);
    minmax_indice->insert(last - 1);
    *base_in_shapes =
        std::vector<DType>(in_shapes->begin(), in_shapes->end() - 2);
  }
}

static bool SgMKLDNNConvInferShape(const nnvm::NodeAttrs &attrs,
                                   std::vector<TShape> *in_shapes,
                                   std::vector<TShape> *out_shapes) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    std::vector<TShape> base_in_shapes;
    std::vector<TShape> base_out_shapes;

    FilterMinMaxIndice<TShape>(param.full_conv_param.mkldnn_param, in_shapes,
                               out_shapes, &base_in_shapes, &base_out_shapes,
                               &minmax_indice);
    bool result =
        DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);
    size_t base_idx = 0;
    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (minmax_indice.count(i)) {
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
      } else {
        in_shapes->at(i) = base_in_shapes[base_idx++];
      }
    }
    out_shapes->at(0) = base_out_shapes[0];
    SHAPE_ASSIGN_CHECK(*out_shapes, 1, Shape1(1));
    SHAPE_ASSIGN_CHECK(*out_shapes, 2, Shape1(1));
    return result;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgMKLDNNConvInferType(const nnvm::NodeAttrs &attrs,
                                  std::vector<int> *in_types,
                                  std::vector<int> *out_types) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    std::vector<int> base_in_types;
    std::vector<int> base_out_types;
    FilterMinMaxIndice<int>(param.full_conv_param.mkldnn_param, in_types,
                            out_types, &base_in_types, &base_out_types,
                            &minmax_indice);
    // Override data type to fp32 for default infer type as bn doesn't support
    // uint8.
    int orig_data = base_in_types[0];
    base_in_types[0] = mshadow::kFloat32;
    int orig_sum = base_in_types[0];
    if (param.full_conv_param.mkldnn_param.with_sum) {
      auto sum_index = GetInSumIndex(param);
      orig_sum = base_in_types[sum_index];
      base_in_types[sum_index] = mshadow::kFloat32;
    }
    bool result = DefaultSubgraphOpType(attrs, &base_in_types, &base_out_types);
    base_in_types[0] = orig_data;
    if (param.full_conv_param.mkldnn_param.with_sum) {
      auto sum_index = GetInSumIndex(param);
      base_in_types[sum_index] = orig_sum;
    }
    size_t base_idx = 0;
    for (size_t i = 0; i < in_types->size(); ++i) {
      if (minmax_indice.count(i)) {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
      } else {
        in_types->at(i) = base_in_types[base_idx++];
      }
    }
    if (param.full_conv_param.mkldnn_param.min_calib_range.has_value() &&
        param.full_conv_param.mkldnn_param.max_calib_range.has_value()) {
      if (IsOutputUInt8(param.full_conv_param.mkldnn_param)) {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kUint8);
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
      }
    } else {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
    }

    TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
    TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    return result;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

static bool SgMKLDNNConvOpStorageType(const nnvm::NodeAttrs &attrs,
                                      const int dev_mask,
                                      DispatchMode *dispatch_mode,
                                      std::vector<int> *in_stypes,
                                      std::vector<int> *out_stypes) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    std::vector<int> base_in_stypes;
    std::vector<int> base_out_stypes;
    FilterMinMaxIndice<int>(param.full_conv_param.mkldnn_param, in_stypes,
                            out_stypes, &base_in_stypes, &base_out_stypes,
                            &minmax_indice);
    bool result = DefaultSubgraphOpStorageType(
        attrs, dev_mask, dispatch_mode, &base_in_stypes, &base_out_stypes);
    size_t base_idx = 0;
    for (size_t i = 0; i < in_stypes->size(); ++i) {
      if (minmax_indice.count(i)) {
        type_assign(&in_stypes->at(i), mxnet::kDefaultStorage);
      } else {
        in_stypes->at(i) = base_in_stypes[base_idx++];
      }
    }
    out_stypes->at(0) = base_out_stypes[0];
    type_assign(&out_stypes->at(1), mxnet::kDefaultStorage);
    type_assign(&out_stypes->at(2), mxnet::kDefaultStorage);
    return result;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        in_stypes, out_stypes);
  }
}

std::vector<std::pair<int, int>> SgMKLDNNConvInplaceOption(
    const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.with_sum) {
    return std::vector<std::pair<int, int>>{{GetInSumIndex(param), 0}};
  } else {
    return std::vector<std::pair<int, int>>();
  }
}

nnvm::NodePtr SgMKLDNNConvQuantizedOp(const NodeAttrs& attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  nnvm::NodePtr node = nnvm::Node::Create();
  node->attrs.op = Op::Get("_sg_mkldnn_conv");
  CHECK_EQ(param.full_conv_param.conv_param.kernel.ndim(), 2U)
      << "Quantized Convolution of MKL-DNN only supports 2D kernel currently."
      <<  "Please exclude this layer from the quantized model.";
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["quantized"] = "true";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

bool SgMKLDNNAvoidQuantizeInput(const NodeAttrs &attrs, size_t index) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  std::unordered_set<size_t> avoid_indice;
  size_t idx = 0;
  idx++;                         // data
  avoid_indice.insert(idx++);    // weight
  if (!param.full_conv_param.conv_param.no_bias) {
    avoid_indice.insert(idx++);  // bias
  }
  if (param.full_conv_param.mkldnn_param.with_bn) {
    avoid_indice.insert(idx++);  // gamma
    avoid_indice.insert(idx++);  // beta
    avoid_indice.insert(idx++);  // mean
    avoid_indice.insert(idx++);  // var
  }
  return avoid_indice.count(index);
}

NNVM_REGISTER_OP(_sg_mkldnn_conv)
.describe(R"code(_sg_mkldnn_conv)code" ADD_FILELINE)
.set_num_inputs(SgMKLDNNConvNumInputs)
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  return param.full_conv_param.mkldnn_param.quantized ? 3 : 1;
})
.set_attr_parser(SgMKLDNNConvParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames", SgMKLDNNConvListInputNames)
.set_attr<nnvm::FListOutputNames>("FListOutputNames", SgMKLDNNConvListOutputNames)
.set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNConvState)
.set_attr<nnvm::FInferShape>("FInferShape", SgMKLDNNConvInferShape)
.set_attr<nnvm::FInferType>("FInferType", SgMKLDNNConvInferType)
.set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNConvOpStorageType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNConvOpForward)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                DefaultSubgraphOpMutableInputs)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FInplaceOption>("FInplaceOption", SgMKLDNNConvInplaceOption)
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNConvQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", SgMKLDNNAvoidQuantizeInput);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
