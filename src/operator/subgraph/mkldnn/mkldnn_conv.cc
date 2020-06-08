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
#include "../../nn/mkldnn/mkldnn_act-inl.h"
#include "../../tensor/matrix_op-inl.h"
#include "mkldnn_common.h"

namespace mxnet {
namespace op {

using red::limits::MaxValue;
using red::limits::MinValue;

template <typename DType>
static void UpdateConvWeightBias(NDArray *weight, NDArray *bias, bool no_bias,
                                 const NDArray &gamma, const NDArray &beta,
                                 const NDArray &mean, const NDArray &variance,
                                 const BatchNormParam *param) {
  NDArray update_weight = NDArray(weight->storage_type(), weight->shape(),
                                  weight->ctx(), true, weight->dtype());
  NDArray update_bias = NDArray(beta.storage_type(), beta.shape(), beta.ctx(),
                                true, weight->dtype());
  const DType *weight_ptr = weight->data().dptr<DType>();
  const DType *bias_ptr = no_bias ? nullptr : bias->data().dptr<DType>();
  const float *gamma_ptr = gamma.data().dptr<float>();
  const float *beta_ptr = beta.data().dptr<float>();
  const float *mean_ptr = mean.data().dptr<float>();
  const float *var_ptr = variance.data().dptr<float>();
  DType *update_weight_ptr = update_weight.data().dptr<DType>();
  DType *update_bias_ptr = update_bias.data().dptr<DType>();
  size_t channel = gamma.shape()[0];
  const auto wshape = weight->shape();
  size_t offset = wshape.ProdShape(1, wshape.ndim());
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (int c = 0; c < static_cast<int>(channel); ++c) {
    const DType *p1 = weight_ptr + c * offset;
    DType *p2 = update_weight_ptr + c * offset;
    float alpha = (param->fix_gamma ? 1.0f : gamma_ptr[c]) / sqrt(var_ptr[c] + param->eps);

    if (bias_ptr)
      update_bias_ptr[c] =
          static_cast<DType>(beta_ptr[c] + alpha * (static_cast<float>(bias_ptr[c]) - mean_ptr[c]));
    else
      update_bias_ptr[c] = static_cast<DType>(beta_ptr[c] - alpha * mean_ptr[c]);

    for (size_t k = 0; k < offset; ++k) {
      p2[k] = static_cast<DType>(static_cast<float>(p1[k]) * alpha);
    }
  }
  *weight = update_weight;
  *bias = update_bias;
}

static inline size_t GetInSumIndex(const MKLDNNConvFusionParam &param) {
  return 2 + (param.full_conv_param.conv_param.no_bias ? 0 : 1) +
         (param.full_conv_param.mkldnn_param.with_bn ? 4 : 0);
}

class SgMKLDNNConvOperator {
 public:
  explicit SgMKLDNNConvOperator(const nnvm::NodeAttrs &attrs)
      : subgraph_sym_(*attrs.subgraphs[0]),
        param_(nnvm::get<MKLDNNConvFusionParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

 private:
  bool initialized_{false};
  bool inplace_{false};
  bool post_requantize_{false};
  nnvm::Symbol subgraph_sym_;
  MKLDNNConvFusionParam param_;
  std::shared_ptr<MKLDNNConvForward> fwd_;
  mkldnn_args_map_t args_;
  NDArray cached_weight_;
  NDArray cached_bias_;
  float cached_data_min_;
  float cached_data_max_;
  float cached_sum_min_;
  float cached_sum_max_;
  float cached_output_min_;
  float cached_output_max_;
  size_t weight_ver_;
  size_t bias_ver_;
  float data_scale_{0.0f};
  std::vector<float> weight_scales_;
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
  CHECK_EQ(input_size, idx);
  bool has_bias = mkldnn_param.with_bn || !conv_param.no_bias;
  NDArray data = inputs[in_data];
  NDArray output = mkldnn_param.with_sum ? inputs[in_sum] : outputs[kOut];

  // Copy inputs[in_sum] into outputs[kOut] in case inplace optimization failed.
  if (mkldnn_param.with_sum) {
    if (!initialized_) {
      // TODO(zhennan): Currently, mkldnn fallback mechanism will break inplace option,
      // which make check (req[kOut] == kWriteInplace) useless.
      auto in_mkl_mem = inputs[in_sum].GetMKLDNNData();
      auto out_mkl_mem = outputs[kOut].GetMKLDNNData();
      if (in_mkl_mem->get_data_handle() == out_mkl_mem->get_data_handle()) {
        inplace_ = true;
      }
    }
    if (!inplace_) {
      auto in_mkl_mem = inputs[in_sum].GetMKLDNNData();
      auto out_mkl_mem = outputs[kOut].GetMKLDNNData();
      if (outputs[kOut].dtype() == mshadow::kInt32) {
        const auto& mem_desc = in_mkl_mem->get_desc();
        const auto this_dtype = get_mkldnn_type(mshadow::kInt32);
        auto omd = mem_desc;
        omd.data.data_type = static_cast<mkldnn_data_type_t>(this_dtype);
        mkldnn_mem_ptr tmp_mem(new mkldnn::memory(omd, CpuEngine::Get()->get_engine(),
                                                  out_mkl_mem->get_data_handle()));
        MKLDNNStream::Get()->RegisterMem(tmp_mem);
        MKLDNNStream::Get()->RegisterPrimArgs(
            mkldnn::reorder(*in_mkl_mem, *tmp_mem),
            {{MKLDNN_ARG_FROM, *in_mkl_mem}, {MKLDNN_ARG_TO, *tmp_mem}});
        output = NDArray(tmp_mem);
      } else {
        mkldnn_mem_ptr tmp_mem(new mkldnn::memory(in_mkl_mem->get_desc(),
                                                  CpuEngine::Get()->get_engine(),
                                                  out_mkl_mem->get_data_handle()));
        MKLDNNStream::Get()->RegisterMem(tmp_mem);
        MKLDNNMemoryCopy(*in_mkl_mem, tmp_mem.get());
        output = NDArray(tmp_mem);
      }
    }
  }

  // Check input change
  // TODO(zhennan): Only update cached_* changed.
  if (initialized_) {
    if (mkldnn_param.with_bn) {
      if (weight_ver_ != inputs[in_weight].version() ||
          ((!conv_param.no_bias) && bias_ver_ != inputs[in_bias].version())) {
        initialized_ = false;
      }
    }
    if (initialized_ && mkldnn_param.quantized) {
      if (cached_data_min_ != data_min || cached_data_max_ != data_max ||
          cached_sum_min_ != sum_min || cached_sum_max_ != sum_max ||
          weight_ver_ != inputs[in_weight].version() ||
          ((!conv_param.no_bias) && bias_ver_ != inputs[in_bias].version())) {
        initialized_ = false;
      }
    }
  }
  if (!initialized_) {
    cached_data_min_ = data_min;
    cached_data_max_ = data_max;
    cached_sum_min_ = sum_min;
    cached_sum_max_ = sum_max;
    cached_weight_ = inputs[in_weight].Reorder2Default();
    weight_ver_ = inputs[in_weight].version();
    if (!conv_param.no_bias) {
      cached_bias_ = inputs[in_bias];
      bias_ver_ = inputs[in_bias].version();
    } else {
      cached_bias_ = NDArray();
    }

    // Update weight and bias after bn fusion.
    if (mkldnn_param.with_bn) {
      MKLDNN_REAL_TYPE_SWITCH(inputs[in_weight].dtype(), DType, {
        UpdateConvWeightBias<DType>(&cached_weight_, &cached_bias_,
                                    conv_param.no_bias, inputs[in_gamma],
                                    inputs[in_beta], inputs[in_mean],
                                    inputs[in_var], bn_param);
      });
    }
    // Quantize weight and bias.
    if (mkldnn_param.quantized) {
      CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
      if (cached_data_min_ < 0.0f) {
        CHECK_EQ(data.dtype(), mshadow::kInt8)
            << "Expect int8 when data_min < 0.0, consider quantize model with int8.";
      }
      auto weight_channelwise_scale = false;
      if (mkldnn_param.min_calib_range.has_value() && mkldnn_param.max_calib_range.has_value()) {
        cached_output_min_ = mkldnn_param.min_calib_range.value();
        cached_output_max_ = mkldnn_param.max_calib_range.value();
        post_requantize_ = true;
        weight_channelwise_scale = true;
      }
      data_scale_ = GetQuantizeScale(data.dtype(), cached_data_min_, cached_data_max_);
      MKLDNN_REAL_TYPE_SWITCH(cached_weight_.dtype(), DType, {
        weight_scales_ = GetWeightScales<DType>(cached_weight_, has_bias ? &cached_bias_ : nullptr,
                                                data_scale_, weight_channelwise_scale);
      });
      // Collect scale.
      size_t channel = cached_weight_.shape()[0];
      float sum_in_scale = 1.0;
      float output_scale;
      if (mkldnn_param.with_sum) {
        sum_in_scale = GetQuantizeScale(inputs[in_sum].dtype(), cached_sum_min_, cached_sum_max_);
      }
      if (post_requantize_) {
        output_scale = GetQuantizeScale(IsOutputUInt8(param_) ? mshadow::kUint8 : mshadow::kInt8,
                                        cached_output_min_, cached_output_max_);
        full_conv_param.requantize_scales.resize(weight_channelwise_scale ? channel : 1);
        for (size_t c = 0; c < full_conv_param.requantize_scales.size(); c++) {
          full_conv_param.requantize_scales[c] = output_scale / data_scale_ / weight_scales_[c];
        }
      } else {
        Stream<cpu> *s = ctx.get_stream<cpu>();
        if (data.dtype() == mshadow::kInt8) {
          mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
              s, 1, &cached_output_min_, &cached_output_max_, &weight_scales_[1],
              &weight_scales_[2], &cached_data_min_, &cached_data_max_);
        } else {
          mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
              s, 1, &cached_output_min_, &cached_output_max_, &weight_scales_[1],
              &weight_scales_[2], &cached_data_min_, &cached_data_max_);
        }
        weight_scales_.resize(1);
        output_scale = data_scale_ * weight_scales_[0];
        full_conv_param.requantize_scales.resize(0);
      }
      if (mkldnn_param.with_sum) {
        full_conv_param.sum_scale = output_scale / sum_in_scale;
      }
      if (mkldnn_param.with_act &&
          full_conv_param.act_param.alg == mkldnn::algorithm::eltwise_bounded_relu) {
        if (mkldnn_param.with_sum) {
          LOG(ERROR) << "mkldnn doesn't support conv + relu + sum fusion yet.";
          full_conv_param.act_param.alpha *= output_scale;
        } else {
          // For conv+relu6 without sum, we don't need post_ops as output_scale can do the cut off.
          mkldnn_param.with_act = false;
        }
      }
      if (mkldnn_param.with_postsum_act) {
        CHECK(full_conv_param.postsum_act_param.alg == mkldnn::algorithm::eltwise_relu);
      }
    }
    fwd_.reset(new MKLDNNConvForward(
        full_conv_param, ctx.is_train, data, cached_weight_,
        has_bias ? &cached_bias_ : nullptr, output));
    mkldnn::memory::desc bias_md;
    if (has_bias) bias_md = fwd_->GetPd().bias_desc();
    ConvertWeightBias2MKLDNN(&cached_weight_, &cached_bias_, has_bias,
                             fwd_->GetPd().weights_desc(),
                             has_bias ? & bias_md : nullptr,
                             full_conv_param.conv_param.num_group,
                             data_scale_, weight_scales_);
    args_[MKLDNN_ARG_SRC] = *data.GetMKLDNNData();
    args_[MKLDNN_ARG_WEIGHTS] = *cached_weight_.GetMKLDNNData();
    if (has_bias) args_[MKLDNN_ARG_BIAS] = *cached_bias_.GetMKLDNNData();
    args_[MKLDNN_ARG_DST] = *output.GetMKLDNNData();
    initialized_ = true;
  }

  if (mkldnn_param.with_sum) {
    const auto& output_mem = output.GetMKLDNNData();
    const auto& out_mem_desc = output_mem->get_desc();
    const auto& dst_mem_desc = fwd_->GetPd().dst_desc();
    if (out_mem_desc != dst_mem_desc) {
      auto tmp_out_mem = output.GetMKLDNNDataReorder(fwd_->GetPd().dst_desc());
      auto data_md = dst_mem_desc;
      data_md.data.data_type = static_cast<mkldnn_data_type_t>(out_mem_desc.data.data_type);
      mkldnn_mem_ptr new_out_mem(new mkldnn::memory(data_md, CpuEngine::Get()->get_engine(),
                                                    output_mem->get_data_handle()));
      MKLDNNStream::Get()->RegisterMem(new_out_mem);
      MKLDNNMemoryCopy(*tmp_out_mem, new_out_mem.get());
      output = NDArray(new_out_mem);
    }
  }

  if (mkldnn_param.quantized) {
    auto data_mem = data.GetMKLDNNDataReorder(fwd_->GetPd().src_desc());
    mkldnn::memory *mem = output.CreateMKLDNNData(fwd_->GetPd().dst_desc());
    args_[MKLDNN_ARG_SRC] = *data_mem;
    args_[MKLDNN_ARG_DST] = *mem;
    MKLDNNStream::Get()->RegisterPrimArgs(fwd_->GetFwd(), args_);
    MKLDNNStream::Get()->Submit();
  } else {
    std::vector<NDArray> new_inputs;
    if (has_bias) {
      new_inputs = {data, cached_weight_, cached_bias_};
    } else {
      new_inputs = {data, cached_weight_};
    }
    MKLDNNConvolutionForwardFullFeature(full_conv_param, ctx, fwd_.get(), new_inputs, req,
                                        {output});
  }

  if (mkldnn_param.quantized) {
    *outputs[kMin].data().dptr<float>() = cached_output_min_;
    *outputs[kMax].data().dptr<float>() = cached_output_max_;
  }
  if (mkldnn_param.with_sum) {
    auto out = const_cast<NDArray &>(outputs[kOut]);
    out.UpdateMKLDNNMemDesc(fwd_->GetPd().dst_desc());
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
    return num_input + 2 + (param.full_conv_param.mkldnn_param.with_sum ? 2 : 0);
  else
    return num_input;
}

static void SgMKLDNNConvParamParser(nnvm::NodeAttrs *attrs) {
  MKLDNNConvFusionParam param_;

  // For back-compatible, rename
  // with_relu -> with_act
  // with_postsum_relu -> with_postsum_act

  auto old = attrs->dict.find("with_relu");
  if (old != attrs->dict.end()) {
    attrs->dict["with_act"] = old->second;
    attrs->dict.erase(old);
  }

  old = attrs->dict.find("with_postsum_relu");
  if (old != attrs->dict.end()) {
    attrs->dict["with_postsum_act"] = old->second;
    attrs->dict.erase(old);
  }

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
  CHECK_EQ(attrs->subgraphs.size(), 1);
  auto subgraph_sym = attrs->subgraphs[0];
  bool with_act = false;
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::ObjectPtr &node) {
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
    } else if (node_name == "Activation" || node_name == "LeakyReLU" || node_name == "clip") {
      auto &post_act_param =
          (param_.full_conv_param.mkldnn_param.with_act && !with_act)
              ? param_.full_conv_param.act_param
              : param_.full_conv_param.postsum_act_param;
      with_act = true;
      if (node_name == "Activation") {
        const auto act_param = nnvm::get<ActivationParam>(node->attrs.parsed);
        post_act_param.alg = GetMKLDNNActAlgo(act_param);
      } else if (node_name == "LeakyReLU") {
        const auto act_param = nnvm::get<LeakyReLUParam>(node->attrs.parsed);
        post_act_param.alpha = act_param.slope;
        post_act_param.alg = GetMKLDNNActAlgo(act_param);
      } else {
        const auto clip_param = nnvm::get<ClipParam>(node->attrs.parsed);
        post_act_param.alg = mkldnn::algorithm::eltwise_bounded_relu;
        post_act_param.alpha = clip_param.a_max;
      }
    }
  });
  attrs->parsed = std::move(param_);
}

static std::vector<std::string> SgMKLDNNConvListInputNames(const NodeAttrs &attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  std::vector<std::string> input_names;
  input_names.emplace_back("data");
  input_names.emplace_back("weight");
  if (!param.full_conv_param.conv_param.no_bias) {
    input_names.emplace_back("bias");
  }
  if (param.full_conv_param.mkldnn_param.with_bn) {
    input_names.emplace_back("gamma");
    input_names.emplace_back("beta");
    input_names.emplace_back("mean");
    input_names.emplace_back("var");
  }
  if (param.full_conv_param.mkldnn_param.with_sum) {
    input_names.emplace_back("sum");
  }
  if (param.full_conv_param.mkldnn_param.quantized) {
    input_names.emplace_back("data_min");
    input_names.emplace_back("data_max");
    if (param.full_conv_param.mkldnn_param.with_sum) {
      input_names.emplace_back("sum_min");
      input_names.emplace_back("sum_max");
    }
  }
  CHECK_EQ(input_names.size(), SgMKLDNNConvNumInputs(attrs));
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
                                          const mxnet::ShapeVector &in_shapes,
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
                                   mxnet::ShapeVector *in_shapes,
                                   mxnet::ShapeVector *out_shapes) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.mkldnn_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes;

    FilterMinMaxIndice<mxnet::TShape>(param.full_conv_param.mkldnn_param, in_shapes,
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
      if (IsOutputUInt8(param)) {
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

nnvm::ObjectPtr SgMKLDNNConvQuantizedOp(const NodeAttrs& attrs) {
  auto const &param = nnvm::get<MKLDNNConvFusionParam>(attrs.parsed);
  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs.op = Op::Get("_sg_mkldnn_conv");
  const int k_ndims = param.full_conv_param.conv_param.kernel.ndim();
  CHECK(k_ndims == 2U || k_ndims == 3U)
      << "Quantized Convolution of MKL-DNN supports 2D/3D kernel currently."
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

bool SgMKLDNNAvoidConvQuantizeInput(const NodeAttrs &attrs, const size_t index,
                                    const std::string quantize_granularity) {
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
.set_attr<mxnet::FInferShape>("FInferShape", SgMKLDNNConvInferShape)
.set_attr<nnvm::FInferType>("FInferType", SgMKLDNNConvInferType)
.set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNConvOpStorageType)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNConvOpForward)
.set_attr<bool>("TIsMKLDNN", true)
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                DefaultSubgraphOpMutableInputs)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<nnvm::FInplaceOption>("FInplaceOption", SgMKLDNNConvInplaceOption)
.set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
    return QuantizeType::kMust;
})
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNConvQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", SgMKLDNNAvoidConvQuantizeInput);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
