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

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <utility>
#include <vector>

#include "operator/nn/dnnl/dnnl_act-inl.h"
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/subgraph/common.h"
#include "dnnl_common.h"
#include "dnnl_conv-inl.h"

namespace mxnet {
namespace op {

using red::limits::MaxValue;
using red::limits::MinValue;

template <typename DType>
static void UpdateConvWeightBias(NDArray* weight,
                                 NDArray* bias,
                                 bool no_bias,
                                 const NDArray& gamma,
                                 const NDArray& beta,
                                 const NDArray& mean,
                                 const NDArray& variance,
                                 const BatchNormParam* param) {
  NDArray update_weight =
      NDArray(weight->storage_type(), weight->shape(), weight->ctx(), true, weight->dtype());
  NDArray update_bias =
      NDArray(beta.storage_type(), beta.shape(), beta.ctx(), true, weight->dtype());
  const DType* weight_ptr  = weight->data().dptr<DType>();
  const DType* bias_ptr    = no_bias ? nullptr : bias->data().dptr<DType>();
  const float* gamma_ptr   = gamma.data().dptr<float>();
  const float* beta_ptr    = beta.data().dptr<float>();
  const float* mean_ptr    = mean.data().dptr<float>();
  const float* var_ptr     = variance.data().dptr<float>();
  DType* update_weight_ptr = update_weight.data().dptr<DType>();
  DType* update_bias_ptr   = update_bias.data().dptr<DType>();
  index_t channel          = static_cast<index_t>(gamma.shape()[0]);
  const auto wshape        = weight->shape();
  size_t offset            = wshape.ProdShape(1, wshape.ndim());
#pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t c = 0; c < channel; ++c) {
    const DType* p1 = weight_ptr + c * offset;
    DType* p2       = update_weight_ptr + c * offset;
    float alpha     = (param->fix_gamma ? 1.0f : gamma_ptr[c]) / sqrt(var_ptr[c] + param->eps);

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
  *bias   = update_bias;
}

static inline size_t GetInSumIndex(const DNNLConvFusionParam& param) {
  if (param.full_conv_param.dnnl_param.dedup_sum) {
    return 0;
  }
  return 2 + (param.full_conv_param.conv_param.no_bias ? 0 : 1) +
         (param.full_conv_param.dnnl_param.with_bn ? 4 : 0);
}

class SgDNNLConvOperator {
 public:
  explicit SgDNNLConvOperator(const nnvm::NodeAttrs& attrs)
      : subgraph_sym_(*attrs.subgraphs[0]), param_(nnvm::get<DNNLConvFusionParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  bool initialized_{false};
  bool inplace_{false};
  bool post_requantize_{false};
  nnvm::Symbol subgraph_sym_;
  DNNLConvFusionParam param_;
  std::shared_ptr<DNNLConvForward> fwd_;
  dnnl_args_map_t args_;
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

void SgDNNLConvOperator::Forward(const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  auto& full_conv_param = param_.full_conv_param;
  auto& dnnl_param      = full_conv_param.dnnl_param;
  auto& conv_param      = full_conv_param.conv_param;
  auto bn_param         = param_.bn_param.get();

  index_t idx = 0;

  auto in_data   = idx++;
  auto in_weight = idx++;
  auto in_bias   = conv_param.no_bias ? 0 : (idx++);
  auto in_gamma  = dnnl_param.with_bn ? (idx++) : 0;
  auto in_beta   = dnnl_param.with_bn ? (idx++) : 0;
  auto in_mean   = dnnl_param.with_bn ? (idx++) : 0;
  auto in_var    = dnnl_param.with_bn ? (idx++) : 0;
  auto in_sum    = dnnl_param.with_sum ? (dnnl_param.dedup_sum ? in_data : idx++) : -1;
  float data_min = dnnl_param.quantized ? inputs[idx++].data().dptr<float>()[0] : 0.0;
  float data_max = dnnl_param.quantized ? inputs[idx++].data().dptr<float>()[0] : 0.0;
  float sum_min  = 0.0f;
  float sum_max  = 0.0f;
  if (dnnl_param.with_sum && dnnl_param.quantized) {
    if (dnnl_param.dedup_sum) {
      sum_min = data_min;
      sum_max = data_max;
    } else {
      sum_min = inputs[idx++].data().dptr<float>()[0];
      sum_max = inputs[idx++].data().dptr<float>()[0];
    }
  }
  CHECK_EQ(inputs.size(), idx);
  bool has_bias  = dnnl_param.with_bn || !conv_param.no_bias;
  NDArray data   = inputs[in_data];
  NDArray output = dnnl_param.with_sum ? inputs[in_sum] : outputs[kOut];
  // Copy inputs[in_sum] into outputs[kOut] in case inplace optimization failed.
  if (dnnl_param.with_sum) {
    if (!initialized_) {
      // TODO(zhennan): Currently, dnnl fallback mechanism will break inplace option,
      // which make check (req[kOut] == kWriteInplace) useless.
      auto in_dnnl_mem  = inputs[in_sum].GetDNNLData();
      auto out_dnnl_mem = outputs[kOut].GetDNNLData();
      if (in_dnnl_mem->get_data_handle() == out_dnnl_mem->get_data_handle()) {
        inplace_ = true;
      }
    }
    if (!inplace_) {
      auto in_dnnl_mem  = inputs[in_sum].GetDNNLData();
      auto out_dnnl_mem = outputs[kOut].GetDNNLData();
      if (outputs[kOut].dtype() == mshadow::kInt32 || outputs[kOut].dtype() == mshadow::kFloat32) {
        const auto& mem_desc  = in_dnnl_mem->get_desc();
        const auto this_dtype = get_dnnl_type(outputs[kOut].dtype());
        auto omd              = mem_desc;
        omd.data.data_type    = static_cast<dnnl_data_type_t>(this_dtype);
        dnnl_mem_ptr tmp_mem(
            new dnnl::memory(omd, CpuEngine::Get()->get_engine(), out_dnnl_mem->get_data_handle()));
        DNNLStream::Get()->RegisterMem(tmp_mem);
        DNNLStream::Get()->RegisterPrimArgs(
            dnnl::reorder(*in_dnnl_mem, *tmp_mem),
            {{DNNL_ARG_FROM, *in_dnnl_mem}, {DNNL_ARG_TO, *tmp_mem}});
        output = NDArray(tmp_mem);
      } else {
        dnnl_mem_ptr tmp_mem(new dnnl::memory(in_dnnl_mem->get_desc(),
                                              CpuEngine::Get()->get_engine(),
                                              out_dnnl_mem->get_data_handle()));
        DNNLStream::Get()->RegisterMem(tmp_mem);
        DNNLMemoryCopy(*in_dnnl_mem, tmp_mem.get());
        output = NDArray(tmp_mem);
      }
    }
  }

  // Check input change
  // TODO(zhennan): Only update cached_* changed.
  if (initialized_) {
    if (dnnl_param.with_bn) {
      if (weight_ver_ != inputs[in_weight].version() ||
          ((!conv_param.no_bias) && bias_ver_ != inputs[in_bias].version())) {
        initialized_ = false;
      }
    }
    if (initialized_ && dnnl_param.quantized) {
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
    cached_sum_min_  = sum_min;
    cached_sum_max_  = sum_max;
    cached_weight_   = inputs[in_weight].Reorder2Default();
    weight_ver_      = inputs[in_weight].version();
    if (!conv_param.no_bias) {
      cached_bias_ = inputs[in_bias];
      bias_ver_    = inputs[in_bias].version();
    } else {
      cached_bias_ = NDArray();
    }

    // Update weight and bias after bn fusion.
    if (dnnl_param.with_bn) {
      DNNL_REAL_TYPE_SWITCH(inputs[in_weight].dtype(), DType, {
        UpdateConvWeightBias<DType>(&cached_weight_,
                                    &cached_bias_,
                                    conv_param.no_bias,
                                    inputs[in_gamma],
                                    inputs[in_beta],
                                    inputs[in_mean],
                                    inputs[in_var],
                                    bn_param);
      });
    }
    // Quantize weight and bias.
    if (dnnl_param.quantized) {
      CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
      if (cached_data_min_ < 0.0f) {
        CHECK_EQ(data.dtype(), mshadow::kInt8)
            << "Expect int8 when data_min < 0.0, consider quantize model with int8.";
      }
      auto weight_channelwise_scale = false;
      if (dnnl_param.min_calib_range.has_value() && dnnl_param.max_calib_range.has_value()) {
        cached_output_min_       = dnnl_param.min_calib_range.value();
        cached_output_max_       = dnnl_param.max_calib_range.value();
        post_requantize_         = true;
        weight_channelwise_scale = true;
      }
      if (dnnl_param.enabled_float_output.has_value()) {
        weight_channelwise_scale = true;
      }
      data_scale_ = GetQuantizeScale(data.dtype(), cached_data_min_, cached_data_max_);
      DNNL_REAL_TYPE_SWITCH(cached_weight_.dtype(), DType, {
        weight_scales_ = GetWeightScales<DType>(cached_weight_,
                                                has_bias ? &cached_bias_ : nullptr,
                                                data_scale_,
                                                weight_channelwise_scale);
      });
      // Collect scale.
      size_t channel     = cached_weight_.shape()[0];
      float sum_in_scale = 1.0;
      float output_scale;
      if (dnnl_param.with_sum) {
        sum_in_scale = GetQuantizeScale(inputs[in_sum].dtype(), cached_sum_min_, cached_sum_max_);
      }
      if (post_requantize_ || dnnl_param.enabled_float_output.has_value()) {
        if (post_requantize_) {
          output_scale = GetQuantizeScale(IsOutputUInt8(param_) ? mshadow::kUint8 : mshadow::kInt8,
                                          cached_output_min_,
                                          cached_output_max_);
        } else {
          output_scale = 1.0;
        }
        full_conv_param.requantize_scales.resize(weight_channelwise_scale ? channel : 1);
        for (size_t c = 0; c < full_conv_param.requantize_scales.size(); c++) {
          full_conv_param.requantize_scales[c] = 1.0 / data_scale_ / weight_scales_[c];
        }
        if (dnnl_param.with_act) {
          full_conv_param.act_param.scale = output_scale;
        } else {
          for (size_t c = 0; c < full_conv_param.requantize_scales.size(); c++) {
            full_conv_param.requantize_scales[c] *= output_scale;
          }
        }
      } else {
        Stream<cpu>* s = ctx.get_stream<cpu>();
        if (data.dtype() == mshadow::kInt8) {
          mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
              s,
              1,
              &cached_output_min_,
              &cached_output_max_,
              &weight_scales_[1],
              &weight_scales_[2],
              &cached_data_min_,
              &cached_data_max_);
        } else {
          mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
              s,
              1,
              &cached_output_min_,
              &cached_output_max_,
              &weight_scales_[1],
              &weight_scales_[2],
              &cached_data_min_,
              &cached_data_max_);
        }
        weight_scales_.resize(1);
        output_scale = data_scale_ * weight_scales_[0];
        full_conv_param.requantize_scales.resize(0);
      }
      if (dnnl_param.with_sum) {
        full_conv_param.sum_scale = output_scale / sum_in_scale;
      }
      if (dnnl_param.with_act &&
          full_conv_param.act_param.alg == dnnl::algorithm::eltwise_bounded_relu) {
        if (dnnl_param.with_sum) {
          LOG(ERROR) << "oneDNN doesn't support conv + relu + sum fusion yet.";
          full_conv_param.act_param.alpha *= output_scale;
        }
      }
    }
    fwd_.reset(new DNNLConvForward(full_conv_param,
                                   ctx.is_train,
                                   data,
                                   cached_weight_,
                                   has_bias ? &cached_bias_ : nullptr,
                                   output));
    dnnl::memory::desc bias_md;
    if (has_bias)
      bias_md = fwd_->GetPd().bias_desc();
    ConvertWeightBias2DNNL(&cached_weight_,
                           &cached_bias_,
                           has_bias,
                           fwd_->GetPd().weights_desc(),
                           has_bias ? &bias_md : nullptr,
                           conv_param.num_group,
                           data_scale_,
                           weight_scales_);
    args_[DNNL_ARG_SRC]     = *data.GetDNNLData();
    args_[DNNL_ARG_WEIGHTS] = *cached_weight_.GetDNNLData();
    if (has_bias)
      args_[DNNL_ARG_BIAS] = *cached_bias_.GetDNNLData();
    args_[DNNL_ARG_DST] = *output.GetDNNLData();
    initialized_        = true;
  }

  if (dnnl_param.with_sum) {
    const auto& output_mem   = output.GetDNNLData();
    const auto& out_mem_desc = output_mem->get_desc();
    const auto& dst_mem_desc = fwd_->GetPd().dst_desc();
    if (out_mem_desc != dst_mem_desc) {
      auto tmp_out_mem       = output.GetDNNLDataReorder(&dst_mem_desc);
      auto data_md           = dst_mem_desc;
      data_md.data.data_type = static_cast<dnnl_data_type_t>(out_mem_desc.data.data_type);
      dnnl_mem_ptr new_out_mem(
          new dnnl::memory(data_md, CpuEngine::Get()->get_engine(), output_mem->get_data_handle()));
      DNNLStream::Get()->RegisterMem(new_out_mem);
      DNNLMemoryCopy(*tmp_out_mem, new_out_mem.get());
      output = NDArray(new_out_mem);
    }
  }

  if (dnnl_param.quantized) {
    auto fwd_src_desc    = fwd_->GetPd().src_desc();
    auto data_mem        = data.GetDNNLDataReorder(&fwd_src_desc);
    auto fwd_pd_dst_desc = fwd_->GetPd().dst_desc();
    dnnl::memory* mem    = output.CreateDNNLData(&fwd_pd_dst_desc);
    args_[DNNL_ARG_SRC]  = *data_mem;
    args_[DNNL_ARG_DST]  = *mem;
    DNNLStream::Get()->RegisterPrimArgs(fwd_->GetFwd(), args_);
    DNNLStream::Get()->Submit();
  } else {
    std::vector<NDArray> new_inputs;
    if (has_bias) {
      new_inputs = {data, cached_weight_, cached_bias_};
    } else {
      new_inputs = {data, cached_weight_};
    }
    DNNLConvolutionForwardFullFeature(full_conv_param, ctx, fwd_.get(), new_inputs, req, {output});
  }

  if (dnnl_param.quantized && !dnnl_param.enabled_float_output.has_value()) {
    *outputs[kMin].data().dptr<float>() = cached_output_min_;
    *outputs[kMax].data().dptr<float>() = cached_output_max_;
  }
  if (dnnl_param.with_sum) {
    auto out          = const_cast<NDArray&>(outputs[kOut]);
    auto fwd_dst_desc = fwd_->GetPd().dst_desc();
    out.UpdateDNNLMemDesc(&fwd_dst_desc);
  }
}

static void SgDNNLConvOpForward(const OpStatePtr& state_ptr,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  SgDNNLConvOperator& op = state_ptr.get_state<SgDNNLConvOperator>();
  op.Forward(ctx, inputs, req, outputs);
}

static uint32_t SgDNNLConvNumInputs(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  auto num_input    = DefaultSubgraphOpNumInputs(attrs);
  if (param.full_conv_param.dnnl_param.quantized)
    return num_input + 2 +
           (param.full_conv_param.dnnl_param.with_sum &&
                    !param.full_conv_param.dnnl_param.dedup_sum ?
                2 :
                0);
  else
    return num_input;
}

static void SgDNNLConvParamParser(nnvm::NodeAttrs* attrs) {
  DNNLConvFusionParam param_;

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
    param_.full_conv_param.dnnl_param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  CHECK_EQ(attrs->subgraphs.size(), 1);
  auto subgraph_sym = attrs->subgraphs[0];
  bool with_act     = false;
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::ObjectPtr& node) {
    if (node->is_variable())
      return;
    auto& node_name = node->op()->name;
    if (node_name == "BatchNorm") {
      CHECK_EQ(param_.full_conv_param.dnnl_param.with_bn, true);
      CHECK(param_.bn_param.get() == nullptr);
      param_.bn_param =
          std::make_shared<BatchNormParam>(nnvm::get<BatchNormParam>(node->attrs.parsed));
    } else if (node_name == "Convolution") {
      param_.full_conv_param.conv_param = nnvm::get<ConvolutionParam>(node->attrs.parsed);
    } else if (node_name == "Activation" || node_name == "LeakyReLU" || node_name == "clip") {
      auto& post_act_param = (param_.full_conv_param.dnnl_param.with_act && !with_act) ?
                                 param_.full_conv_param.act_param :
                                 param_.full_conv_param.postsum_act_param;
      if (node_name == "Activation") {
        const auto act_param = nnvm::get<ActivationParam>(node->attrs.parsed);
        post_act_param.alg   = GetDNNLActAlgo(act_param);
      } else if (node_name == "LeakyReLU") {
        const auto act_param = nnvm::get<LeakyReLUParam>(node->attrs.parsed);
        post_act_param.alpha = act_param.slope;
        post_act_param.alg   = GetDNNLActAlgo(act_param);
      } else {
        const auto clip_param = nnvm::get<ClipParam>(node->attrs.parsed);
        post_act_param.alg    = dnnl::algorithm::eltwise_bounded_relu;
        post_act_param.alpha  = clip_param.a_max;
      }
      with_act = true;
    }
  });
  attrs->parsed = std::move(param_);
}

static std::vector<std::string> SgDNNLConvListInputNames(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  std::vector<std::string> input_names;
  input_names.emplace_back("data");
  input_names.emplace_back("weight");
  if (!param.full_conv_param.conv_param.no_bias) {
    input_names.emplace_back("bias");
  }
  if (param.full_conv_param.dnnl_param.with_bn) {
    input_names.emplace_back("gamma");
    input_names.emplace_back("beta");
    input_names.emplace_back("mean");
    input_names.emplace_back("var");
  }
  auto& dnnl_param = param.full_conv_param.dnnl_param;
  if (dnnl_param.with_sum && !dnnl_param.dedup_sum) {
    input_names.emplace_back("sum");
  }
  if (param.full_conv_param.dnnl_param.quantized) {
    input_names.emplace_back("data_min");
    input_names.emplace_back("data_max");
    if (dnnl_param.with_sum && !dnnl_param.dedup_sum) {
      input_names.emplace_back("sum_min");
      input_names.emplace_back("sum_max");
    }
  }
  CHECK_EQ(input_names.size(), SgDNNLConvNumInputs(attrs));
  return input_names;
}

static std::vector<std::string> SgDNNLConvListOutputNames(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.quantized &&
      !param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
    return std::vector<std::string>{"output", "output_min", "output_max"};
  } else {
    return std::vector<std::string>{"output"};
  }
}

static OpStatePtr CreateSgDNNLConvState(const nnvm::NodeAttrs& attrs,
                                        Context ctx,
                                        const mxnet::ShapeVector& in_shapes,
                                        const std::vector<int>& in_types) {
  return OpStatePtr::Create<SgDNNLConvOperator>(attrs);
}

template <typename DType>
static void FilterMinMaxIndice(const DNNLConvParam& dnnl_param,
                               std::vector<DType>* in_shapes,
                               std::vector<DType>* out_shapes,
                               std::vector<DType>* base_in_shapes,
                               std::vector<DType>* base_out_shapes,
                               std::unordered_set<size_t>* minmax_indice) {
  base_out_shapes->push_back(out_shapes->at(0));
  size_t last = in_shapes->size() - 1;
  if (dnnl_param.with_sum && !dnnl_param.dedup_sum) {
    minmax_indice->insert(last);
    minmax_indice->insert(last - 1);
    minmax_indice->insert(last - 2);
    minmax_indice->insert(last - 3);
    *base_in_shapes = std::vector<DType>(in_shapes->begin(), in_shapes->end() - 4);
  } else {
    minmax_indice->insert(last);
    minmax_indice->insert(last - 1);
    *base_in_shapes = std::vector<DType>(in_shapes->begin(), in_shapes->end() - 2);
  }
}

static bool SgDNNLConvInferShape(const nnvm::NodeAttrs& attrs,
                                 mxnet::ShapeVector* in_shapes,
                                 mxnet::ShapeVector* out_shapes) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes;

    FilterMinMaxIndice<mxnet::TShape>(param.full_conv_param.dnnl_param,
                                      in_shapes,
                                      out_shapes,
                                      &base_in_shapes,
                                      &base_out_shapes,
                                      &minmax_indice);
    bool result     = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);
    size_t base_idx = 0;
    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (minmax_indice.count(i)) {
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
      } else {
        in_shapes->at(i) = base_in_shapes[base_idx++];
      }
    }
    out_shapes->at(0) = base_out_shapes[0];
    if (!param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
      SHAPE_ASSIGN_CHECK(*out_shapes, 1, Shape1(1));
      SHAPE_ASSIGN_CHECK(*out_shapes, 2, Shape1(1));
    }
    return result;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgDNNLConvInferType(const nnvm::NodeAttrs& attrs,
                                std::vector<int>* in_types,
                                std::vector<int>* out_types) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.quantized) {
    if (in_types->at(0) == mshadow::kBfloat16) {
      return false;
    }

    std::unordered_set<size_t> minmax_indice;
    std::vector<int> base_in_types;
    std::vector<int> base_out_types;
    FilterMinMaxIndice<int>(param.full_conv_param.dnnl_param,
                            in_types,
                            out_types,
                            &base_in_types,
                            &base_out_types,
                            &minmax_indice);
    // Override data type to fp32 for default infer type as bn doesn't support
    // uint8.
    int orig_data    = base_in_types[0];
    base_in_types[0] = mshadow::kFloat32;
    int orig_sum     = base_in_types[0];
    auto& dnnl_param = param.full_conv_param.dnnl_param;
    if (param.full_conv_param.dnnl_param.with_sum && !dnnl_param.dedup_sum) {
      auto sum_index           = GetInSumIndex(param);
      orig_sum                 = base_in_types[sum_index];
      base_in_types[sum_index] = mshadow::kFloat32;
    }
    bool result      = DefaultSubgraphOpType(attrs, &base_in_types, &base_out_types);
    base_in_types[0] = orig_data;
    if (param.full_conv_param.dnnl_param.with_sum && !dnnl_param.dedup_sum) {
      auto sum_index           = GetInSumIndex(param);
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

    if (param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
      TYPE_ASSIGN_CHECK(
          *out_types, 0, param.full_conv_param.dnnl_param.enabled_float_output.value());
    } else {
      if (param.full_conv_param.dnnl_param.min_calib_range.has_value() &&
          param.full_conv_param.dnnl_param.max_calib_range.has_value()) {
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
    }
    return result;
  } else {
    bool result = DefaultSubgraphOpType(attrs, in_types, out_types);
    if (param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
      (*out_types)[0] = param.full_conv_param.dnnl_param.enabled_float_output.value();
    }
    return result;
  }
}

static bool SgDNNLConvOpStorageType(const nnvm::NodeAttrs& attrs,
                                    const int dev_mask,
                                    DispatchMode* dispatch_mode,
                                    std::vector<int>* in_stypes,
                                    std::vector<int>* out_stypes) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.quantized) {
    std::unordered_set<size_t> minmax_indice;
    std::vector<int> base_in_stypes;
    std::vector<int> base_out_stypes;
    FilterMinMaxIndice<int>(param.full_conv_param.dnnl_param,
                            in_stypes,
                            out_stypes,
                            &base_in_stypes,
                            &base_out_stypes,
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
    if (!param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
      type_assign(&out_stypes->at(1), mxnet::kDefaultStorage);
      type_assign(&out_stypes->at(2), mxnet::kDefaultStorage);
    }
    return result;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode, in_stypes, out_stypes);
  }
}

std::vector<std::pair<int, int>> SgDNNLConvInplaceOption(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  if (param.full_conv_param.dnnl_param.with_sum && !param.full_conv_param.dnnl_param.dedup_sum) {
    return std::vector<std::pair<int, int>>{{GetInSumIndex(param), 0}};
  } else {
    return std::vector<std::pair<int, int>>();
  }
}

nnvm::ObjectPtr SgDNNLConvQuantizedOp(const NodeAttrs& attrs) {
  auto const& param    = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs.op       = Op::Get("_sg_onednn_conv");
  const int k_ndims    = param.full_conv_param.conv_param.kernel.ndim();
  CHECK(k_ndims == 2U || k_ndims == 3U)
      << "Quantized Convolution of oneDNN supports 2D/3D kernel currently."
      << "Please exclude this layer from the quantized model.";
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["quantized"] = "true";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

bool SgDNNLAvoidConvQuantizeInput(const NodeAttrs& attrs,
                                  const size_t index,
                                  const std::string quantize_granularity) {
  auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
  std::unordered_set<size_t> avoid_indice;
  size_t idx = 0;
  idx++;                       // data
  avoid_indice.insert(idx++);  // weight
  if (!param.full_conv_param.conv_param.no_bias) {
    avoid_indice.insert(idx++);  // bias
  }
  if (param.full_conv_param.dnnl_param.with_bn) {
    avoid_indice.insert(idx++);  // gamma
    avoid_indice.insert(idx++);  // beta
    avoid_indice.insert(idx++);  // mean
    avoid_indice.insert(idx++);  // var
  }
  return avoid_indice.count(index);
}

NNVM_REGISTER_OP(_sg_onednn_conv)
    .add_alias("_sg_mkldnn_conv")
    .describe(R"code(_sg_onednn_conv)code" ADD_FILELINE)
    .set_num_inputs(SgDNNLConvNumInputs)
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto const& param = nnvm::get<DNNLConvFusionParam>(attrs.parsed);
      if (param.full_conv_param.dnnl_param.quantized &&
          !param.full_conv_param.dnnl_param.enabled_float_output.has_value()) {
        return 3;
      }
      return 1;
    })
    .set_attr_parser(SgDNNLConvParamParser)
    .set_attr<nnvm::FListInputNames>("FListInputNames", SgDNNLConvListInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames", SgDNNLConvListOutputNames)
    .set_attr<FCreateOpState>("FCreateOpState", CreateSgDNNLConvState)
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLConvInferShape)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLConvInferType)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLConvOpStorageType)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgDNNLConvOpForward)
    .set_attr<bool>("TIsDNNL", true)
    // TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
    // will be reverted after the improvement of CachedOP is done.
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FMutateInputs>("FMutateInputs", DefaultSubgraphOpMutableInputs)
    .set_attr<std::string>("key_var_num_args", "num_args")
    .set_attr<nnvm::FInplaceOption>("FInplaceOption", SgDNNLConvInplaceOption)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLConvQuantizedOp)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", SgDNNLAvoidConvQuantizeInput);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
