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
 * \file dnnl_fc.cc
 * \brief DNNL (Quantized) FullyConnected operator based on subgraph
 * \author Ciyong Chen
 */

#if MXNET_USE_ONEDNN == 1

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "operator/nn/dnnl/dnnl_act-inl.h"
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_fully_connected-inl.h"
#include "operator/quantization/quantization_utils.h"
#include "operator/tensor/matrix_op-inl.h"
#include "operator/subgraph/common.h"
#include "dnnl_common.h"
#include "dnnl_fc-inl.h"

namespace mxnet {
namespace op {

class SgDNNLFCOp {
 public:
  explicit SgDNNLFCOp(const nnvm::NodeAttrs& attrs)
      : subgraph_sym_(*attrs.subgraphs[0]),
        attrs(attrs),
        full_param_(nnvm::get<DNNLFCFullParam>(attrs.parsed)) {}

  void Forward(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

  void Backward(const OpContext& ctx,
                const std::vector<NDArray>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<NDArray>& outputs) {
    LOG(FATAL) << "Not implemented: subgraph oneDNN fully connected only supports "
                  "inference computation.";
  }

 private:
  enum { kDataMin = 0, kDataMax, kWeightMin, kWeightMax, kBiasMin, kBiasMax, kSumMin, kSumMax };
  const size_t MIN_MAX_COUNT = 8;
  const float u8_to_s8_scale = 0.5;

  NDArray PrepareOutputWithSum(const NDArray& sum_input, const NDArray& output);
  bool CheckInitializationConditions(const std::vector<NDArray>& inputs,
                                     const std::vector<float>& min_max_vec,
                                     bool is_channel_wise);
  bool PrepareQuantization(const OpContext& ctx,
                           const std::vector<NDArray>& in_data,
                           const NDArray& output,
                           const std::vector<float>& min_max_vec);
  dnnl::memory::desc CreateOutputMemoryDesc(const mxnet::TShape& oshape, int out_dtype);
  void GetCachedWeightsAndBias(const NDArray& weight,
                               bool support_channelwise_scale,
                               bool has_bias);
  nnvm::Symbol subgraph_sym_;
  nnvm::NodeAttrs attrs;
  DNNLFCFullParam full_param_;
  bool initialized_{false};
  bool reorder_data_{false};
  bool inplace_{false};
  dnnl_args_map_t args_;
  std::shared_ptr<DNNLFullyConnectedForward> fwd_;
  std::shared_ptr<dnnl::memory> cached_data_mem_;
  std::shared_ptr<dnnl::memory> cached_out_mem_;
  NDArray cached_weight_;
  NDArray cached_bias_;
  float cached_data_min_;
  float cached_data_max_;
  float cached_weight_min_;
  float cached_weight_max_;
  float cached_sum_min_;
  float cached_sum_max_;
  float cached_bias_min_;
  float cached_bias_max_;
  size_t weight_ver_;
  size_t bias_ver_;
  float cached_output_min_;
  float cached_output_max_;
  float data_scale_{0.0f};
  std::vector<float> weight_scales_;
};

void SgDNNLFCOp::Forward(const OpContext& ctx,
                         const std::vector<NDArray>& in_data,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& out_data) {
  const FCInputIndex idx(full_param_);
  CHECK_EQ(in_data.size(), idx.GetTotal());
  const int out_index     = 0;
  const int out_min_index = 1;
  const int out_max_index = 2;

  const auto& default_param = full_param_.default_param;
  const auto& dnnl_param    = full_param_.dnnl_param;

  NDArray output;
  if (dnnl_param.with_sum) {
    output = PrepareOutputWithSum(in_data[idx.sum], out_data[out_index]);
  } else {
    output = out_data[out_index];
  }

  if (!dnnl_param.quantized) {
    // dispatch to float version
    DNNLFCForwardImpl(full_param_, ctx, in_data, req, {output});
    return;
  }

  const bool has_bias = !default_param.no_bias;
  const bool channel_wise =
      dnnl_param.channel_wise_quantize.has_value() && dnnl_param.channel_wise_quantize.value();

  std::vector<float> min_max_vec(MIN_MAX_COUNT);
  min_max_vec[kDataMin]   = 0.0f;
  min_max_vec[kDataMax]   = 0.0f;
  min_max_vec[kWeightMin] = 0.0f;
  min_max_vec[kWeightMax] = 0.0f;
  min_max_vec[kBiasMin]   = 0.0f;
  min_max_vec[kBiasMax]   = 0.0f;

  min_max_vec[kSumMin]  = idx.sum_min ? in_data[idx.sum_min].data().dptr<float>()[0] : 0.0f;
  min_max_vec[kSumMax]  = idx.sum_max ? in_data[idx.sum_max].data().dptr<float>()[0] : 0.0f;
  NDArray data          = in_data[idx.data];
  const NDArray& weight = in_data[idx.weight];

  if (!channel_wise) {
    min_max_vec[kWeightMin] = in_data[idx.weight_min].data().dptr<float>()[0];
    min_max_vec[kWeightMax] = in_data[idx.weight_max].data().dptr<float>()[0];
    if (has_bias) {
      min_max_vec[kBiasMin] = in_data[idx.bias_min].data().dptr<float>()[0];
      min_max_vec[kBiasMax] = in_data[idx.bias_max].data().dptr<float>()[0];
    }
  }
  min_max_vec[kDataMin] = in_data[idx.data_min].data().dptr<float>()[0];
  min_max_vec[kDataMax] = in_data[idx.data_max].data().dptr<float>()[0];

  initialized_ = CheckInitializationConditions(in_data, min_max_vec, channel_wise);

  if (!initialized_) {
    const auto engine  = CpuEngine::Get()->get_engine();
    cached_data_min_   = min_max_vec[kDataMin];
    cached_data_max_   = min_max_vec[kDataMax];
    cached_weight_min_ = min_max_vec[kWeightMin];
    cached_weight_max_ = min_max_vec[kWeightMax];
    weight_ver_        = weight.version();
    cached_weight_     = weight;
    cached_sum_min_    = min_max_vec[kSumMin];
    cached_sum_max_    = min_max_vec[kSumMax];
    if (has_bias) {
      cached_bias_min_ = min_max_vec[kBiasMin];
      cached_bias_max_ = min_max_vec[kBiasMax];
      bias_ver_        = in_data[idx.bias].version();
      cached_bias_     = in_data[idx.bias];
    } else {
      cached_bias_ = NDArray();
    }
    const mxnet::TShape ishape = data.shape();
    const auto data_ndim       = ishape.ndim();
    if (data.IsDNNLData()) {
      reorder_data_ = true;
      data          = data.Reorder2Default();
    }
    if (data_ndim != 2) {
      if (!default_param.flatten) {
        data =
            data.DNNLDataReshape(Shape2(ishape.ProdShape(0, data_ndim - 1), ishape[data_ndim - 1]));
      } else {
        data = data.DNNLDataReshape(Shape2(ishape[0], ishape.ProdShape(1, data_ndim)));
      }
    }

    // create cached out_md
    dnnl::memory::desc out_md = CreateOutputMemoryDesc(output.shape(), output.dtype());
    cached_out_mem_           = std::make_shared<dnnl::memory>(out_md, engine);

    bool support_channelwise_scale = PrepareQuantization(ctx, in_data, output, min_max_vec);

    fwd_.reset(new DNNLFullyConnectedForward(full_param_,
                                             ctx.is_train,
                                             data,
                                             cached_weight_,
                                             (has_bias ? &cached_bias_ : nullptr),
                                             out_md));
    GetCachedWeightsAndBias(weight, support_channelwise_scale, has_bias);

    const auto data_mem = static_cast<const dnnl::memory*>(data.GetDNNLData());
    cached_data_mem_    = std::make_shared<dnnl::memory>(data_mem->get_desc(), engine);

    args_[DNNL_ARG_SRC]     = *cached_data_mem_;
    args_[DNNL_ARG_WEIGHTS] = *static_cast<const dnnl::memory*>(cached_weight_.GetDNNLData());
    if (has_bias)
      args_[DNNL_ARG_BIAS] = *static_cast<const dnnl::memory*>(cached_bias_.GetDNNLData());
    args_[DNNL_ARG_DST] = *cached_out_mem_;
    initialized_        = true;
  }

  if (reorder_data_) {
    data = data.Reorder2Default();
  }

  cached_data_mem_->set_data_handle(reinterpret_cast<void*>(data.data().dptr_));
  cached_out_mem_->set_data_handle(reinterpret_cast<void*>(output.data().dptr_));

  DNNLStream::Get()->RegisterPrimArgs(fwd_->GetFwd(), args_);
  DNNLStream::Get()->Submit();

  if (!dnnl_param.enabled_float_output.has_value()) {
    float* output_min_ptr = out_data[out_min_index].data().dptr<float>();
    float* output_max_ptr = out_data[out_max_index].data().dptr<float>();

    *output_min_ptr = cached_output_min_;
    *output_max_ptr = cached_output_max_;
  }
}

NDArray SgDNNLFCOp::PrepareOutputWithSum(const NDArray& sum_input, const NDArray& output) {
  if (!initialized_) {
    // TODO(zhennan): Currently, dnnl fallback mechanism will break inplace option,
    // which make check (req[out_index] == kWriteInplace) useless.
    auto in_dnnl_mem  = static_cast<const dnnl::memory*>(sum_input.GetDNNLData());
    auto out_dnnl_mem = static_cast<const dnnl::memory*>(output.GetDNNLData());
    if (in_dnnl_mem->get_data_handle() == out_dnnl_mem->get_data_handle() &&
        sum_input.dtype() == output.dtype()) {
      inplace_ = true;
    }
  }
  if (inplace_) {
    return sum_input;
  } else {
    // Not in place: copy sum_input into output.
    auto in_dnnl_mem  = static_cast<const dnnl::memory*>(sum_input.GetDNNLData());
    auto out_dnnl_mem = static_cast<const dnnl::memory*>(output.GetDNNLData());
    if (output.dtype() == mshadow::kInt32) {
      auto mem_desc           = in_dnnl_mem->get_desc();
      auto this_dtype         = get_dnnl_type(mshadow::kInt32);
      mem_desc.data.data_type = static_cast<dnnl_data_type_t>(this_dtype);
      dnnl_mem_ptr tmp_mem(new dnnl::memory(
          mem_desc, CpuEngine::Get()->get_engine(), out_dnnl_mem->get_data_handle()));
      DNNLStream::Get()->RegisterMem(tmp_mem);
      DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(*in_dnnl_mem, *tmp_mem),
                                          {{DNNL_ARG_FROM, *in_dnnl_mem}, {DNNL_ARG_TO, *tmp_mem}});
      return NDArray(tmp_mem);
    } else if (sum_input.dtype() == mshadow::kUint8 && output.dtype() == mshadow::kInt8) {
      auto sum_mem_desc           = in_dnnl_mem->get_desc();
      auto out_dtype              = get_dnnl_type(mshadow::kInt8);
      sum_mem_desc.data.data_type = static_cast<dnnl_data_type_t>(out_dtype);
      dnnl_mem_ptr tmp_mem(new dnnl::memory(
          sum_mem_desc, CpuEngine::Get()->get_engine(), out_dnnl_mem->get_data_handle()));
      DNNLStream::Get()->RegisterMem(tmp_mem);
      std::vector<float> reorder_scale = {u8_to_s8_scale};
      dnnl::primitive_attr reorder_attr;
      reorder_attr.set_output_scales(0, reorder_scale);
      const auto reorder_pd = dnnl::reorder::primitive_desc(CpuEngine::Get()->get_engine(),
                                                            in_dnnl_mem->get_desc(),
                                                            CpuEngine::Get()->get_engine(),
                                                            sum_mem_desc,
                                                            reorder_attr);
      DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(reorder_pd),
                                          {{DNNL_ARG_FROM, *in_dnnl_mem}, {DNNL_ARG_TO, *tmp_mem}});
      return NDArray(tmp_mem);
    } else {
      dnnl_mem_ptr tmp_mem(new dnnl::memory(in_dnnl_mem->get_desc(),
                                            CpuEngine::Get()->get_engine(),
                                            out_dnnl_mem->get_data_handle()));
      DNNLStream::Get()->RegisterMem(tmp_mem);
      DNNLMemoryCopy(*in_dnnl_mem, tmp_mem.get());
      return NDArray(tmp_mem);
    }
  }
}

bool SgDNNLFCOp::CheckInitializationConditions(const std::vector<NDArray>& inputs,
                                               const std::vector<float>& min_max_vec,
                                               bool is_channel_wise) {
  if (initialized_ && dmlc::GetEnv("MXNET_ONEDNN_QFC_DYNAMIC_PARAMS", 0)) {
    bool has_bias = !full_param_.default_param.no_bias;
    if (cached_data_min_ != min_max_vec[kDataMin] || cached_data_max_ != min_max_vec[kDataMax] ||
        cached_sum_min_ != min_max_vec[kSumMin] || cached_sum_max_ != min_max_vec[kSumMax]) {
      return false;
    }

    if (is_channel_wise) {
      if (weight_ver_ != inputs[fullc::kWeight].version() ||
          (has_bias && (bias_ver_ != inputs[fullc::kBias].version()))) {
        return false;
      }
    } else {
      if (cached_weight_min_ != min_max_vec[kWeightMin] ||
          cached_weight_max_ != min_max_vec[kWeightMax] ||
          (has_bias && (cached_bias_min_ != min_max_vec[kBiasMin] ||
                        cached_bias_max_ != min_max_vec[kBiasMax]))) {
        return false;
      }
    }
    return true;
  }
  return false;
}

dnnl::memory::desc SgDNNLFCOp::CreateOutputMemoryDesc(const mxnet::TShape& oshape, int out_dtype) {
  auto default_param = full_param_.default_param;
  dnnl::memory::dims out_dims(2);
  if (oshape.ndim() == 2) {
    out_dims[0] = static_cast<index_t>(oshape[0]);
    out_dims[1] = static_cast<index_t>(oshape[1]);
  } else {
    if (!default_param.flatten) {
      out_dims[0] = static_cast<index_t>(oshape.ProdShape(0, oshape.ndim() - 1));
      out_dims[1] = static_cast<index_t>(oshape[oshape.ndim() - 1]);
    } else {
      out_dims[0] = static_cast<index_t>(oshape[0]);
      out_dims[1] = static_cast<index_t>(oshape.ProdShape(1, oshape.ndim()));
    }
  }
  dnnl::memory::desc out_md =
      dnnl::memory::desc(out_dims,
                         get_dnnl_type(out_dtype),
                         static_cast<dnnl::memory::format_tag>(GetDefaultFormat(2)));
  return out_md;
}

bool SgDNNLFCOp::PrepareQuantization(const OpContext& ctx,
                                     const std::vector<NDArray>& in_data,
                                     const NDArray& output,
                                     const std::vector<float>& min_max_vec) {
  const auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  const FCInputIndex idx(full_param_);
  bool support_channelwise_scale = false;
  auto dnnl_param                = full_param_.dnnl_param;
  bool has_bias                  = !full_param_.default_param.no_bias;
  const NDArray& data            = in_data[fullc::kData];
  const NDArray& weight          = in_data[fullc::kWeight];
  const bool channel_wise =
      dnnl_param.channel_wise_quantize.has_value() && dnnl_param.channel_wise_quantize.value();

  CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
  data_scale_ = GetQuantizeScale(data.dtype(), cached_data_min_, cached_data_max_);

  bool fuse_requantize = false;
  // Channelwise scaling is only supported when fusion is enabled (requantize or dequantize).
  if (dnnl_param.min_calib_range.has_value() && dnnl_param.max_calib_range.has_value()) {
    cached_output_min_        = dnnl_param.min_calib_range.value();
    cached_output_max_        = dnnl_param.max_calib_range.value();
    support_channelwise_scale = true;
    fuse_requantize           = true;
  }
  if (dnnl_param.enabled_float_output.has_value()) {
    support_channelwise_scale = true;
  }
  // channel_wise  support_channelwise_scale  result
  // True          True                       True
  // True          False                      Error
  // False         True/False                 False
  if (channel_wise && !support_channelwise_scale) {
    LOG(FATAL) << "Currently, channel-wise quantization requires fuse requantize or dequantize."
               << " Please make sure the `min_calib_range` and `max_calib_range` are set when only"
               << " fuse requantize (outputs of FullyConnected are collected during calibration "
                  "phase),"
               << " or the env var of `MXNET_DISABLE_ONEDNN_QFC_FLOAT_OUTPUT` and "
               << " `MXNET_DISABLE_ONEDNN_QFC_FUSE_ALL` are not set to true (default is false)";
  }
  support_channelwise_scale = support_channelwise_scale && channel_wise;

  if (support_channelwise_scale) {
    MSHADOW_REAL_TYPE_SWITCH(cached_weight_.dtype(), DType, {
      weight_scales_ = GetWeightScales<DType>(cached_weight_,
                                              has_bias ? &cached_bias_ : nullptr,
                                              data_scale_,
                                              support_channelwise_scale);
    });
  } else {
    weight_scales_.resize(1);
    weight_scales_[0] =
        GetQuantizeScale(cached_weight_.dtype(), cached_weight_min_, cached_weight_max_);
    if (has_bias) {
      if (cached_bias_.dtype() == mshadow::kInt8) {
        float bias_scale = GetQuantizeScale(mshadow::kInt8, cached_bias_min_, cached_bias_max_);

        float bias_int32_rescale = data_scale_ * weight_scales_[0] / bias_scale;
        // TODO(zhennan): dnnl has bug to handle INT_MAX in bias, so set
        // the maximum value of bias to INT_MAX / 2.
        float bias_max_rescale =
            MaxValue<int32_t>() / 2 / MaxAbs(cached_bias_min_, cached_bias_max_) / bias_scale;
        if (bias_int32_rescale > bias_max_rescale) {
          // avoid overflow on bias
          bias_int32_rescale   = bias_max_rescale;
          float weight_rescale = bias_int32_rescale * bias_scale / data_scale_ / weight_scales_[0];
          int8_t* weight_ptr   = weight.data().dptr<int8_t>();
          size_t weight_size   = weight.shape().Size();
#pragma omp parallel for num_threads(nthreads)
          for (index_t i = 0; i < static_cast<index_t>(weight_size); ++i) {
            weight_ptr[i] = std::round(weight_ptr[i] * weight_rescale);
          }
          weight_scales_[0] *= weight_rescale;
        }
        NDArray bias = in_data[fullc::kBias];
        cached_bias_ =
            NDArray(bias.storage_type(), bias.shape(), bias.ctx(), true, mshadow::kInt32);
        int8_t* bias_ptr            = bias.data().dptr<int8_t>();
        int32_t* quantized_bias_ptr = cached_bias_.data().dptr<int32_t>();
        size_t bias_size            = bias.shape().Size();

#pragma omp parallel for num_threads(nthreads)
        for (index_t i = 0; i < static_cast<index_t>(bias_size); ++i) {
          quantized_bias_ptr[i] = std::round(bias_ptr[i] * bias_int32_rescale);
        }
      }
    }
  }

  size_t num_channel = cached_weight_.shape()[0];
  float out_scale    = 1.0f;
  if (fuse_requantize || dnnl_param.enabled_float_output.has_value()) {
    float tmp_scale_ = 1.0f;
    if (fuse_requantize) {
      if (dnnl_param.with_eltwise) {
        tmp_scale_ = 1.0 / data_scale_;
        full_param_.eltwise_param.scale =
            GetQuantizeScale(output.dtype(), cached_output_min_, cached_output_max_);
      } else {
        out_scale  = GetQuantizeScale(output.dtype(), cached_output_min_, cached_output_max_);
        tmp_scale_ = out_scale / data_scale_;
      }
    } else {
      tmp_scale_ = 1.0 / data_scale_;
    }

    if (support_channelwise_scale) {
      full_param_.output_scales.resize(num_channel);
#pragma omp parallel for num_threads(nthreads)
      for (index_t i = 0; i < static_cast<index_t>(num_channel); ++i) {
        full_param_.output_scales[i] = tmp_scale_ / weight_scales_[i];
      }
    } else {
      full_param_.output_scales.resize(1);
      full_param_.output_scales[0] = tmp_scale_ / weight_scales_[0];
    }
  } else {
    Stream<cpu>* s = ctx.get_stream<cpu>();
    if (data.dtype() == mshadow::kInt8) {
      mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
          s,
          1,
          &cached_output_min_,
          &cached_output_max_,
          &(min_max_vec[kDataMin]),
          &(min_max_vec[kDataMax]),
          &(min_max_vec[kWeightMin]),
          &(min_max_vec[kWeightMax]));
    } else {
      mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
          s,
          1,
          &cached_output_min_,
          &cached_output_max_,
          &(min_max_vec[kDataMin]),
          &(min_max_vec[kDataMax]),
          &(min_max_vec[kWeightMin]),
          &(min_max_vec[kWeightMax]));
    }
    full_param_.output_scales.resize(0);
    out_scale = data_scale_ * weight_scales_[0];
  }

  if (dnnl_param.with_sum && !dnnl_param.enabled_float_output.has_value()) {
    float sum_in_scale =
        GetQuantizeScale(in_data[idx.sum].dtype(), cached_sum_min_, cached_sum_max_);
    full_param_.sum_scale = out_scale / sum_in_scale;
    if (in_data[idx.sum].dtype() == mshadow::kUint8 && output.dtype() == mshadow::kInt8) {
      // In this case, reorder with scale 0.5 is used on in_data[idx.sum] to
      // scale it to s8 range, so sum_scale has to be rescaled as well
      full_param_.sum_scale *= 1.0 / u8_to_s8_scale;
    }
  }
  return support_channelwise_scale;
}

void SgDNNLFCOp::GetCachedWeightsAndBias(const NDArray& weight,
                                         bool support_channelwise_scale,
                                         bool has_bias) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLFullyconSignature, std::pair<NDArray, NDArray>, OpHash>
      fcWeightsAndBias;
#else
  static MX_THREAD_LOCAL
      std::unordered_map<DNNLFullyconSignature, std::pair<NDArray, NDArray>, OpHash>
          fcWeightsAndBias;
#endif
  static const bool use_cache = !(dmlc::GetEnv("MXNET_ONEDNN_DISABLE_FC_CACHE", 0));
  const bool has_id           = attrs.dict.count("__identifier__");
  bool read_from_cache        = false;

  DNNLFullyconSignature key(full_param_);
  if (use_cache && has_id) {
    key.AddSign(fwd_->fwd_pd.weights_desc());
    key.AddSign(attrs.dict["__identifier__"]);
    key.AddSign(attrs.name);

    auto it = fcWeightsAndBias.find(key);
    if (it != fcWeightsAndBias.end()) {
      cached_weight_  = it->second.first;
      cached_bias_    = it->second.second;
      read_from_cache = true;
      common::LogOnce(
          "oneDNN optimized version of FullyConnected for inference is being used. Weights and "
          "bias are cached and can not be dynamically changed during runtime. To disable caching "
          "mechanism use MXNET_ONEDNN_DISABLE_FC_CACHE=1.");
    }
  }

  if (!read_from_cache) {
    // convert weight and bias to the format that oneDNN requires
    if (!full_param_.dnnl_param.quantized || support_channelwise_scale) {
      dnnl::memory::desc bias_md;
      if (has_bias)
        bias_md = fwd_->fwd_pd.bias_desc();
      ConvertWeightBias2DNNL(&cached_weight_,
                             &cached_bias_,
                             has_bias,
                             fwd_->fwd_pd.weights_desc(),
                             has_bias ? &bias_md : nullptr,
                             1,
                             data_scale_,
                             weight_scales_,
                             false);
    } else {
      const auto def_weight_mem = weight.GetDNNLData();
      if (def_weight_mem->get_desc() != fwd_->fwd_pd.weights_desc()) {
        auto weight_desc       = fwd_->fwd_pd.weights_desc();
        cached_weight_         = NDArray(&weight_desc);
        auto cached_weight_mem = cached_weight_.GetDNNLData();
        std::unordered_map<int, dnnl::memory> args(
            {{DNNL_ARG_FROM, *def_weight_mem}, {DNNL_ARG_TO, *cached_weight_mem}});
        DNNLStream::Get()->RegisterPrimArgs(dnnl::reorder(*def_weight_mem, *cached_weight_mem),
                                            args);
      }
    }
    if (use_cache && has_id)
      AddToCache(&fcWeightsAndBias, key, {cached_weight_, cached_bias_});
  }
}

static void SgDNNLFCParamParser(nnvm::NodeAttrs* attrs) {
  // For backward compatible, with_relu->with_eltwise
  auto legacy = attrs->dict.find("with_relu");
  if (legacy != attrs->dict.end()) {
    attrs->dict["with_eltwise"] = attrs->dict["with_relu"];
    attrs->dict.erase(legacy);
  }

  DNNLFCFullParam full_param;
  try {
    full_param.dnnl_param.Init(attrs->dict);
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
  auto subgraph_sym = attrs->subgraphs[0];
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::ObjectPtr& node) {
    if (node->is_variable())
      return;
    auto& op_name = node->op()->name;
    if (op_name == "FullyConnected") {
      full_param.default_param = nnvm::get<FullyConnectedParam>(node->attrs.parsed);
    } else if (SupportDNNLFCEltwiseFusion(op_name)) {
      if (op_name == "Activation") {
        const ActivationParam act_param = nnvm::get<ActivationParam>(node->attrs.parsed);
        full_param.eltwise_param.alg    = GetDNNLActAlgo(act_param);
      } else if (op_name == "LeakyReLU") {
        const auto act_param           = nnvm::get<LeakyReLUParam>(node->attrs.parsed);
        full_param.eltwise_param.alpha = act_param.slope;
        full_param.eltwise_param.alg   = GetDNNLActAlgo(act_param);
      } else if (op_name == "clip") {
        const ClipParam clip_param     = nnvm::get<ClipParam>(node->attrs.parsed);
        full_param.eltwise_param.alg   = dnnl::algorithm::eltwise_bounded_relu;
        full_param.eltwise_param.alpha = clip_param.a_max;
      } else {
        full_param.eltwise_param.alg = GetDNNLEltwiseAlgo(op_name);
      }
    }
  });
  attrs->parsed = std::move(full_param);
}

static std::vector<std::string> SgDNNLFCListInputNames(const NodeAttrs& attrs) {
  auto const& full_param               = nnvm::get<DNNLFCFullParam>(attrs.parsed);
  auto const& dnnl_param               = full_param.dnnl_param;
  std::vector<std::string> input_names = DefaultSubgraphOpListInputs(attrs);
  if (dnnl_param.quantized) {
    const bool channel_wise =
        dnnl_param.channel_wise_quantize.has_value() && dnnl_param.channel_wise_quantize;
    input_names.emplace_back("data_min");
    input_names.emplace_back("data_max");
    if (!channel_wise) {
      input_names.emplace_back("weight_min");
      input_names.emplace_back("weight_max");
      if (!full_param.default_param.no_bias) {
        input_names.emplace_back("bias_min");
        input_names.emplace_back("bias_max");
      }
    }
    if (dnnl_param.with_sum && !dnnl_param.enabled_float_output.has_value()) {
      input_names.emplace_back("sum_min");
      input_names.emplace_back("sum_max");
    }
  }
  return input_names;
}

static std::vector<std::string> SgDNNLFCListOutputNames(const NodeAttrs& attrs) {
  auto const& full_param = nnvm::get<DNNLFCFullParam>(attrs.parsed);
  if (full_param.dnnl_param.quantized) {
    if (full_param.dnnl_param.enabled_float_output.has_value())
      return std::vector<std::string>{"output"};
    else
      return std::vector<std::string>{"output", "output_min", "output_max"};
  } else {
    return std::vector<std::string>{"output"};
  }
}

template <typename T>
static inline void FillBaseInputOutputInfo(const DNNLFCFullParam& param,
                                           std::vector<T>* base_in_attrs,
                                           std::vector<T>* base_out_attrs,
                                           std::vector<T>* in_attrs,
                                           std::vector<T>* out_attrs) {
  auto base_num_inputs = FCInputIndex(param).GetBase();

  base_out_attrs->push_back(out_attrs->at(0));
  for (int i = 0; i < base_num_inputs; ++i) {
    base_in_attrs->push_back(in_attrs->at(i));
  }
}

static bool SgDNNLFCInferShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector* in_shapes,
                               mxnet::ShapeVector* out_shapes) {
  auto const& full_param = nnvm::get<DNNLFCFullParam>(attrs.parsed);
  if (full_param.dnnl_param.quantized) {
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes;
    FillBaseInputOutputInfo(full_param, &base_in_shapes, &base_out_shapes, in_shapes, out_shapes);
    bool ret = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);

    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (i < base_in_shapes.size())
        in_shapes->at(i) = base_in_shapes[i];
      else
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
    }

    out_shapes->at(0) = base_out_shapes[0];
    if (!full_param.dnnl_param.enabled_float_output.has_value()) {
      SHAPE_ASSIGN_CHECK(*out_shapes, 1, Shape1(1));
      SHAPE_ASSIGN_CHECK(*out_shapes, 2, Shape1(1));
    }
    return ret;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgDNNLFCInferType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_types,
                              std::vector<int>* out_types) {
  auto const& full_param = nnvm::get<DNNLFCFullParam>(attrs.parsed);
  if (full_param.dnnl_param.quantized) {
    const bool channel_wise = full_param.dnnl_param.channel_wise_quantize.has_value() &&
                              full_param.dnnl_param.channel_wise_quantize;
    const FCInputIndex idx(full_param);

    if (in_types->at(idx.data) == mshadow::kBfloat16) {
      return false;
    }

    CHECK(in_types->at(idx.data) == mshadow::kInt8 || in_types->at(idx.data) == mshadow::kUint8)
        << "QuantizedFullyConnected  data input only supports int8/uint8, while "
        << in_types->at(idx.data) << " is given.";
    if (channel_wise) {
      TYPE_ASSIGN_CHECK(*in_types, idx.weight, mshadow::kFloat32);
      if (idx.IsBiasExist()) {
        TYPE_ASSIGN_CHECK(*in_types, idx.bias, mshadow::kFloat32);
      }
    } else {
      TYPE_ASSIGN_CHECK(*in_types, idx.weight, mshadow::kInt8);
      if (idx.IsBiasExist()) {
        if (in_types->at(idx.bias) == -1) {
          TYPE_ASSIGN_CHECK(*in_types, idx.bias, mshadow::kInt32);
        } else {
          CHECK(in_types->at(idx.bias) == mshadow::kInt8 ||
                in_types->at(idx.bias) == mshadow::kInt32)
              << "QuantizedFullyConnected bias input only supports int8/int32, while "
              << in_types->at(idx.bias) << " is given.";
        }
      }
    }
    if (idx.IsSumExist()) {
      if (full_param.dnnl_param.enabled_float_output.has_value()) {
        TYPE_ASSIGN_CHECK(*in_types, idx.sum, full_param.dnnl_param.enabled_float_output.value());
      } else {
        CHECK(in_types->at(idx.sum) == mshadow::kInt8 || in_types->at(idx.sum) == mshadow::kUint8)
            << "QuantizedFullyConnected sum input only supports int8/uint8, while "
            << in_types->at(idx.sum) << " is given.";
      }
    }
    for (size_t i = idx.data_min; i < in_types->size(); ++i) {
      TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
    }

    if (full_param.dnnl_param.enabled_float_output.has_value()) {
      TYPE_ASSIGN_CHECK(*out_types, 0, full_param.dnnl_param.enabled_float_output.value());
    } else {
      if (full_param.dnnl_param.min_calib_range.has_value() &&
          full_param.dnnl_param.max_calib_range.has_value()) {
        if (IsOutputUint8(full_param) &&
            (!idx.IsSumExist() || in_types->at(idx.sum) == mshadow::kUint8)) {
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
    return true;
  } else {
    bool result = DefaultSubgraphOpType(attrs, in_types, out_types);
    if (full_param.dnnl_param.enabled_float_output.has_value()) {
      (*out_types)[0] = full_param.dnnl_param.enabled_float_output.value();
    }
    return result;
  }
}

static bool SgDNNLFCStorageType(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int>* in_attrs,
                                std::vector<int>* out_attrs) {
  auto const& full_param = nnvm::get<DNNLFCFullParam>(attrs.parsed);
  if (full_param.dnnl_param.quantized) {
    std::vector<int> base_in_attrs;
    std::vector<int> base_out_attrs;
    FillBaseInputOutputInfo(full_param, &base_in_attrs, &base_out_attrs, in_attrs, out_attrs);
    bool ret = DefaultSubgraphOpStorageType(
        attrs, dev_mask, dispatch_mode, &base_in_attrs, &base_out_attrs);

    for (size_t i = 0; i < in_attrs->size(); ++i) {
      if (i < base_in_attrs.size())
        in_attrs->at(i) = base_in_attrs[i];
      else
        type_assign(&in_attrs->at(i), mxnet::kDefaultStorage);
    }

    out_attrs->at(0) = base_out_attrs[0];
    if (!full_param.dnnl_param.enabled_float_output.has_value()) {
      type_assign(&out_attrs->at(1), mxnet::kDefaultStorage);
      type_assign(&out_attrs->at(2), mxnet::kDefaultStorage);
    }
    return ret;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
  }
}

std::vector<std::pair<int, int>> SgDNNLFCInplaceOption(const NodeAttrs& attrs) {
  auto const& param = nnvm::get<DNNLFCFullParam>(attrs.parsed);
  if (param.dnnl_param.with_sum) {
    return std::vector<std::pair<int, int>>{{FCInputIndex(param).sum, 0}};
  } else {
    return std::vector<std::pair<int, int>>();
  }
}

static OpStatePtr CreateSgDNNLFCState(const nnvm::NodeAttrs& attrs,
                                      Context ctx,
                                      const mxnet::ShapeVector& in_shapes,
                                      const std::vector<int>& in_types) {
  return OpStatePtr::Create<SgDNNLFCOp>(attrs);
}

static void SgDNNLFCForward(const OpStatePtr& state_pointer,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
  SgDNNLFCOp& op = state_pointer.get_state<SgDNNLFCOp>();
  op.Forward(ctx, inputs, req, outputs);
}

nnvm::ObjectPtr SgDNNLFCQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node          = nnvm::Node::Create();
  node->attrs.op                = Op::Get("_sg_onednn_fully_connected");
  node->attrs.name              = "quantized_" + attrs.name;
  node->attrs.dict              = attrs.dict;
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

static bool SgDNNLAvoidFCQuantizeInput(const NodeAttrs& attrs,
                                       const size_t index_to_check,
                                       const std::string quantize_granularity) {
  auto const& full_param = nnvm::get<DNNLFCFullParam>(attrs.parsed);
  std::unordered_set<size_t> avoid_indexes;
  FCInputIndex idx(full_param);
  if (quantize_granularity == "channel-wise") {
    avoid_indexes.insert(fullc::kWeight);  // weight
    if (!full_param.default_param.no_bias) {
      avoid_indexes.insert(fullc::kBias);  // bias
    }
  }
  if (idx.IsSumInputFloat()) {
    avoid_indexes.insert(idx.sum);
  }
  return avoid_indexes.count(index_to_check);
}

NNVM_REGISTER_OP(_sg_onednn_fully_connected)
    .add_alias("_sg_mkldnn_fully_connected")
    .describe(R"code(_sg_onednn_fully_connected)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      auto const& full_param = nnvm::get<DNNLFCFullParam>(attrs.parsed);
      return FCInputIndex(full_param).GetTotal();
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      auto const& full_param = nnvm::get<DNNLFCFullParam>(attrs.parsed);
      if (full_param.dnnl_param.quantized &&
          !full_param.dnnl_param.enabled_float_output.has_value()) {
        return 3;
      }
      return 1;
    })
    .set_attr_parser(SgDNNLFCParamParser)
    .set_attr<nnvm::FListInputNames>("FListInputNames", SgDNNLFCListInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames", SgDNNLFCListOutputNames)
    .set_attr<mxnet::FInferShape>("FInferShape", SgDNNLFCInferShape)
    .set_attr<nnvm::FInferType>("FInferType", SgDNNLFCInferType)
    .set_attr<FInferStorageType>("FInferStorageType", SgDNNLFCStorageType)
    .set_attr<FCreateOpState>("FCreateOpState", CreateSgDNNLFCState)
    .set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgDNNLFCForward)
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
    .set_attr<nnvm::FInplaceOption>("FInplaceOption", SgDNNLFCInplaceOption)
    .set_attr<FQuantizable>("FQuantizable",
                            [](const NodeAttrs& attrs) { return QuantizeType::kMust; })
    .set_attr<FQuantizedOp>("FQuantizedOp", SgDNNLFCQuantizedOp)
    .set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
    .set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", SgDNNLAvoidFCQuantizeInput);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
