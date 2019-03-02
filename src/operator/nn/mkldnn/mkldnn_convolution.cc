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
 * \file mkldnn_convolution.cc
 * \brief
 * \author Da Zheng
*/


#if MXNET_USE_MKLDNN == 1

#include "../convolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_convolution-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MKLDNNConvParam);

bool SupportMKLDNNConv(const ConvolutionParam& params, const NDArray &input) {
  if ((params.kernel.ndim() != 1) &&
      (params.kernel.ndim() != 2))
    return false;
  return SupportMKLDNNQuantize(input.dtype()) &&
         ((input.shape().ndim() == 3) ||
          (input.shape().ndim() == 4));
}

mkldnn::convolution_forward::primitive_desc GetConvFwdImpl(const MKLDNNConvFullParam &param,
                                                           const bool is_train, const NDArray &data,
                                                           const NDArray &weights,
                                                           const NDArray *bias,
                                                           const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.conv_param.num_group, param.mkldnn_param.quantized);
  auto out_md = GetMemDesc(output);
  auto bias_md =
      bias ? (param.mkldnn_param.quantized ? GetMemDesc(*bias, mshadow::kInt32) : GetMemDesc(*bias))
           : mkldnn::memory::desc{
             {}, mkldnn::memory::data_type::data_undef, mkldnn::memory::format::any};
  auto bias_md_ptr = bias ? &bias_md : nullptr;

  mkldnn::memory::dims strides(param.conv_param.kernel.ndim());
  mkldnn::memory::dims padding(param.conv_param.kernel.ndim());
  if (param.conv_param.kernel.ndim() == 1) {
    CHECK_GE(param.conv_param.stride.ndim(), 1U);
    CHECK_GE(param.conv_param.pad.ndim(), 1U);
    CHECK_GE(param.conv_param.dilate.ndim(), 1U);
    strides[0] = param.conv_param.stride[0];
    padding[0] = param.conv_param.pad[0];
  } else if (param.conv_param.kernel.ndim() == 2) {
    CHECK_GE(param.conv_param.stride.ndim(), 2U);
    CHECK_GE(param.conv_param.pad.ndim(), 2U);
    CHECK_GE(param.conv_param.dilate.ndim(), 2U);
    strides[0] = param.conv_param.stride[0];
    strides[1] = param.conv_param.stride[1];
    padding[0] = param.conv_param.pad[0];
    padding[1] = param.conv_param.pad[1];
  } else {
    LOG(FATAL) << "Unexpected MKL-DNN Conv kernel size "
               << param.conv_param.kernel.ndim() << ", supporting only 1 or 2.";
  }
  mkldnn::primitive_attr attr;
  mkldnn::post_ops ops;
  if (param.mkldnn_param.with_relu) {
    float scale = 1.0f;  // for fp32, scale is 1.
    float alpha = 0.0f;  // negative slope for mkldnn_eltwise_relu.
    float beta = 1.0f;   // ignored for mkldnn_eltwise_relu.
    ops.append_eltwise(scale, eltwise_relu, alpha, beta);
  }
  if (param.mkldnn_param.with_sum) {
    ops.append_sum(param.sum_scale);
  }
  if (param.mkldnn_param.with_postsum_relu) {
    float scale = 1.0f;  // for fp32, scale is 1.
    float alpha = 0.0f;  // negative slope for mkldnn_eltwise_relu.
    float beta = 1.0f;   // ignored for mkldnn_eltwise_relu.
    ops.append_eltwise(scale, eltwise_relu, alpha, beta);
  }
  attr.set_post_ops(ops);

  if (param.mkldnn_param.quantized && param.requantize_scales.size()) {
    int mask = (param.requantize_scales.size() > 1) ? 2 : 0;
    attr.set_output_scales(mask, param.requantize_scales);
    attr.set_int_output_round_mode(round_nearest);
  }
  auto GetConvFwdPd = [&param, &data, &weights, &output,
                       &attr](const mkldnn::convolution_forward::desc &desc) {
    auto engine = CpuEngine::Get()->get_engine();
    try {
      auto conv_pd = mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
      while (conv_pd.dst_primitive_desc().get_size() != GetArraySize(output) ||
             conv_pd.src_primitive_desc().get_size() != GetArraySize(data) ||
             (!param.mkldnn_param.quantized &&
              conv_pd.weights_primitive_desc().get_size() != GetArraySize(weights))) {
        // next_impl() will visit desc and engine, please make sure they are still alive here.
        CHECK(conv_pd.next_impl()) << "No convolution implementation for this request.";
      }
      return conv_pd;
    } catch (mkldnn::error &e) {
      if (e.status == mkldnn_unimplemented && param.mkldnn_param.quantized) {
        LOG(ERROR) << "AVX512-BW support or Intel(R) MKL dependency is "
                      "required for int8 convolution";
      } else {
        LOG(ERROR) << e.message;
      }
      throw;
    }
  };

  if (param.conv_param.dilate.ndim() == 0 && bias_md_ptr == nullptr) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct, data_md,
                                           weight_md, out_md, strides, padding, padding,
                                           mkldnn::padding_kind::zero);
    return GetConvFwdPd(desc);
  } else if (param.conv_param.dilate.ndim() == 0) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct, data_md,
                                           weight_md, *bias_md_ptr, out_md, strides, padding,
                                           padding, mkldnn::padding_kind::zero);
    return GetConvFwdPd(desc);
  } else {
    mkldnn::memory::dims dilates(param.conv_param.kernel.ndim());
    if (param.conv_param.dilate.ndim() == 1) {
      dilates[0] = param.conv_param.dilate[0] - 1;
    } else if (param.conv_param.dilate.ndim() == 2) {
      dilates[0] = param.conv_param.dilate[0] - 1;
      dilates[1] = param.conv_param.dilate[1] - 1;
    } else {
      LOG(FATAL) << "Unexpected MKL-DNN Conv dilate size " << param.conv_param.dilate.ndim()
                 << ", supporting only 1 or 2.";
    }
    if (bias_md_ptr == nullptr) {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct, data_md,
                                             weight_md, out_md, strides, dilates, padding, padding,
                                             mkldnn::padding_kind::zero);
      return GetConvFwdPd(desc);
    } else {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct, data_md,
                                             weight_md, *bias_md_ptr, out_md, strides, dilates,
                                             padding, padding, mkldnn::padding_kind::zero);
      return GetConvFwdPd(desc);
    }
  }
}

static mkldnn::convolution_backward_data::primitive_desc GetConvBwdData(
    const ConvolutionParam& param, const NDArray &data, const NDArray &weights,
    const NDArray &output, const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides(param.kernel.ndim());
  mkldnn::memory::dims padding(param.kernel.ndim());
  if (param.kernel.ndim() == 1) {
    CHECK_GE(param.stride.ndim(), 1U);
    CHECK_GE(param.pad.ndim(), 1U);
    CHECK_GE(param.dilate.ndim(), 1U);
    strides[0] = param.stride[0];
    padding[0] = param.pad[0];
  } else if (param.kernel.ndim() == 2) {
    CHECK_GE(param.stride.ndim(), 2U);
    CHECK_GE(param.pad.ndim(), 2U);
    CHECK_GE(param.dilate.ndim(), 2U);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else {
    LOG(FATAL) << "Unexpected MKL-DNN Conv kernel size " << param.kernel.ndim()
               << ", supporting only 1 or 2.";
  }

  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // for computation compared with the actual tensor size. Currently, MKL-DNN
  // operators are still reusing those memory from memory planning and the
  // memory size may smaller than what MKL-DNN kernels require. So here we need
  // select suboptimal kernel for computation according to tensor sizes.
  if (param.dilate.ndim() == 0) {
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    auto conv_pd = mkldnn::convolution_backward_data::primitive_desc(desc, engine, fwd_pd);
    while (conv_pd.diff_dst_primitive_desc().get_size() != GetArraySize(output) ||
           conv_pd.diff_src_primitive_desc().get_size() != GetArraySize(data) ||
           conv_pd.weights_primitive_desc().get_size() != GetArraySize(weights)) {
      CHECK(conv_pd.next_impl()) << "No implementation";
    }
    return conv_pd;
  } else {
    mkldnn::memory::dims dilates(param.kernel.ndim());
    if (param.dilate.ndim() == 1) {
      dilates[0] = param.dilate[0] - 1;
    } else if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    } else {
      LOG(FATAL) << "Unexpected MKL-DNN Conv dilate size "
                 << param.dilate.ndim() << ", supporting only 1 or 2.";
    }
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, dilates, padding, padding,
        mkldnn::padding_kind::zero);
    auto conv_pd = mkldnn::convolution_backward_data::primitive_desc(desc, engine, fwd_pd);
    while (conv_pd.diff_dst_primitive_desc().get_size() != GetArraySize(output) ||
           conv_pd.diff_src_primitive_desc().get_size() != GetArraySize(data) ||
           conv_pd.weights_primitive_desc().get_size() != GetArraySize(weights)) {
      CHECK(conv_pd.next_impl()) << "No implementation";
    }
    return conv_pd;
  }
}

static mkldnn::convolution_backward_weights::primitive_desc GetConvBwdWeights(
    const ConvolutionParam& param, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output,
    const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides(param.kernel.ndim());
  mkldnn::memory::dims padding(param.kernel.ndim());
  if (param.kernel.ndim() == 1) {
    CHECK_GE(param.stride.ndim(), 1U);
    CHECK_GE(param.pad.ndim(), 1U);
    CHECK_GE(param.dilate.ndim(), 1U);
    strides[0] = param.stride[0];
    padding[0] = param.pad[0];
  } else if (param.kernel.ndim() == 2) {
    CHECK_GE(param.stride.ndim(), 2U);
    CHECK_GE(param.pad.ndim(), 2U);
    CHECK_GE(param.dilate.ndim(), 2U);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else {
    LOG(FATAL) << "Unexpected MKL-DNN Conv kernel size " << param.kernel.ndim()
               << ", supporting only 1 or 2.";
  }

  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // for computation compared with the actual tensor size. Currently, MKL-DNN
  // operators are still reusing those memory from memory planning and the
  // memory size may smaller than what MKL-DNN kernels require. So here we need
  // select suboptimal kernel for computation according to tensor sizes.
  if (param.dilate.ndim() == 0 && bias == nullptr) {
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    auto conv_pd = mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    while (conv_pd.diff_dst_primitive_desc().get_size() != GetArraySize(output) ||
           conv_pd.src_primitive_desc().get_size() != GetArraySize(data) ||
           conv_pd.diff_weights_primitive_desc().get_size() != GetArraySize(weights)) {
      CHECK(conv_pd.next_impl()) << "No implementation";
    }
    return conv_pd;
  } else if (param.dilate.ndim() == 0) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, bias_md, out_md, strides, padding, padding,
        mkldnn::padding_kind::zero);
    auto conv_pd = mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    while (conv_pd.diff_dst_primitive_desc().get_size() != GetArraySize(output) ||
           conv_pd.src_primitive_desc().get_size() != GetArraySize(data) ||
           conv_pd.diff_weights_primitive_desc().get_size() != GetArraySize(weights)) {
      CHECK(conv_pd.next_impl()) << "No implementation";
    }
    return conv_pd;
  } else {
    mkldnn::memory::dims dilates(param.kernel.ndim());
    if (param.dilate.ndim() == 1) {
      dilates[0] = param.dilate[0] - 1;
    } else if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    } else {
      LOG(FATAL) << "Unexpected MKL-DNN Conv dilate size "
                 << param.dilate.ndim() << ", supporting only 1 or 2.";
    }
    if (bias == nullptr) {
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, dilates, padding, padding,
          mkldnn::padding_kind::zero);
      auto conv_pd = mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
      while (conv_pd.diff_dst_primitive_desc().get_size() != GetArraySize(output) ||
             conv_pd.src_primitive_desc().get_size() != GetArraySize(data) ||
             conv_pd.diff_weights_primitive_desc().get_size() != GetArraySize(weights)) {
        CHECK(conv_pd.next_impl()) << "No implementation";
      }
      return conv_pd;
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
                                                      data_md, weight_md, bias_md, out_md,
                                                      strides, dilates, padding, padding,
                                                      mkldnn::padding_kind::zero);
      auto conv_pd = mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
      while (conv_pd.diff_dst_primitive_desc().get_size() != GetArraySize(output) ||
             conv_pd.src_primitive_desc().get_size() != GetArraySize(data) ||
             conv_pd.diff_weights_primitive_desc().get_size() != GetArraySize(weights)) {
        CHECK(conv_pd.next_impl()) << "No implementation";
      }
      return conv_pd;
    }
  }
}

MKLDNNConvForward::MKLDNNConvForward(const MKLDNNConvFullParam &param, const bool is_train,
                                     const NDArray &data, const NDArray &weights,
                                     const NDArray *bias, const NDArray &output)
    : fwd_pd(GetConvFwdImpl(param, is_train, data, weights, bias, output)) {
  data_ = std::make_shared<mkldnn::memory>(fwd_pd.src_primitive_desc(), nullptr);
  weight_ = std::make_shared<mkldnn::memory>(fwd_pd.weights_primitive_desc(), nullptr);
  out_ = std::make_shared<mkldnn::memory>(fwd_pd.dst_primitive_desc(), nullptr);
  if (bias) {
    bias_ = std::make_shared<mkldnn::memory>(fwd_pd.bias_primitive_desc(), nullptr);
    fwd_ = std::make_shared<mkldnn::convolution_forward>(fwd_pd, *this->data_, *this->weight_,
                                                         *this->bias_, *this->out_);
  } else {
    fwd_ = std::make_shared<mkldnn::convolution_forward>(fwd_pd, *this->data_, *this->weight_,
                                                         *this->out_);
  }
}

void MKLDNNConvForward::SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                                  const mkldnn::memory *bias, const mkldnn::memory &output) {
  data_->set_data_handle(data.get_data_handle());
  weight_->set_data_handle(weight.get_data_handle());
  out_->set_data_handle(output.get_data_handle());
  if (bias != nullptr) bias_->set_data_handle(bias->get_data_handle());
}

MKLDNNConvForward &GetConvFwd(const ConvolutionParam &param,
                              const bool is_train, const NDArray &data,
                              const NDArray &weights, const NDArray *bias,
                              const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash> fwds;
#endif
  MKLDNNConvSignature key(param);
  key.AddSign(is_train);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConvFullParam full_param;
    full_param.conv_param = param;
    full_param.mkldnn_param.Init(std::unordered_map<std::string, std::string>());
    MKLDNNConvForward fwd(full_param, is_train, data, weights, bias, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNConvolutionForwardFullFeature(const MKLDNNConvFullParam &param,
                                         const OpContext &ctx,
                                         MKLDNNConvForward *fwd,
                                         const std::vector<NDArray> &in_data,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);

  auto data = in_data[conv::kData];
  if (data.IsView() && data.IsMKLDNNData())
    data = data.Reorder2Default();

  auto weight = in_data[conv::kWeight];
  if (weight.IsView() && weight.IsMKLDNNData())
    weight = weight.Reorder2Default();

  bool no_bias = param.conv_param.no_bias && !param.mkldnn_param.with_bn;

  auto data_mem = data.GetMKLDNNDataReorder(
      fwd->fwd_pd.src_primitive_desc());
  const mkldnn::memory *weight_mem;
  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (weight.IsMKLDNNData())
      // This asks the engine to change the layout of the weight array after
      // it's used.
      weight.Reorder2DefaultAsync();
    weight_mem = GetWeights(weight, fwd->fwd_pd.weights_primitive_desc(),
                            param.conv_param.num_group);
  } else {
    // For inference, we want to reorder the weight array so we don't need to
    // reorder data every time.
    if (weight.IsDefaultData()) {
      weight_mem = GetWeights(weight, fwd->fwd_pd.weights_primitive_desc(),
                              param.conv_param.num_group);
      // We also need to modify the layout on the original weight array. The
      // data conversion happens after the weight array is used.
      weight.MKLDNNDataReorderAsync(fwd->fwd_pd.weights_primitive_desc());
    } else {
      weight_mem = weight.GetMKLDNNData();
      CHECK(weight_mem->get_primitive_desc() == fwd->fwd_pd.weights_primitive_desc());
    }
  }
  mkldnn_output_t out_mem;
  if (param.mkldnn_param.with_sum) {
    out_mem = mkldnn_output_t(
        OutDataOp::Noop,
        const_cast<mkldnn::memory *>(out_data[conv::kOut].GetMKLDNNData()));
  } else {
    out_mem = CreateMKLDNNMem(out_data[conv::kOut],
                              fwd->fwd_pd.dst_primitive_desc(), req[conv::kOut]);
  }

  const mkldnn::memory *bias_mem = nullptr;
  if (!no_bias) {
    bias_mem = in_data[conv::kBias].GetMKLDNNData();
  }
  fwd->SetNewMem(*data_mem, *weight_mem, bias_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd->GetFwd());

  CommitOutput(out_data[conv::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}

void MKLDNNConvolutionForward(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<NDArray> &in_data,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &out_data) {
  MKLDNNConvFullParam param;
  param.conv_param = nnvm::get<ConvolutionParam>(attrs.parsed);
  param.mkldnn_param.Init(std::unordered_map<std::string, std::string>());
  auto &fwd = GetConvFwd(
      param.conv_param, ctx.is_train, in_data[conv::kData], in_data[conv::kWeight],
      param.conv_param.no_bias ? nullptr : &in_data[conv::kBias],
      out_data[conv::kOut]);
  MKLDNNConvolutionForwardFullFeature(param, ctx, &fwd, in_data, req, out_data);
}

class MKLDNNConvBackward {
  std::shared_ptr<mkldnn::convolution_backward_data> bwd_data;
  std::shared_ptr<mkldnn::convolution_backward_weights> bwd_weight;
  // conv::kData
  std::shared_ptr<mkldnn::memory> out_grad;
  std::shared_ptr<mkldnn::memory> in_grad;
  std::shared_ptr<mkldnn::memory> weight;
  // conv::kWeight
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> output;
  std::shared_ptr<mkldnn::memory> in_grad_weight;
  std::shared_ptr<mkldnn::memory> in_grad_bias;

 public:
  mkldnn::convolution_backward_data::primitive_desc bwdData_pd;
  mkldnn::convolution_backward_weights::primitive_desc bwdWeights_pd;

  MKLDNNConvBackward(
      const ConvolutionParam &param, const NDArray &data,
      const NDArray &weights, const NDArray *bias, const NDArray &output,
      const mkldnn::convolution_forward::primitive_desc &fwd_pd):
      bwdData_pd(GetConvBwdData(param, data, weights, output, fwd_pd)),
      bwdWeights_pd(GetConvBwdWeights(param, data, weights, bias, output, fwd_pd)) {
  }

  void SetDataNewMem(const mkldnn::memory &out_grad, const mkldnn::memory &weight,
                     const mkldnn::memory &in_grad) {
    if (this->out_grad == nullptr)
      this->out_grad = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
        bwdData_pd.diff_dst_primitive_desc(), out_grad.get_data_handle()));
    else
      this->out_grad->set_data_handle(out_grad.get_data_handle());
    if (this->in_grad == nullptr)
      this->in_grad = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
        bwdData_pd.diff_src_primitive_desc(), in_grad.get_data_handle()));
    else
      this->in_grad->set_data_handle(in_grad.get_data_handle());
    if (this->weight == nullptr)
      this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
         bwdData_pd.weights_primitive_desc(), weight.get_data_handle()));
    else
      this->weight->set_data_handle(weight.get_data_handle());
    if (this->bwd_data == nullptr)
      this->bwd_data = std::shared_ptr<mkldnn::convolution_backward_data>(
        new mkldnn::convolution_backward_data(
          this->bwdData_pd, mkldnn::primitive::at(*this->out_grad),
          mkldnn::primitive::at(*this->weight), *this->in_grad));
  }

void SetWeightNewMem(const mkldnn::memory &data,
                     const mkldnn::memory &out_grad,
                     const mkldnn::memory &in_grad_weight) {
    if (this->data == nullptr)
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          bwdWeights_pd.src_primitive_desc(), data.get_data_handle()));
    else
      this->data->set_data_handle(data.get_data_handle());
    if (this->output == nullptr)
      this->output = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          bwdWeights_pd.diff_dst_primitive_desc(), out_grad.get_data_handle()));
    else
      this->output->set_data_handle(out_grad.get_data_handle());
    if (this->in_grad_weight == nullptr)
      this->in_grad_weight = std::shared_ptr<mkldnn::memory>(
          new mkldnn::memory(bwdWeights_pd.diff_weights_primitive_desc(),
                             in_grad_weight.get_data_handle()));
    else
      this->in_grad_weight->set_data_handle(in_grad_weight.get_data_handle());

    if (this->bwd_weight == nullptr)
      this->bwd_weight = std::shared_ptr<mkldnn::convolution_backward_weights>(
          new mkldnn::convolution_backward_weights(
              this->bwdWeights_pd, mkldnn::primitive::at(*this->data),
              mkldnn::primitive::at(*this->output), *this->in_grad_weight));
  }

  void SetWeightNewMem(const mkldnn::memory &data,
                       const mkldnn::memory &out_grad,
                       const mkldnn::memory &in_grad_weight,
                       const mkldnn::memory &in_grad_bias) {
    if (this->data == nullptr)
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          bwdWeights_pd.src_primitive_desc(), data.get_data_handle()));
    else
      this->data->set_data_handle(data.get_data_handle());
    if (this->output == nullptr)
      this->output = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          bwdWeights_pd.diff_dst_primitive_desc(), out_grad.get_data_handle()));
    else
      this->output->set_data_handle(out_grad.get_data_handle());
    if (this->in_grad_weight == nullptr)
      this->in_grad_weight = std::shared_ptr<mkldnn::memory>(
          new mkldnn::memory(bwdWeights_pd.diff_weights_primitive_desc(),
                             in_grad_weight.get_data_handle()));
    else
      this->in_grad_weight->set_data_handle(in_grad_weight.get_data_handle());

    if (this->in_grad_bias == nullptr)
      this->in_grad_bias = std::shared_ptr<mkldnn::memory>(
          new mkldnn::memory(bwdWeights_pd.diff_bias_primitive_desc(),
                             in_grad_bias.get_data_handle()));
    else
      this->in_grad_bias->set_data_handle(in_grad_bias.get_data_handle());
    if (this->bwd_weight == nullptr)
      this->bwd_weight = std::shared_ptr<mkldnn::convolution_backward_weights>(
          new mkldnn::convolution_backward_weights(
              this->bwdWeights_pd, mkldnn::primitive::at(*this->data),
              mkldnn::primitive::at(*this->output), *this->in_grad_weight,
              *this->in_grad_bias));
  }

  const mkldnn::convolution_backward_data &GetBwdData() const {
    return *bwd_data;
  }

  const mkldnn::convolution_backward_weights &GetBwdWeights() const {
    return *bwd_weight;
  }
};

static inline MKLDNNConvBackward &GetConvBwd(
    const nnvm::NodeAttrs &attrs, const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output,
    const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNConvSignature, MKLDNNConvBackward, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNConvSignature, MKLDNNConvBackward, OpHash> bwds;
#endif
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  MKLDNNConvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);


  auto it = bwds.find(key);
  if (it == bwds.end()) {
    MKLDNNConvBackward bwd(param, data, weights, bias, output, fwd_pd);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNConvolutionBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  MKLDNNConvFullParam full_param;
  full_param.conv_param = nnvm::get<ConvolutionParam>(attrs.parsed);
  full_param.mkldnn_param.Init(std::unordered_map<std::string, std::string>());

  auto data = inputs[conv::kData + 1];
  if (data.IsView() && data.IsMKLDNNData())
    data = data.Reorder2Default();

  auto weight = inputs[conv::kWeight + 1];
  if (weight.IsView() && weight.IsMKLDNNData())
    weight = weight.Reorder2Default();

  const NDArray* bias = full_param.conv_param.no_bias ? nullptr : &inputs[conv::kBias + 1];

  auto out_grad = inputs[conv::kOut];
  if (out_grad.IsView() && out_grad.IsMKLDNNData())
    out_grad = out_grad.Reorder2Default();

  mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwdImpl(
      full_param, ctx.is_train, data, weight, bias, out_grad);
  const ConvolutionParam &param = full_param.conv_param;

  CHECK_NE(req[conv::kWeight], kWriteInplace) << "cannot write weight inplace";
  MKLDNNConvBackward &convBwd = GetConvBwd(attrs, data,
      weight, bias, out_grad, fwd_pd);
  auto out_grad_mem = out_grad.GetMKLDNNDataReorder(
      convBwd.bwdData_pd.diff_dst_primitive_desc());
  if (req[conv::kData]) {
    auto weight_mem = GetWeights(weight,
        convBwd.bwdData_pd.weights_primitive_desc(), param.num_group);
    auto in_grad_mem = CreateMKLDNNMem(in_grad[conv::kData],
        convBwd.bwdData_pd.diff_src_primitive_desc(), req[conv::kData]);
    convBwd.SetDataNewMem(*out_grad_mem, *weight_mem, *in_grad_mem.second);
    MKLDNNStream::Get()->RegisterPrim(convBwd.GetBwdData());
    CommitOutput(in_grad[conv::kData], in_grad_mem);
  }
  if (req[conv::kWeight]) {
    MKLDNNConvBackward &convBwdWeight = GetConvBwd(attrs, data,
        weight, bias, out_grad, fwd_pd);
    if (convBwdWeight.bwdData_pd.diff_dst_primitive_desc() !=
        convBwdWeight.bwdWeights_pd.diff_dst_primitive_desc())
      out_grad_mem = out_grad.GetMKLDNNDataReorder(
          convBwdWeight.bwdWeights_pd.diff_dst_primitive_desc());
    auto data_mem = data.GetMKLDNNDataReorder(
        convBwdWeight.bwdWeights_pd.src_primitive_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(
        in_grad[conv::kWeight],
        convBwdWeight.bwdWeights_pd.diff_weights_primitive_desc(),
        req[conv::kWeight]);
    mkldnn_output_t in_grad_bias;
    if (param.no_bias) {
      convBwdWeight.SetWeightNewMem(*data_mem, *out_grad_mem,
                              *in_grad_weight.second);
      MKLDNNStream::Get()->RegisterPrim(convBwdWeight.GetBwdWeights());
    } else {
      in_grad_bias = CreateMKLDNNMem(
          in_grad[conv::kBias],
          convBwdWeight.bwdWeights_pd.diff_bias_primitive_desc(), req[conv::kBias]);
      convBwdWeight.SetWeightNewMem(*data_mem, *out_grad_mem,
                              *in_grad_weight.second, *in_grad_bias.second);
      MKLDNNStream::Get()->RegisterPrim(convBwdWeight.GetBwdWeights());
      CommitOutput(in_grad[conv::kBias], in_grad_bias);
    }
    CommitOutput(in_grad[conv::kWeight], in_grad_weight);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
