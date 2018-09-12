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
  if (params.kernel.ndim() != 2)
    return false;
  return input.shape().ndim() == 4;
}

inline static mkldnn::memory::desc GetInDataMemDesc(const NDArray &arr) {
  mkldnn::memory::dims dims(arr.shape().ndim());
  for (size_t i = 0; i < dims.size(); i++) dims[i] = arr.shape()[i];
  int mkldnn_dtype;
  // For INT8 case, currently we only support uint8 as input data so need
  // to create the memory primitive of uint8 type
  if (arr.dtype() == mshadow::kInt8) {
    mkldnn_dtype = mshadow::kUint8;
  } else {
    mkldnn_dtype = arr.dtype();
  }
  return mkldnn::memory::desc{dims, get_mkldnn_type(mkldnn_dtype),
                              mkldnn::memory::format::any};
}

mkldnn::convolution_forward::primitive_desc GetConvFwdImpl(
    const MKLDNNConvFullParam &param, const bool is_train,
    const NDArray &data, const NDArray &weights, const NDArray *bias,
    const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto data_md = GetInDataMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.conv_param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  CHECK_GE(param.conv_param.stride.ndim(), 2U);
  CHECK_GE(param.conv_param.pad.ndim(), 2U);
  CHECK_GE(param.conv_param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.conv_param.stride[0];
  strides[1] = param.conv_param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.conv_param.pad[0];
  padding[1] = param.conv_param.pad[1];
  mkldnn::primitive_attr attr;
  mkldnn::post_ops ops;
  if (param.mkldnn_param.with_relu) {
    float scale = 1.0f;            // for fp32, scale is 1.
    float alpha = 0.0f;            // negative slope for mkldnn_eltwise_relu.
    float beta = 1.0f;             // ignored for mkldnn_eltwise_relu.
    ops.append_eltwise(scale, eltwise_relu, alpha, beta);
  }
  if (param.mkldnn_param.with_sum) {
    ops.append_sum(param.sum_scale);
  }
  if (param.mkldnn_param.with_postsum_relu) {
    float scale = 1.0f;            // for fp32, scale is 1.
    float alpha = 0.0f;            // negative slope for mkldnn_eltwise_relu.
    float beta = 1.0f;             // ignored for mkldnn_eltwise_relu.
    ops.append_eltwise(scale, eltwise_relu, alpha, beta);
  }
  attr.set_post_ops(ops);

  if (param.mkldnn_param.quantized) {
    int mask = param.mkldnn_param.weight_channelwise_scale ? 2 : 0;
    attr.set_output_scales(mask, param.requantize_scales);
    attr.set_int_output_round_mode(round_nearest);
  }

  if (param.conv_param.dilate.ndim() == 0 && bias == nullptr) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
  } else if (param.conv_param.dilate.ndim() == 0) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, bias_md, out_md, strides, padding, padding,
        mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
  } else {
    mkldnn::memory::dims dilates{0, 0};
    dilates[0] = param.conv_param.dilate[0] - 1;
    dilates[1] = param.conv_param.dilate[1] - 1;
    if (bias == nullptr) {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, dilates, padding, padding,
          mkldnn::padding_kind::zero);
      return mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
                                             data_md, weight_md, bias_md, out_md, strides,
                                             dilates, padding, padding,
                                             mkldnn::padding_kind::zero);
      return mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
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
  CHECK_GE(param.stride.ndim(), 2U);
  CHECK_GE(param.pad.ndim(), 2U);
  CHECK_GE(param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];
  if (param.dilate.ndim() == 0) {
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(desc, engine, fwd_pd);
  } else {
    mkldnn::memory::dims dilates{0, 0};
    dilates[0] = param.dilate[0] - 1;
    dilates[1] = param.dilate[1] - 1;
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, dilates, padding, padding,
        mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(desc, engine, fwd_pd);
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
  CHECK_GE(param.stride.ndim(), 2U);
  CHECK_GE(param.pad.ndim(), 2U);
  CHECK_GE(param.dilate.ndim(), 2U);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];
  if (param.dilate.ndim() == 0 && bias == nullptr) {
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
  } else if (param.dilate.ndim() == 0) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, bias_md, out_md, strides, padding, padding,
        mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
  } else {
    mkldnn::memory::dims dilates{0, 0};
    dilates[0] = param.dilate[0] - 1;
    dilates[1] = param.dilate[1] - 1;
    if (bias == nullptr) {
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, dilates, padding, padding,
          mkldnn::padding_kind::zero);
      return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
                                                      data_md, weight_md, bias_md, out_md,
                                                      strides, dilates, padding, padding,
                                                      mkldnn::padding_kind::zero);
      return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
    }
  }
}

void MKLDNNConvForward::SetNewMem(const mkldnn::memory &data,
                                  const mkldnn::memory &weight,
                                  const mkldnn::memory *bias,
                                  const mkldnn::memory &output) {
  if (this->data_ == nullptr)
    this->data_ = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
            fwd_pd.src_primitive_desc(), data.get_data_handle()));
  else
    this->data_->set_data_handle(data.get_data_handle());

  if (this->weight_ == nullptr)
    this->weight_ = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
            fwd_pd.weights_primitive_desc(), weight.get_data_handle()));
  else
    this->weight_->set_data_handle(weight.get_data_handle());

  if (this->out_ == nullptr)
    this->out_ = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
            fwd_pd.dst_primitive_desc(), output.get_data_handle()));
  else
    this->out_->set_data_handle(output.get_data_handle());

  if (bias != nullptr) {
    if (this->bias_ == nullptr)
      this->bias_ = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.bias_primitive_desc(), bias->get_data_handle()));
    else
      this->bias_->set_data_handle(bias->get_data_handle());
    if (this->fwd_ == nullptr)
      this->fwd_ = std::shared_ptr<mkldnn::convolution_forward>(
          new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data_),
                                          mkldnn::primitive::at(*this->weight_),
                                          mkldnn::primitive::at(*this->bias_),
                                          *this->out_));
  } else if (this->fwd_ == nullptr) {
    this->fwd_ = std::shared_ptr<mkldnn::convolution_forward>(
        new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data_),
                                        mkldnn::primitive::at(*this->weight_),
                                        *this->out_));
  }
}

MKLDNNConvForward &GetConvFwd(const MKLDNNConvFullParam &param,
                              const bool is_train, const NDArray &data,
                              const NDArray &weights, const NDArray *bias,
                              const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash> fwds;
#endif
  MKLDNNConvSignature key(param.conv_param);
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
    MKLDNNConvForward fwd(param, is_train, data, weights, bias, output);
    auto ins_ret = fwds.insert(
        std::pair<MKLDNNConvSignature, MKLDNNConvForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
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
  NDArray weight = in_data[conv::kWeight];
  bool no_bias = param.conv_param.no_bias && !param.mkldnn_param.with_bn;
  auto data_mem = in_data[conv::kData].GetMKLDNNDataReorder(
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
        const_cast<mkldnn::memory *>(out_data[conv::kOut].GetMKLDNNDataReorder(
            fwd->fwd_pd.dst_primitive_desc())));
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
      param, ctx.is_train, in_data[conv::kData], in_data[conv::kWeight],
      param.conv_param.no_bias ? nullptr : &in_data[conv::kBias],
      out_data[conv::kOut]);
  MKLDNNConvolutionForwardFullFeature(param, ctx, &fwd, in_data, req, out_data);
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
  mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwdImpl(
      full_param, ctx.is_train, inputs[conv::kData + 1], inputs[conv::kWeight + 1],
      full_param.conv_param.no_bias ? nullptr : &inputs[conv::kBias + 1],
      inputs[conv::kOut]);
  const ConvolutionParam &param = full_param.conv_param;

  CHECK_NE(req[conv::kWeight], kWriteInplace) << "cannot write weight inplace";
  mkldnn::convolution_backward_data::primitive_desc bwdData_pd
    = GetConvBwdData(param, inputs[conv::kData + 1], inputs[conv::kWeight + 1],
        inputs[conv::kOut], fwd_pd);
  auto out_grad_mem = inputs[conv::kOut].GetMKLDNNDataReorder(
      bwdData_pd.diff_dst_primitive_desc());
  if (req[conv::kData]) {
    auto weight_mem = GetWeights(inputs[conv::kWeight + 1],
        bwdData_pd.weights_primitive_desc(), param.num_group);
    auto in_grad_mem = CreateMKLDNNMem(in_grad[conv::kData],
        bwdData_pd.diff_src_primitive_desc(), req[conv::kData]);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::convolution_backward_data(bwdData_pd,
          *out_grad_mem, *weight_mem, *in_grad_mem.second));
    CommitOutput(in_grad[conv::kData], in_grad_mem);
  }
  if (req[conv::kWeight]) {
    mkldnn::convolution_backward_weights::primitive_desc bwdWeights_pd
        = GetConvBwdWeights(param, inputs[conv::kData + 1], inputs[conv::kWeight + 1],
                            param.no_bias ? nullptr : &inputs[conv::kBias + 1],
                            inputs[conv::kOut], fwd_pd);
    if (bwdData_pd.diff_dst_primitive_desc() != bwdWeights_pd.diff_dst_primitive_desc())
      out_grad_mem = inputs[conv::kOut].GetMKLDNNDataReorder(
          bwdWeights_pd.diff_dst_primitive_desc());
    auto data_mem = inputs[conv::kData + 1].GetMKLDNNDataReorder(
        bwdWeights_pd.src_primitive_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(in_grad[conv::kWeight],
                                                 bwdWeights_pd.diff_weights_primitive_desc(),
                                                 req[conv::kWeight]);
    mkldnn_output_t in_grad_bias;
    if (param.no_bias) {
      MKLDNNStream::Get()->RegisterPrim(mkldnn::convolution_backward_weights(
              bwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second));
    } else {
      in_grad_bias = CreateMKLDNNMem(in_grad[conv::kBias],
                                     bwdWeights_pd.diff_bias_primitive_desc(),
                                     req[conv::kBias]);
      MKLDNNStream::Get()->RegisterPrim(mkldnn::convolution_backward_weights(
              bwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second,
              *in_grad_bias.second));
      CommitOutput(in_grad[conv::kBias], in_grad_bias);
    }
    CommitOutput(in_grad[conv::kWeight], in_grad_weight);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
