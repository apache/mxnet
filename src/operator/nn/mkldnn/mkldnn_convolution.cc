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

bool SupportMKLDNNConv(const ConvolutionParam& params, const NDArray &input) {
  if (params.kernel.ndim() != 2)
    return false;
  return input.dtype() == mshadow::kFloat32 && input.shape().ndim() == 4;
}

mkldnn::convolution_forward::primitive_desc GetConvFwdImpl(
    const ConvolutionParam& param, const bool is_train, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
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
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  } else if (param.dilate.ndim() == 0) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, bias_md, out_md, strides, padding, padding,
        mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  } else {
    mkldnn::memory::dims dilates{0, 0};
    dilates[0] = param.dilate[0] - 1;
    dilates[1] = param.dilate[1] - 1;
    if (bias == nullptr) {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, dilates, padding, padding,
          mkldnn::padding_kind::zero);
      return mkldnn::convolution_forward::primitive_desc(desc, engine);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
                                             data_md, weight_md, bias_md, out_md, strides,
                                             dilates, padding, padding,
                                             mkldnn::padding_kind::zero);
      return mkldnn::convolution_forward::primitive_desc(desc, engine);
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

MKLDNNConvForward &GetConvFwd(const nnvm::NodeAttrs& attrs, const bool is_train,
                              const NDArray &data, const NDArray &weights,
                              const NDArray *bias, const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash> fwds;
#endif
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
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
    MKLDNNConvForward fwd(param, is_train, data, weights, bias, output);
    auto ins_ret = fwds.insert(
        std::pair<MKLDNNConvSignature, MKLDNNConvForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNConvolutionForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray> &in_data,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);

  auto data = in_data[conv::kData];
  if (data.IsView() && data.IsMKLDNNData())
    data = data.Reorder2Default();

  auto weight = in_data[conv::kWeight];
  if (weight.IsView() && weight.IsMKLDNNData())
    weight = weight.Reorder2Default();

  const NDArray* bias = param.no_bias ? nullptr : &in_data[conv::kBias];

  MKLDNNConvForward &fwd = GetConvFwd(attrs, ctx.is_train, data, weight,
                                      bias, out_data[conv::kOut]);

  auto data_mem = data.GetMKLDNNDataReorder(fwd.fwd_pd.src_primitive_desc());
  const mkldnn::memory *weight_mem;
  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (weight.IsMKLDNNData())
      // This asks the engine to change the layout of the weight array after
      // it's used.
      weight.Reorder2DefaultAsync();
    weight_mem = GetWeights(weight, fwd.fwd_pd.weights_primitive_desc(), param.num_group);
  } else {
    // For inference, we want to reorder the weight array so we don't need to
    // reorder data every time.
    if (weight.IsDefaultData()) {
      weight_mem = GetWeights(weight, fwd.fwd_pd.weights_primitive_desc(), param.num_group);
      // We also need to modify the layout on the original weight array. The
      // data conversion happens after the weight array is used.
      weight.MKLDNNDataReorderAsync(fwd.fwd_pd.weights_primitive_desc());
    } else {
      weight_mem = weight.GetMKLDNNData();
      CHECK(weight_mem->get_primitive_desc() == fwd.fwd_pd.weights_primitive_desc());
    }
  }
  auto out_mem = CreateMKLDNNMem(out_data[conv::kOut], fwd.fwd_pd.dst_primitive_desc(),
                                 req[conv::kOut]);
  const mkldnn::memory *bias_mem = nullptr;
  if (!param.no_bias)
    bias_mem = in_data[conv::kBias].GetMKLDNNDataReorder(fwd.fwd_pd.bias_primitive_desc());
  fwd.SetNewMem(*data_mem, *weight_mem, bias_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());

  CommitOutput(out_data[conv::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
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
    auto ins_ret = bwds.insert(
        std::pair<MKLDNNConvSignature, MKLDNNConvBackward>(key, bwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNConvolutionBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);

  auto data = inputs[conv::kData + 1];
  if (data.IsView() && data.IsMKLDNNData())
    data = data.Reorder2Default();

  auto weight = inputs[conv::kWeight + 1];
  if (weight.IsView() && weight.IsMKLDNNData())
    weight = weight.Reorder2Default();

  const NDArray* bias = param.no_bias ? nullptr : &inputs[conv::kBias + 1];

  auto out_grad = inputs[conv::kOut];
  if (out_grad.IsView() && out_grad.IsMKLDNNData())
    out_grad = out_grad.Reorder2Default();

  mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwdImpl(param, ctx.is_train,
      data, weight, param.no_bias ? nullptr : bias, out_grad);

  CHECK_NE(req[conv::kWeight], kWriteInplace) << "cannot write weight inplace";
  MKLDNNConvBackward &convBwd = GetConvBwd(attrs, data,
      weight, bias, inputs[conv::kOut], fwd_pd);
  auto out_grad_mem = inputs[conv::kOut].GetMKLDNNDataReorder(
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
        weight, &bias, inputs[conv::kOut], fwd_pd);
    if (convBwdWeight.bwdData_pd.diff_dst_primitive_desc() !=
        convBwdWeight.bwdWeights_pd.diff_dst_primitive_desc())
      out_grad_mem = inputs[conv::kOut].GetMKLDNNDataReorder(
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
