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

#include "../convolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

bool SupportMKLDNNConv(const ConvolutionParam& params, const NDArray &input) {
  if (params.kernel.ndim() != 2)
    return false;
  return input.dtype() == mshadow::kFloat32 && input.shape().ndim() == 4;
}

static mkldnn::convolution_forward::primitive_desc GetConvFwdImpl(
    const ConvolutionParam& param, bool is_train, const NDArray &data,
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

class MKLDNNConvForward {
  std::shared_ptr<mkldnn::convolution_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::convolution_forward::primitive_desc fwd_pd;

  MKLDNNConvForward(const ConvolutionParam& param, bool is_train,
                    const NDArray &data, const NDArray &weights,
                    const NDArray *bias, const NDArray &output): fwd_pd(
                        GetConvFwdImpl(param, is_train, data, weights, bias, output)) {
  }

  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                 const mkldnn::memory *bias, const mkldnn::memory &output) {
    if (this->data == nullptr)
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.src_primitive_desc(), data.get_data_handle()));
    else
      this->data->set_data_handle(data.get_data_handle());

    if (this->weight == nullptr)
      this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.weights_primitive_desc(), weight.get_data_handle()));
    else
      this->weight->set_data_handle(weight.get_data_handle());

    if (this->out == nullptr)
      this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.dst_primitive_desc(), output.get_data_handle()));
    else
      this->out->set_data_handle(output.get_data_handle());

    if (bias != nullptr) {
      if (this->bias == nullptr)
        this->bias = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
                fwd_pd.bias_primitive_desc(), bias->get_data_handle()));
      else
        this->bias->set_data_handle(bias->get_data_handle());
      if (this->fwd == nullptr)
        this->fwd = std::shared_ptr<mkldnn::convolution_forward>(
            new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                            mkldnn::primitive::at(*this->weight),
                                            mkldnn::primitive::at(*this->bias),
                                            *this->out));
    } else if (this->fwd == nullptr) {
      this->fwd = std::shared_ptr<mkldnn::convolution_forward>(
          new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                          mkldnn::primitive::at(*this->weight),
                                          *this->out));
    }
  }

  const mkldnn::convolution_forward &GetFwd() const {
    return *fwd;
  }
};

typedef ParamOpSign<ConvolutionParam> MKLDNNConvSignature;

static inline MKLDNNConvForward &GetConvFwd(
    const nnvm::NodeAttrs& attrs, bool is_train,
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
  NDArray weight = in_data[conv::kWeight];
  MKLDNNConvForward &fwd = GetConvFwd(attrs, ctx.is_train, in_data[conv::kData], weight,
      param.no_bias ? nullptr : &in_data[conv::kBias], out_data[conv::kOut]);

  auto data_mem = in_data[conv::kData].GetMKLDNNDataReorder(fwd.fwd_pd.src_primitive_desc());
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

void MKLDNNConvolutionBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwdImpl(param, ctx.is_train,
      inputs[conv::kData + 1], inputs[conv::kWeight + 1],
      param.no_bias ? nullptr : &inputs[conv::kBias + 1], inputs[conv::kOut]);

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
