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
 * \file mkldnn_fully_connected.cc
 * \brief
 * \author Da Zheng
*/

#if MXNET_USE_MKLDNN == 1
#include "mkldnn_fully_connected-inl.h"

namespace mxnet {
namespace op {

mkldnn::inner_product_forward::primitive_desc GetIPFwd(
    const FullyConnectedParam &param, const bool is_train,
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const mkldnn::memory::desc &out_md) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetMemDesc(weight);
  auto engine = CpuEngine::Get()->get_engine();
  auto propagation =
    is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;

  mkldnn::primitive_attr attr;
  if ((param.fuse_requantize.has_value() && param.fuse_requantize.value() == true) ||
      (param.fuse_dequantize.has_value() && param.fuse_dequantize.value() == true)) {
    int mask = 0;
    std::vector<float> scales = {0.0};
    if (param.requantize_scales.size()) {
      scales[0] = param.requantize_scales[0];
    } else if (param.output_scales.size()) {
      scales[0] = param.output_scales[0];
    } else {
      LOG(FATAL) << "Must specified either output_scales or requantize_scales!";
    }

    attr.set_output_scales(mask, scales);
    attr.set_int_output_round_mode(round_nearest);
  }

  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::inner_product_forward::desc ipFwd_desc(propagation,
        data_md, weight_md, bias_md, out_md);
    return mkldnn::inner_product_forward::primitive_desc(ipFwd_desc, attr, engine);
  } else {
    mkldnn::inner_product_forward::desc ipFwd_desc(propagation,
        data_md, weight_md, out_md);
    return mkldnn::inner_product_forward::primitive_desc(ipFwd_desc, attr, engine);
  }
}

inline static mkldnn::inner_product_backward_data::primitive_desc GetIpBwdData(
    const NDArray &data, const NDArray &weight, const NDArray &output,
    mkldnn::inner_product_forward::primitive_desc ipFwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetMemDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::inner_product_backward_data::desc desc(data_md, weight_md, out_md);
  return mkldnn::inner_product_backward_data::primitive_desc(desc, engine, ipFwd_pd);
}

inline static mkldnn::inner_product_backward_weights::primitive_desc GetIPBwdWeights(
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const NDArray &output, mkldnn::inner_product_forward::primitive_desc ipFwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetMemDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::inner_product_backward_weights::desc ipBwdWeights_desc(data_md,
        weight_md, bias_md, out_md);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        ipBwdWeights_desc, engine, ipFwd_pd);
  } else {
    mkldnn::inner_product_backward_weights::desc ipBwdWeights_desc(data_md,
        weight_md, out_md);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        ipBwdWeights_desc, engine, ipFwd_pd);
  }
}

void MKLDNNFullyConnectForward::SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                                          const mkldnn::memory *bias, const mkldnn::memory &output) {
  if (this->data == nullptr)
    this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
            ipFwd_pd.src_primitive_desc(), data.get_data_handle()));
  else
    this->data->set_data_handle(data.get_data_handle());

  if (this->weight == nullptr)
    this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
            ipFwd_pd.weights_primitive_desc(), weight.get_data_handle()));
  else
    this->weight->set_data_handle(weight.get_data_handle());

  if (this->out == nullptr)
    this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
            ipFwd_pd.dst_primitive_desc(), output.get_data_handle()));
  else
    this->out->set_data_handle(output.get_data_handle());

  if (bias != nullptr) {
    if (this->bias == nullptr)
      this->bias = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
      ipFwd_pd.bias_primitive_desc(), bias->get_data_handle()));
    else
      this->bias->set_data_handle(bias->get_data_handle());
    if (this->ipFwd == nullptr)
      this->ipFwd = std::shared_ptr<mkldnn::inner_product_forward>(
          new mkldnn::inner_product_forward(
              ipFwd_pd, mkldnn::primitive::at(*this->data),
              mkldnn::primitive::at(*this->weight),
              mkldnn::primitive::at(*this->bias), *this->out));
  } else if (this->ipFwd == nullptr) {
    this->ipFwd = std::shared_ptr<mkldnn::inner_product_forward>(
        new mkldnn::inner_product_forward(
            ipFwd_pd, mkldnn::primitive::at(*this->data),
            mkldnn::primitive::at(*this->weight), *this->out));
  }
}

MKLDNNFullyConnectForward &GetFCFwd(
    const FullyConnectedParam &param, const bool is_train,
    const NDArray &data, const NDArray &weight,
    const NDArray *bias, const mkldnn::memory::desc &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNFullyconSignature,
              MKLDNNFullyConnectForward, OpHash> fcFwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNFullyconSignature,
              MKLDNNFullyConnectForward, OpHash> fcFwds;
#endif
  MKLDNNFullyconSignature key(param);
  key.AddSign(data);
  key.AddSign(weight);
  key.AddSign(is_train);

  if (bias)
    key.AddSign(*bias);

  auto it = fcFwds.find(key);
  if (it == fcFwds.end()) {
    MKLDNNFullyConnectForward fcFwd(param, is_train, data, weight, bias, output);
    it = AddToCache(&fcFwds, key, fcFwd);
  }
  return it->second;
}

void MKLDNNFCForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                     const std::vector<NDArray> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const mxnet::TShape& ishape = in_data[fullc::kData].shape();
  const mxnet::TShape& oshape = out_data[fullc::kOut].shape();
  NDArray weight = in_data[fullc::kWeight];
  NDArray data = in_data[fullc::kData];
  // If the input data is a view of an MKLDNN array, we should create a new
  // NDArray with reordered data.
  if (data.IsMKLDNNData() && data.IsView())
    data = in_data[fullc::kData].Reorder2Default();

  auto out_md = GetMemDesc(out_data[fullc::kOut]);
  if (data.shape().ndim() != 2 && !param.flatten) {
    data = data.MKLDNNDataReshape(Shape2(ishape.ProdShape(0, ishape.ndim()-1),
                                     ishape[ishape.ndim()-1]));
    mkldnn::memory::dims out_dims{static_cast<int>(oshape.ProdShape(0, oshape.ndim()-1)),
      static_cast<int>(oshape[ishape.ndim()-1])};
    out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(out_data[fullc::kOut].dtype()),
      mkldnn::memory::format::any);
  } else if (data.shape().ndim() != 2) {
    data = data.MKLDNNDataReshape(Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())));
    mkldnn::memory::dims out_dims{static_cast<int>(oshape[0]),
      static_cast<int>(oshape.ProdShape(1, oshape.ndim()))};
    out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(out_data[fullc::kOut].dtype()),
      mkldnn::memory::format::any);
  }

  MKLDNNFullyConnectForward &FCFwd =
      GetFCFwd(param, ctx.is_train, data, weight,
               param.no_bias ? nullptr : &in_data[fullc::kBias], out_md);
  auto data_mem = data.GetMKLDNNDataReorder(FCFwd.ipFwd_pd.src_primitive_desc());
  auto weight_mem = weight.GetMKLDNNDataReorder(FCFwd.ipFwd_pd.weights_primitive_desc());
  auto out_mem = CreateMKLDNNMem(out_data[fullc::kOut],
      FCFwd.ipFwd_pd.dst_primitive_desc(), req[fullc::kOut], &data);
  if (!param.no_bias) {
    auto bias_mem = in_data[fullc::kBias].GetMKLDNNDataReorder(
        FCFwd.ipFwd_pd.bias_primitive_desc());
    FCFwd.SetNewMem(*data_mem, *weight_mem, bias_mem, *out_mem.second);
  } else {
    FCFwd.SetNewMem(*data_mem, *weight_mem, nullptr, *out_mem.second);
  }
  MKLDNNStream::Get()->RegisterPrim(FCFwd.GetIpFwd());
  CommitOutput(out_data[fullc::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}

void MKLDNNFCBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                      const std::vector<NDArray> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[fullc::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const mxnet::TShape& ishape = inputs[fullc::kData + 1].shape();
  const mxnet::TShape& oshape = inputs[fullc::kOut].shape();

  NDArray weight = inputs[fullc::kWeight + 1];
  NDArray data = inputs[fullc::kData + 1];
  if (data.shape().ndim() != 2 && !param.flatten)
    data = data.MKLDNNDataReshape(Shape2(ishape.ProdShape(0, ishape.ndim()-1),
                                     ishape[ishape.ndim()-1]));
  else if (data.shape().ndim() != 2)
    data = data.MKLDNNDataReshape(Shape2(ishape[0],
                                     ishape.ProdShape(1, ishape.ndim())));
  NDArray out_grad = inputs[fullc::kOut];
  if (out_grad.shape().ndim() != 2 && !param.flatten)
    out_grad = out_grad.MKLDNNDataReshape(Shape2(oshape.ProdShape(0, oshape.ndim()-1),
                                             oshape[oshape.ndim()-1]));
  else if (out_grad.shape().ndim() != 2)
    out_grad = out_grad.MKLDNNDataReshape(Shape2(oshape[0],
                                             oshape.ProdShape(1, oshape.ndim())));

  mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(param, ctx.is_train,
      data, weight, param.no_bias ? nullptr : &in_grad[fullc::kBias], GetMemDesc(out_grad));

  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  if (req[fullc::kData]) {
    mkldnn::inner_product_backward_data::primitive_desc ipBwdData_pd = GetIpBwdData(
        data, weight, out_grad, ipFwd_pd);
    auto out_grad_mem = out_grad.GetMKLDNNDataReorder(
        ipBwdData_pd.diff_dst_primitive_desc());
    auto weight_mem = weight.GetMKLDNNDataReorder(ipBwdData_pd.weights_primitive_desc());
    auto in_grad_mem = CreateMKLDNNMem(in_grad[fullc::kData],
                                       ipBwdData_pd.diff_src_primitive_desc(),
                                       req[fullc::kData]);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::inner_product_backward_data(
          ipBwdData_pd, *out_grad_mem, *weight_mem, *in_grad_mem.second));
    CommitOutput(in_grad[fullc::kData], in_grad_mem);
  }
  if (req[fullc::kWeight]) {
    mkldnn::inner_product_backward_weights::primitive_desc ipBwdWeights_pd
      = GetIPBwdWeights(data, weight, param.no_bias ? nullptr : &in_grad[fullc::kBias],
          out_grad, ipFwd_pd);
    auto out_grad_mem = out_grad.GetMKLDNNDataReorder(
        ipBwdWeights_pd.diff_dst_primitive_desc());
    auto data_mem = data.GetMKLDNNDataReorder(ipBwdWeights_pd.src_primitive_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(in_grad[fullc::kWeight],
                                                 ipBwdWeights_pd.diff_weights_primitive_desc(),
                                                 req[fullc::kWeight]);
    mkldnn_output_t in_grad_bias;
    if (param.no_bias) {
      MKLDNNStream::Get()->RegisterPrim(mkldnn::inner_product_backward_weights(
            ipBwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second));
    } else {
      in_grad_bias = CreateMKLDNNMem(in_grad[fullc::kBias],
                                     ipBwdWeights_pd.diff_bias_primitive_desc(),
                                     req[fullc::kBias]);
      MKLDNNStream::Get()->RegisterPrim(mkldnn::inner_product_backward_weights(
            ipBwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second,
            *in_grad_bias.second));
    }
    CommitOutput(in_grad[fullc::kWeight], in_grad_weight);
    CommitOutput(in_grad[fullc::kBias], in_grad_bias);
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
