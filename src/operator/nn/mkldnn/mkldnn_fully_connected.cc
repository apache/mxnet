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

#include "../fully_connected-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

inline static mkldnn::inner_product_forward::primitive_desc GetIPFwd(
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const NDArray &output) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Instance().get_engine();
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::inner_product_forward::desc ipFwd_desc(mkldnn::prop_kind::forward_training,
        data_md, weight_md, bias_md, out_md);
    return mkldnn::inner_product_forward::primitive_desc(ipFwd_desc, engine);
  }
  else {
    mkldnn::inner_product_forward::desc ipFwd_desc(mkldnn::prop_kind::forward_training,
        data_md, weight_md, out_md);
    return mkldnn::inner_product_forward::primitive_desc(ipFwd_desc, engine);
  }
}

inline static mkldnn::inner_product_backward_data::primitive_desc GetIpBwdData(
    const NDArray &data, const NDArray &weight, const NDArray &output,
    mkldnn::inner_product_forward::primitive_desc ipFwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Instance().get_engine();
  mkldnn::inner_product_backward_data::desc desc(data_md, weight_md, out_md);
  return mkldnn::inner_product_backward_data::primitive_desc(desc, engine, ipFwd_pd);
}

inline static mkldnn::inner_product_backward_weights::primitive_desc GetIPBwdWeights(
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const NDArray &output, mkldnn::inner_product_forward::primitive_desc ipFwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Instance().get_engine();
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::inner_product_backward_weights::desc ipBwdWeights_desc(data_md,
        weight_md, bias_md, out_md);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        ipBwdWeights_desc, engine, ipFwd_pd);
  }
  else {
    mkldnn::inner_product_backward_weights::desc ipBwdWeights_desc(data_md,
        weight_md, out_md);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        ipBwdWeights_desc, engine, ipFwd_pd);
  }
}

void MKLDNNFC_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  std::vector<mkldnn::primitive> net;
  mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(
      in_data[fullc::kData], in_data[fullc::kWeight],
      param.no_bias ? nullptr : &in_data[fullc::kBias], out_data[fullc::kOut]);
  auto data_mem = in_data[fullc::kData].GetMKLDNNData(ipFwd_pd.src_primitive_desc(), net);
  auto weight_mem = in_data[fullc::kWeight].GetMKLDNNData(
      ipFwd_pd.weights_primitive_desc(), net);
  auto out_mem = const_cast<NDArray &>(out_data[fullc::kOut]).CreateMKLDNNData(
      ipFwd_pd.dst_primitive_desc());
  bool copy_back = false;
  if (out_mem == nullptr) {
    out_mem = CreateMKLDNNMem(ipFwd_pd.dst_primitive_desc());
    copy_back = true;
  }
  if (param.no_bias) {
    net.push_back(mkldnn::inner_product_forward(ipFwd_pd, *data_mem, *weight_mem,
          *out_mem));
  } else {
    auto bias_mem = in_data[fullc::kBias].GetMKLDNNData(ipFwd_pd.bias_primitive_desc(), net);
    net.push_back(mkldnn::inner_product_forward(ipFwd_pd, *data_mem, *weight_mem,
          *bias_mem, *out_mem));
  }
  if (copy_back)
    const_cast<NDArray &>(out_data[fullc::kOut]).CopyFrom(*out_mem, net);
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

void MKLDNNFC_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &inputs, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &outputs) {
  const std::vector<NDArray> &in_grad = outputs;
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(
      inputs[fullc::kData + 1], inputs[fullc::kWeight + 1],
      param.no_bias ? nullptr : &in_grad[fullc::kBias], inputs[fullc::kOut]);

  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  std::vector<mkldnn::primitive> net;
  mkldnn_mem_ptr in_grad_mem, in_grad_weight, in_grad_bias;
  if (req[fullc::kData]) {
    mkldnn::inner_product_backward_data::primitive_desc ipBwdData_pd = GetIpBwdData(
        inputs[fullc::kData + 1], inputs[fullc::kWeight + 1], inputs[fullc::kOut],
        ipFwd_pd);
    auto out_grad_mem = inputs[fullc::kOut].GetMKLDNNData(
        ipBwdData_pd.diff_dst_primitive_desc(), net);
    auto weight_mem = inputs[fullc::kWeight + 1].GetMKLDNNData(
        ipBwdData_pd.weights_primitive_desc(), net);
    in_grad_mem = const_cast<NDArray &>(in_grad[fullc::kData]).CreateMKLDNNData(
        ipBwdData_pd.diff_src_primitive_desc());
    bool copy_back = false;
    if (in_grad_mem == nullptr) {
      in_grad_mem = CreateMKLDNNMem(ipBwdData_pd.diff_src_primitive_desc());
      copy_back = true;
    }
    net.push_back(mkldnn::inner_product_backward_data(ipBwdData_pd, *out_grad_mem,
          *weight_mem, *in_grad_mem));
    if (copy_back)
      const_cast<NDArray &>(in_grad[fullc::kData]).CopyFrom(*in_grad_mem, net);
  }
  if (req[fullc::kWeight]) {
    mkldnn::inner_product_backward_weights::primitive_desc ipBwdWeights_pd
      = GetIPBwdWeights(inputs[fullc::kData + 1], inputs[fullc::kWeight + 1],
          param.no_bias ? nullptr : &in_grad[fullc::kBias], inputs[fullc::kOut],
          ipFwd_pd);
    auto out_grad_mem = inputs[fullc::kOut].GetMKLDNNData(
        ipBwdWeights_pd.diff_dst_primitive_desc(), net);
    auto data_mem = inputs[fullc::kData + 1].GetMKLDNNData(
        ipBwdWeights_pd.src_primitive_desc(), net);
    in_grad_weight = const_cast<NDArray &>(in_grad[fullc::kWeight]).CreateMKLDNNData(
        ipBwdWeights_pd.diff_weights_primitive_desc());
    bool copy_back_weight = false;
    bool copy_back_bias = false;
    if (in_grad_weight == nullptr) {
      in_grad_weight = CreateMKLDNNMem(ipBwdWeights_pd.diff_weights_primitive_desc());
      copy_back_weight = true;
    }
    if (param.no_bias) {
      net.push_back(mkldnn::inner_product_backward_weights(ipBwdWeights_pd,
            *data_mem, *out_grad_mem, *in_grad_weight));
    } else {
      in_grad_bias = const_cast<NDArray &>(in_grad[fullc::kBias]).CreateMKLDNNData(
          ipBwdWeights_pd.diff_bias_primitive_desc());
      if (in_grad_bias == nullptr) {
        in_grad_bias = CreateMKLDNNMem(ipBwdWeights_pd.diff_bias_primitive_desc());
        copy_back_bias = true;
      }
      net.push_back(mkldnn::inner_product_backward_weights(ipBwdWeights_pd,
            *data_mem, *out_grad_mem, *in_grad_weight, *in_grad_bias));
    }
    if (copy_back_weight)
      const_cast<NDArray &>(in_grad[fullc::kWeight]).CopyFrom(*in_grad_weight, net);
    if (copy_back_bias)
      const_cast<NDArray &>(in_grad[fullc::kBias]).CopyFrom(*in_grad_bias, net);
  }
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

}
}
#endif  // MXNET_USE_MKLDNN == 1
