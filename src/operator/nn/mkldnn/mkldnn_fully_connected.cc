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
    const mkldnn::memory::desc &data_desc, const mkldnn::memory::desc &weight_desc,
    const mkldnn::memory::desc &out_desc, const mkldnn::engine &engine,
    std::shared_ptr<const mkldnn::memory> bias_mem) {
  if (bias_mem) {
    auto bias_desc = bias_mem->get_primitive_desc().desc();
    mkldnn::inner_product_forward::desc ipFwd_desc(mkldnn::prop_kind::forward_training,
        data_desc, weight_desc, bias_desc, out_desc);
    return mkldnn::inner_product_forward::primitive_desc(ipFwd_desc, engine);
  }
  else {
    mkldnn::inner_product_forward::desc ipFwd_desc(mkldnn::prop_kind::forward_training,
        data_desc, weight_desc, out_desc);
    return mkldnn::inner_product_forward::primitive_desc(ipFwd_desc, engine);
  }
}

inline static mkldnn::inner_product_backward_weights::primitive_desc GetIPBwd(
    const mkldnn::memory::desc &data_desc, const mkldnn::memory::desc &weight_desc,
    const mkldnn::memory::desc &out_desc, const mkldnn::engine &engine,
    mkldnn::inner_product_forward::primitive_desc ipFwd_pd,
    std::shared_ptr<const mkldnn::memory> bias_mem) {
  if (bias_mem) {
    mkldnn::inner_product_backward_weights::desc ipBwdWeights_desc(data_desc,
        weight_desc, bias_mem->get_primitive_desc().desc(), out_desc);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        ipBwdWeights_desc, engine, ipFwd_pd);
  }
  else {
    mkldnn::inner_product_backward_weights::desc ipBwdWeights_desc(data_desc,
        weight_desc, out_desc);
    return mkldnn::inner_product_backward_weights::primitive_desc(
        ipBwdWeights_desc, engine, ipFwd_pd);
  }
}

void MKLDNNFC_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  auto data_mem = in_data[fullc::kData].GetMKLDNNData();
  auto data_desc = data_mem->get_primitive_desc().desc();
  auto cpu_engine = data_mem->get_primitive_desc().get_engine();
  auto weight_mem = GetWeights(in_data[fullc::kWeight], cpu_engine);
  auto weight_desc = weight_mem->get_primitive_desc().desc();
  auto out_mem = const_cast<NDArray &>(out_data[fullc::kOut]).GetMKLDNNData();
  auto out_desc = out_mem->get_primitive_desc().desc();

  std::vector<mkldnn::primitive> net;
  if (param.no_bias) {
    mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(
        data_desc, weight_desc, out_desc, cpu_engine, nullptr);
    CHECK(ipFwd_pd.src_primitive_desc() == data_mem->get_primitive_desc());
    CHECK(ipFwd_pd.weights_primitive_desc() == weight_mem->get_primitive_desc());
    CHECK(ipFwd_pd.dst_primitive_desc() == out_mem->get_primitive_desc());
    net.push_back(mkldnn::inner_product_forward(ipFwd_pd, *data_mem, *weight_mem,
          *out_mem));
  } else {
    auto bias_mem = in_data[fullc::kBias].GetMKLDNNData();
    mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(
        data_desc, weight_desc, out_desc, cpu_engine, bias_mem);
    CHECK(ipFwd_pd.src_primitive_desc() == data_mem->get_primitive_desc());
    CHECK(ipFwd_pd.weights_primitive_desc() == weight_mem->get_primitive_desc());
    CHECK(ipFwd_pd.bias_primitive_desc() == bias_mem->get_primitive_desc());
    CHECK(ipFwd_pd.dst_primitive_desc() == out_mem->get_primitive_desc());
    net.push_back(mkldnn::inner_product_forward(ipFwd_pd, *data_mem, *weight_mem,
          *bias_mem, *out_mem));
  }
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

void MKLDNNFC_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &inputs, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &outputs) {
  const std::vector<NDArray> &in_grad = outputs;
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  auto out_grad_mem = inputs[fullc::kOut].GetMKLDNNData();
  auto out_grad_desc = out_grad_mem->get_primitive_desc().desc();
  auto data_mem = inputs[fullc::kData + 1].GetMKLDNNData();
  auto data_desc = data_mem->get_primitive_desc().desc();
  auto cpu_engine = data_mem->get_primitive_desc().get_engine();
  auto weight_mem = GetWeights(inputs[fullc::kWeight + 1], cpu_engine);
  auto weight_desc = weight_mem->get_primitive_desc().desc();
  std::shared_ptr<const mkldnn::memory> in_grad_bias;
  if (!param.no_bias)
    in_grad_bias = const_cast<NDArray &>(in_grad[fullc::kBias]).GetMKLDNNData();
  mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(data_desc,
      weight_desc, out_grad_desc, cpu_engine, in_grad_bias);

  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  std::vector<mkldnn::primitive> net;
  mkldnn_mem_ptr in_grad_mem, in_grad_weight;
  if (req[fullc::kData]) {
    mkldnn::inner_product_backward_data::desc ipBwdData_desc(data_desc, weight_desc,
        out_grad_desc);
    mkldnn::inner_product_backward_data::primitive_desc ipBwdData_pd(ipBwdData_desc,
        cpu_engine, ipFwd_pd);
    CHECK(ipBwdData_pd.diff_dst_primitive_desc() == out_grad_mem->get_primitive_desc());
    CHECK(ipBwdData_pd.weights_primitive_desc() == weight_mem->get_primitive_desc());
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
    mkldnn::inner_product_backward_weights::primitive_desc ipBwdWeights_pd = GetIPBwd(
        data_desc, weight_desc, out_grad_desc, cpu_engine, ipFwd_pd, in_grad_bias);
    CHECK(ipBwdWeights_pd.diff_dst_primitive_desc() == out_grad_mem->get_primitive_desc());
    CHECK(ipBwdWeights_pd.src_primitive_desc() == data_mem->get_primitive_desc());
    in_grad_weight = const_cast<NDArray &>(in_grad[fullc::kWeight]).CreateMKLDNNData(
        ipBwdWeights_pd.diff_weights_primitive_desc());
    bool copy_back_weight = false;
    if (in_grad_weight == nullptr) {
      in_grad_weight = CreateMKLDNNMem(ipBwdWeights_pd.diff_weights_primitive_desc());
      copy_back_weight = true;
    }
    if (param.no_bias) {
      net.push_back(mkldnn::inner_product_backward_weights(ipBwdWeights_pd,
            *data_mem, *out_grad_mem, *in_grad_weight));
    } else {
      net.push_back(mkldnn::inner_product_backward_weights(ipBwdWeights_pd,
            *data_mem, *out_grad_mem, *in_grad_weight, *in_grad_bias));
    }
    if (copy_back_weight)
      const_cast<NDArray &>(in_grad[fullc::kWeight]).CopyFrom(*in_grad_weight, net);
  }
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

}
}
#endif  // MXNET_USE_MKLDNN == 1
