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
  auto weight_md = GetMemDesc(weight);
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
  auto weight_md = GetMemDesc(weight);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Instance().get_engine();
  mkldnn::inner_product_backward_data::desc desc(data_md, weight_md, out_md);
  return mkldnn::inner_product_backward_data::primitive_desc(desc, engine, ipFwd_pd);
}

inline static mkldnn::inner_product_backward_weights::primitive_desc GetIPBwdWeights(
    const NDArray &data, const NDArray &weight, const NDArray *bias,
    const NDArray &output, mkldnn::inner_product_forward::primitive_desc ipFwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetMemDesc(weight);
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
  const TShape& ishape = in_data[fullc::kData].shape();
  NDArray weight = in_data[fullc::kWeight];
  NDArray data = in_data[fullc::kData];
  if (data.shape().ndim() != 2 && !param.flatten)
    data = data.Reshape(Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]));
  else if (data.shape().ndim() != 2)
    data = data.Reshape(Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())));

  mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(data, weight,
      param.no_bias ? nullptr : &in_data[fullc::kBias], out_data[fullc::kOut]);
  auto data_mem = data.GetMKLDNNDataReorder(ipFwd_pd.src_primitive_desc());
  auto weight_mem = weight.GetMKLDNNDataReorder(ipFwd_pd.weights_primitive_desc());
  auto out_mem = CreateMKLDNNMem(out_data[fullc::kOut],
      ipFwd_pd.dst_primitive_desc(), req[fullc::kOut]);
  if (param.no_bias) {
    MKLDNNStream::Instance().RegisterPrim(mkldnn::inner_product_forward(
          ipFwd_pd, *data_mem, *weight_mem, *out_mem.second));
  } else {
    auto bias_mem = in_data[fullc::kBias].GetMKLDNNDataReorder(ipFwd_pd.bias_primitive_desc());
    MKLDNNStream::Instance().RegisterPrim(mkldnn::inner_product_forward(ipFwd_pd,
          *data_mem, *weight_mem, *bias_mem, *out_mem.second));
  }
  CommitOutput(out_data[fullc::kOut], out_mem);
  MKLDNNStream::Instance().Submit();
}

void MKLDNNFC_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &inputs, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &outputs) {
  const std::vector<NDArray> &in_grad = outputs;
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  const TShape& ishape = inputs[fullc::kData + 1].shape();
  const TShape& oshape = inputs[fullc::kOut].shape();

  NDArray weight = inputs[fullc::kWeight + 1];
  NDArray data = inputs[fullc::kData + 1];
  if (data.shape().ndim() != 2 && !param.flatten)
    data = data.Reshape(Shape2(ishape.ProdShape(0, ishape.ndim()-1), ishape[ishape.ndim()-1]));
  else if (data.shape().ndim() != 2)
    data = data.Reshape(Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())));
  NDArray out_grad = inputs[fullc::kOut];
  if (out_grad.shape().ndim() != 2 && !param.flatten)
    out_grad = out_grad.Reshape(Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]));
  else if (out_grad.shape().ndim() != 2)
    out_grad = out_grad.Reshape(Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())));

  mkldnn::inner_product_forward::primitive_desc ipFwd_pd = GetIPFwd(data, weight,
      param.no_bias ? nullptr : &in_grad[fullc::kBias], out_grad);

  CHECK_NE(req[fullc::kWeight], kWriteInplace) << "cannot write weight inplace";
  if (req[fullc::kData]) {
    mkldnn::inner_product_backward_data::primitive_desc ipBwdData_pd = GetIpBwdData(
        data, weight, out_grad, ipFwd_pd);
    auto out_grad_mem = out_grad.GetMKLDNNDataReorder(
        ipBwdData_pd.diff_dst_primitive_desc());
    auto weight_mem = weight.GetMKLDNNDataReorder(ipBwdData_pd.weights_primitive_desc());
    auto in_grad_mem = CreateMKLDNNMem(in_grad[fullc::kData],
        ipBwdData_pd.diff_src_primitive_desc(), req[fullc::kData]);
    MKLDNNStream::Instance().RegisterPrim(mkldnn::inner_product_backward_data(
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
    auto in_grad_weight = CreateMKLDNNMem(in_grad[fullc::kWeight],
        ipBwdWeights_pd.diff_weights_primitive_desc(), req[fullc::kWeight]);
    mkldnn_output_t in_grad_bias;
    if (param.no_bias) {
      MKLDNNStream::Instance().RegisterPrim(mkldnn::inner_product_backward_weights(
            ipBwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second));
    } else {
      in_grad_bias = CreateMKLDNNMem(in_grad[fullc::kBias],
          ipBwdWeights_pd.diff_bias_primitive_desc(), req[fullc::kBias]);
      MKLDNNStream::Instance().RegisterPrim(mkldnn::inner_product_backward_weights(
            ipBwdWeights_pd, *data_mem, *out_grad_mem, *in_grad_weight.second,
            *in_grad_bias.second));
    }
    CommitOutput(in_grad[fullc::kWeight], in_grad_weight);
    CommitOutput(in_grad[fullc::kBias], in_grad_bias);
  }
  MKLDNNStream::Instance().Submit();
}

}
}
#endif  // MXNET_USE_MKLDNN == 1
