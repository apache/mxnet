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

static mkldnn::convolution_forward::primitive_desc GetConvFwd(
    const ConvolutionParam& param, bool is_train,
    const mkldnn::memory::desc &data_md, const mkldnn::memory::desc &weights_md,
    const mkldnn::memory::desc &out_md, const mkldnn::engine &engine,
    std::shared_ptr<const mkldnn::memory> bias_mem) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  }
  if (/*param.dilate.ndim() == 0 &&*/ bias_mem == nullptr) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weights_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  }
  else /*if (param.dilate.ndim() == 0)*/ {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weights_md, bias_mem->get_primitive_desc().desc(), out_md,
        strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  }
//  else {
//    // TODO I should test the case with dilate.
//    mkldnn::memory::dims dilates{0, 0};
//    if (param.dilate.ndim() == 2) {
//      dilates[0] = param.dilate[0];
//      dilates[1] = param.dilate[1];
//    }
//    if (bias_mem == nullptr) {
//      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
//          data_md, weights_md, out_md, strides, dilates, padding, padding,
//          mkldnn::padding_kind::zero);
//      return mkldnn::convolution_forward::primitive_desc(desc, engine);
//    }
//    else {
//      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
//          data_md, weights_md, bias_mem->get_primitive_desc().desc(), out_md,
//          strides, dilates, padding, padding, mkldnn::padding_kind::zero);
//      return mkldnn::convolution_forward::primitive_desc(desc, engine);
//    }
//  }
}

static mkldnn::convolution_backward_data::primitive_desc GetConvBwdData(
    const ConvolutionParam& param, const mkldnn::memory::desc &data_md,
    const mkldnn::memory::desc &weights_md, const mkldnn::memory::desc &out_md,
    const mkldnn::engine &engine,
    const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  }
//  if (param.dilate.ndim() == 0) {
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weights_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(desc, engine, fwd_pd);
//  }
//  else {
//    // TODO I should test the case with dilate.
//    mkldnn::memory::dims dilates{0, 0};
//    if (param.dilate.ndim() == 2) {
//      dilates[0] = param.dilate[0];
//      dilates[1] = param.dilate[1];
//    }
//    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
//        data_md, weights_md, out_md, strides, dilates, padding, padding,
//        mkldnn::padding_kind::zero);
//    return mkldnn::convolution_backward_data::primitive_desc(desc, engine, fwd_pd);
//  }
}

static mkldnn::convolution_backward_weights::primitive_desc GetConvBwdWeights(
    const ConvolutionParam& param, const mkldnn::memory::desc &data_md,
    const mkldnn::memory::desc &weights_md, const mkldnn::memory::desc &out_md,
    const mkldnn::engine &engine, const mkldnn::convolution_forward::primitive_desc &fwd_pd,
    std::shared_ptr<const mkldnn::memory> bias_mem) {
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  }
  if (/*param.dilate.ndim() == 0 &&*/ bias_mem == nullptr) {
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weights_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
  }
  else /*if (param.dilate.ndim() == 0)*/ {
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weights_md, bias_mem->get_primitive_desc().desc(), out_md,
        strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
  }
//  else {
//    // TODO I should test the case with dilate.
//    mkldnn::memory::dims dilates{0, 0};
//    if (param.dilate.ndim() == 2) {
//      dilates[0] = param.dilate[0];
//      dilates[1] = param.dilate[1];
//    }
//    if (bias_mem == nullptr) {
//      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
//          data_md, weights_md, out_md, strides, dilates, padding, padding,
//          mkldnn::padding_kind::zero);
//      return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
//    }
//    else {
//      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
//          data_md, weights_md, bias_mem->get_primitive_desc().desc(), out_md,
//          strides, dilates, padding, padding, mkldnn::padding_kind::zero);
//      return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
//    }
//  }
}

void MKLDNNConvolution_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  auto data_mem = in_data[conv::kData].GetMKLDNNData();
  auto data_desc = data_mem->get_primitive_desc().desc();
  auto cpu_engine = data_mem->get_primitive_desc().get_engine();
  auto weight_mem = GetWeights(in_data[conv::kWeight], cpu_engine, param.num_group);
  auto weight_desc = weight_mem->get_primitive_desc().desc();
  auto out_mem = const_cast<NDArray &>(out_data[conv::kOut]).GetMKLDNNData();
  auto out_desc = out_mem->get_primitive_desc().desc();

  std::vector<mkldnn::primitive> net;
  if (param.no_bias) {
    mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwd(param,
        ctx.is_train, data_desc, weight_desc, out_desc, cpu_engine, nullptr);
    CHECK(fwd_pd.src_primitive_desc() == data_mem->get_primitive_desc());
    CHECK(fwd_pd.weights_primitive_desc() == weight_mem->get_primitive_desc());
    CHECK(fwd_pd.dst_primitive_desc() == out_mem->get_primitive_desc());
    net.push_back(mkldnn::convolution_forward(fwd_pd, *data_mem, *weight_mem,
          *out_mem));
  } else {
    auto bias_mem = in_data[conv::kBias].GetMKLDNNData();
    mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwd(param,
        ctx.is_train, data_desc, weight_desc, out_desc, cpu_engine, bias_mem);
    CHECK(fwd_pd.src_primitive_desc() == data_mem->get_primitive_desc());
    CHECK(fwd_pd.weights_primitive_desc() == weight_mem->get_primitive_desc());
    CHECK(fwd_pd.bias_primitive_desc() == bias_mem->get_primitive_desc());
    CHECK(fwd_pd.dst_primitive_desc() == out_mem->get_primitive_desc());
    net.push_back(mkldnn::convolution_forward(fwd_pd, *data_mem, *weight_mem,
          *bias_mem, *out_mem));
  }
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

void MKLDNNConvolution_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  const std::vector<NDArray> &in_grad = outputs;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  auto out_grad_mem = inputs[conv::kOut].GetMKLDNNData();
  auto out_grad_desc = out_grad_mem->get_primitive_desc().desc();
  auto data_mem = inputs[conv::kData + 1].GetMKLDNNData();
  auto data_desc = data_mem->get_primitive_desc().desc();
  auto cpu_engine = data_mem->get_primitive_desc().get_engine();
  auto weight_mem = GetWeights(inputs[conv::kWeight + 1], cpu_engine,
      param.num_group);
  auto weight_desc = weight_mem->get_primitive_desc().desc();
  std::shared_ptr<const mkldnn::memory> in_grad_bias;
  if (!param.no_bias)
    in_grad_bias = const_cast<NDArray &>(in_grad[conv::kBias]).GetMKLDNNData();
  mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwd(param, ctx.is_train,
      data_desc, weight_desc, out_grad_desc, cpu_engine, in_grad_bias);

  CHECK_NE(req[conv::kWeight], kWriteInplace) << "cannot write weight inplace";
  std::vector<mkldnn::primitive> net;
  std::shared_ptr<mkldnn::memory> in_grad_mem, in_grad_weight;
  if (req[conv::kData]) {
    mkldnn::convolution_backward_data::primitive_desc bwdData_pd
      = GetConvBwdData(param, data_desc, weight_desc, out_grad_desc, cpu_engine, fwd_pd);
    CHECK(bwdData_pd.diff_dst_primitive_desc() == out_grad_mem->get_primitive_desc());
    CHECK(bwdData_pd.weights_primitive_desc() == weight_mem->get_primitive_desc());

    in_grad_mem = const_cast<NDArray &>(in_grad[conv::kData]).CreateMKLDNNData(
        bwdData_pd.diff_src_primitive_desc());
    bool copy_back = false;
    if (in_grad_mem == nullptr) {
      in_grad_mem = CreateMKLDNNMem(bwdData_pd.diff_src_primitive_desc());
      copy_back = true;
    }
    net.push_back(mkldnn::convolution_backward_data(bwdData_pd, *out_grad_mem,
          *weight_mem, *in_grad_mem));
    if (copy_back)
      const_cast<NDArray &>(in_grad[conv::kData]).CopyFrom(*in_grad_mem, net);
  }
  if (req[conv::kWeight]) {
    mkldnn::convolution_backward_weights::primitive_desc bwdWeights_pd
      = GetConvBwdWeights(param, data_desc, weight_desc, out_grad_desc,
          cpu_engine, fwd_pd, in_grad_bias);
    CHECK(bwdWeights_pd.diff_dst_primitive_desc() == out_grad_mem->get_primitive_desc());
    CHECK(bwdWeights_pd.src_primitive_desc() == data_mem->get_primitive_desc());
    in_grad_weight = const_cast<NDArray &>(in_grad[conv::kWeight]).CreateMKLDNNData(
        bwdWeights_pd.diff_weights_primitive_desc());
    bool copy_back = false;
    if (in_grad_weight == nullptr) {
      in_grad_weight = CreateMKLDNNMem(bwdWeights_pd.diff_weights_primitive_desc());
      copy_back = true;
    }
    if (param.no_bias) {
      net.push_back(mkldnn::convolution_backward_weights(bwdWeights_pd,
            *data_mem, *out_grad_mem, *in_grad_weight));
    } else {
      net.push_back(mkldnn::convolution_backward_weights(bwdWeights_pd,
            *data_mem, *out_grad_mem, *in_grad_weight, *in_grad_bias));
    }
    if (copy_back) {
      const_cast<NDArray &>(in_grad[conv::kWeight]).CopyFrom(*in_grad_weight, net);
    }
  }
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

}
}

#endif // MXNET_USE_MKLDNN == 1
