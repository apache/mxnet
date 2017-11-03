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
    const ConvolutionParam& param, bool is_train, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Instance().get_engine();
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
  if (/*param.dilate.ndim() == 0 &&*/ bias == nullptr) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  }
  else /*if (param.dilate.ndim() == 0)*/ {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, bias_md, out_md, strides, padding, padding,
        mkldnn::padding_kind::zero);
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
    const ConvolutionParam& param, const NDArray &data, const NDArray &weights,
    const NDArray &output, const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Instance().get_engine();
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
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
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
    const ConvolutionParam& param, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output,
    const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Instance().get_engine();
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
  if (/*param.dilate.ndim() == 0 &&*/ bias == nullptr) {
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
  }
  else /*if (param.dilate.ndim() == 0)*/ {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        data_md, weight_md, bias_md, out_md, strides, padding, padding,
        mkldnn::padding_kind::zero);
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
  mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwd(param,
      ctx.is_train, in_data[conv::kData], in_data[conv::kWeight],
      param.no_bias ? nullptr : &in_data[conv::kBias], out_data[conv::kOut]);
  std::vector<mkldnn::primitive> net;
  printf("src layout: %d\n", fwd_pd.src_primitive_desc().desc().data.format);
  printf("weight layout: %d\n", fwd_pd.weights_primitive_desc().desc().data.format);
  printf("out layout: %d\n", fwd_pd.dst_primitive_desc().desc().data.format);
  auto data_mem = in_data[conv::kData].GetMKLDNNData(fwd_pd.src_primitive_desc(), net);
  auto engine = CpuEngine::Instance().get_engine();
  auto weight_data = GetWeights(in_data[conv::kWeight],
      fwd_pd.weights_primitive_desc(), param.num_group, net);
  auto weight_mem = weight_data.first;

  auto out_mem = const_cast<NDArray &>(out_data[conv::kOut]).CreateMKLDNNData(
      fwd_pd.dst_primitive_desc());

  if (param.no_bias) {
    net.push_back(mkldnn::convolution_forward(fwd_pd, *data_mem, *weight_mem,
          *out_mem));
  } else {
    auto bias_mem = in_data[conv::kBias].GetMKLDNNData(fwd_pd.bias_primitive_desc(), net);
    net.push_back(mkldnn::convolution_forward(fwd_pd, *data_mem, *weight_mem,
          *bias_mem, *out_mem));
  }
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

void MKLDNNConvolution_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  const std::vector<NDArray> &in_grad = outputs;
  auto engine = CpuEngine::Instance().get_engine();
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  mkldnn::convolution_forward::primitive_desc fwd_pd = GetConvFwd(param, ctx.is_train,
      inputs[conv::kData + 1], inputs[conv::kWeight + 1],
      param.no_bias ? nullptr : &inputs[conv::kBias + 1], inputs[conv::kOut]);

  CHECK_NE(req[conv::kWeight], kWriteInplace) << "cannot write weight inplace";
  std::vector<mkldnn::primitive> net;
  std::shared_ptr<mkldnn::memory> in_grad_mem, in_grad_weight, in_grad_bias;
  std::pair<mkldnn_mem_const_ptr, mkldnn_mem_const_ptr> weight_data;
  if (req[conv::kData]) {
    mkldnn::convolution_backward_data::primitive_desc bwdData_pd
      = GetConvBwdData(param, inputs[conv::kData + 1], inputs[conv::kWeight + 1],
          inputs[conv::kOut], fwd_pd);
    auto out_grad_mem = inputs[conv::kOut].GetMKLDNNData(
        bwdData_pd.diff_dst_primitive_desc(), net);
    weight_data = GetWeights(inputs[conv::kWeight + 1],
        bwdData_pd.weights_primitive_desc(), param.num_group, net);
    auto weight_mem = weight_data.first;
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
      = GetConvBwdWeights(param, inputs[conv::kData + 1], inputs[conv::kWeight + 1], 
          param.no_bias ? nullptr : &inputs[conv::kBias + 1], inputs[conv::kOut], fwd_pd);
    auto out_grad_mem = inputs[conv::kOut].GetMKLDNNData(
        bwdWeights_pd.diff_dst_primitive_desc(), net);
    auto data_mem = inputs[conv::kData + 1].GetMKLDNNData(
        bwdWeights_pd.src_primitive_desc(), net);
    in_grad_weight = const_cast<NDArray &>(in_grad[conv::kWeight]).CreateMKLDNNData(
        bwdWeights_pd.diff_weights_primitive_desc());
    bool copy_back_weight = false;
    bool copy_back_bias = false;
    if (in_grad_weight == nullptr) {
      in_grad_weight = CreateMKLDNNMem(bwdWeights_pd.diff_weights_primitive_desc());
      copy_back_weight = true;
    }
    if (param.no_bias) {
      net.push_back(mkldnn::convolution_backward_weights(bwdWeights_pd,
            *data_mem, *out_grad_mem, *in_grad_weight));
    } else {
      in_grad_bias = const_cast<NDArray &>(in_grad[conv::kBias]).CreateMKLDNNData(
          bwdWeights_pd.diff_bias_primitive_desc());
      if (in_grad_bias == nullptr) {
        in_grad_bias = CreateMKLDNNMem(bwdWeights_pd.diff_bias_primitive_desc());
        copy_back_bias = true;
      }
      net.push_back(mkldnn::convolution_backward_weights(bwdWeights_pd,
            *data_mem, *out_grad_mem, *in_grad_weight, *in_grad_bias));
    }
    if (copy_back_weight)
      const_cast<NDArray &>(in_grad[conv::kWeight]).CopyFrom(*in_grad_weight, net);
    if (copy_back_bias)
      const_cast<NDArray &>(in_grad[conv::kBias]).CopyFrom(*in_grad_bias, net);
  }
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
}

}
}

#endif // MXNET_USE_MKLDNN == 1
