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
 * \file mkldnn_deconvolution.cc
 * \brief
 * \author Da Zheng
*/

#include "../deconvolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

static mkldnn::convolution_forward::primitive_desc GetDeconvBwd_(
    const mkldnn::memory::desc &data_md, const mkldnn::memory::desc &weights_md,
    const mkldnn::memory::desc *bias_md, const mkldnn::memory::desc &out_md,
    const mkldnn::engine &engine, const mkldnn::memory::dims &strides,
    const mkldnn::memory::dims &padding) {
  // TODO when dilate > 1
  if (bias_md == nullptr) {
    mkldnn::convolution_forward::desc desc(mkldnn::prop_kind::forward_training,
        mkldnn::algorithm::convolution_direct, out_md, weights_md, data_md, strides,
        padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  }
  else {
    mkldnn::convolution_forward::desc desc(mkldnn::prop_kind::forward_training,
        mkldnn::algorithm::convolution_direct, out_md, weights_md,
        *bias_md, data_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  }
}

static mkldnn::convolution_backward_data::primitive_desc GetDeconvFwd(
    const DeconvolutionParam& param, const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output) {
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
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    auto bwd_pd = GetDeconvBwd_(data_md, weight_md, &bias_md,
        out_md, engine, strides, padding);
    // TODO when dilate > 1
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        out_md, weight_md, data_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(desc, engine, bwd_pd);
  }
  else {
    auto bwd_pd = GetDeconvBwd_(data_md, weight_md, nullptr, out_md, engine,
        strides, padding);
    // TODO when dilate > 1
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
        out_md, weight_md, data_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_data::primitive_desc(desc, engine, bwd_pd);
  }
}

static mkldnn::convolution_forward::primitive_desc GetDeconvBwdData(
    const DeconvolutionParam &param, const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output) {
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
  // TODO dilate
  if (bias) {
    auto bias_md = GetMemDesc(*bias);
    return GetDeconvBwd_(data_md, weight_md, &bias_md, out_md,
        engine, strides, padding);
  }
  else
    return GetDeconvBwd_(data_md, weight_md, nullptr, out_md,
        engine, strides, padding);
}

static mkldnn::convolution_backward_weights::primitive_desc GetDeconvBwdWeights(
    const DeconvolutionParam& param, const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output,
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
        out_md, weight_md, data_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
  }
  else /*if (param.dilate.ndim() == 0)*/ {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        out_md, weight_md, bias_md, data_md, strides, padding, padding,
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

void MKLDNNDeconvolution_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  mkldnn::convolution_backward_data::primitive_desc deconvFwd_pd = GetDeconvFwd(
      param, in_data[deconv::kData], in_data[deconv::kWeight],
      param.no_bias ? nullptr : &in_data[deconv::kBias], out_data[deconv::kOut]);
  auto data_mem = in_data[deconv::kData].GetMKLDNNDataReorder(
      deconvFwd_pd.diff_dst_primitive_desc());
  auto weight_mem = GetWeights(in_data[deconv::kWeight],
      deconvFwd_pd.weights_primitive_desc(), param.num_group);
  auto out_mem = CreateMKLDNNMem(out_data[deconv::kOut],
      deconvFwd_pd.diff_src_primitive_desc(), req[deconv::kOut]);

  MKLDNNStream::Instance().RegisterPrim(mkldnn::convolution_backward_data(
        deconvFwd_pd, *data_mem, *weight_mem, *out_mem.second));
  CommitOutput(out_data[deconv::kOut], out_mem);
  MKLDNNStream::Instance().Submit();
  if (!param.no_bias) {
    // add bias, broadcast bias to dim 1: channel
    // TODO this is problematic if the layout isn't expected.
    // we need to handle the type correctly.
    typedef float DType;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 1, DType> bias = in_data[deconv::kBias].data().get<cpu, 1, DType>(s);
    Tensor<cpu, 4, DType> out_cpu = out_data[deconv::kOut].data().get<cpu, 4, DType>(s);
    out_cpu += mshadow::expr::broadcast<1>(bias, out_cpu.shape_);
  }
}

void MKLDNNDeconvolution_Backward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray>& inputs, const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  const std::vector<NDArray> &in_grad = outputs;
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  CHECK_NE(req[deconv::kWeight], kWriteInplace) << "cannot write weight inplace";
  mkldnn::convolution_forward::primitive_desc bwdData_pd = GetDeconvBwdData(
      param, inputs[deconv::kData + 1], inputs[deconv::kWeight + 1], nullptr,
      inputs[deconv::kOut]);
  if (req[deconv::kData]) {
    auto out_grad_mem = inputs[deconv::kOut].GetMKLDNNDataReorder(
        bwdData_pd.src_primitive_desc());
    auto weight_mem = GetWeights(inputs[deconv::kWeight + 1],
        bwdData_pd.weights_primitive_desc(), param.num_group);
    auto in_grad_mem = CreateMKLDNNMem(in_grad[deconv::kData],
        bwdData_pd.dst_primitive_desc(), req[deconv::kData]);
    MKLDNNStream::Instance().RegisterPrim(mkldnn::convolution_forward(bwdData_pd,
          *out_grad_mem, *weight_mem, *in_grad_mem.second));
    CommitOutput(in_grad[deconv::kData], in_grad_mem);
  }
  if (req[deconv::kWeight]) {
    mkldnn::convolution_backward_weights::primitive_desc bwdWeights_pd
      = GetDeconvBwdWeights(param, inputs[deconv::kData + 1],
          inputs[deconv::kWeight + 1],
          param.no_bias ? nullptr : &inputs[deconv::kWeight + 1],
          inputs[deconv::kOut], bwdData_pd);
    auto out_grad_mem = inputs[deconv::kOut].GetMKLDNNDataReorder(
        bwdWeights_pd.src_primitive_desc());
    auto data_mem = inputs[deconv::kData + 1].GetMKLDNNDataReorder(
        bwdWeights_pd.diff_dst_primitive_desc());
    auto in_grad_weight = CreateMKLDNNMem(in_grad[deconv::kWeight],
        bwdWeights_pd.diff_weights_primitive_desc(), req[deconv::kWeight]);
    mkldnn_output_t in_grad_bias;
    if (param.no_bias) {
      MKLDNNStream::Instance().RegisterPrim(mkldnn::convolution_backward_weights(
            bwdWeights_pd, *out_grad_mem, *data_mem, *in_grad_weight.second));
    } else {
      in_grad_bias = CreateMKLDNNMem(in_grad[deconv::kBias],
          bwdWeights_pd.diff_bias_primitive_desc(), req[deconv::kBias]);
      MKLDNNStream::Instance().RegisterPrim(mkldnn::convolution_backward_weights(
            bwdWeights_pd, *out_grad_mem, *data_mem, *in_grad_weight.second,
            *in_grad_bias.second));
    }
    CommitOutput(in_grad[deconv::kWeight], in_grad_weight);
    CommitOutput(in_grad[deconv::kBias], in_grad_bias);
  }
  MKLDNNStream::Instance().Submit();
}

}
}

#endif // MXNET_USE_MKLDNN == 1
