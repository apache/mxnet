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

static inline mkldnn::memory::desc GetBiasDesc(mkldnn::memory::desc md) {
  mkldnn::memory::dims dims(1);
  // This is convolution on 4D data. The second dimension is the channel.
  dims[0] = md.data.dims[1];
  return mkldnn::memory::desc(dims,
      static_cast<mkldnn::memory::data_type>(md.data.data_type),
      mkldnn::memory::format::any);
}

static mkldnn::convolution_forward::primitive_desc GetDeconvBwd_(
    const mkldnn::memory::desc &data_md, const mkldnn::memory::desc &weights_md,
    bool has_bias, const mkldnn::memory::desc &out_md,
    const mkldnn::engine &engine, const mkldnn::memory::dims &strides,
    const mkldnn::memory::dims &padding, const mkldnn::memory::dims &dilates) {
  if (!has_bias) {
    mkldnn::convolution_forward::desc desc(mkldnn::prop_kind::forward_training,
        mkldnn::algorithm::convolution_direct, out_md, weights_md, data_md, strides,
        dilates, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  } else {
    auto bias_md = GetBiasDesc(data_md);
    mkldnn::convolution_forward::desc desc(mkldnn::prop_kind::forward_training,
        mkldnn::algorithm::convolution_direct, out_md, weights_md, bias_md,
        data_md, strides, dilates, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, engine);
  }
}

static mkldnn::convolution_backward_data::primitive_desc GetDeconvFwd(
    const DeconvolutionParam& param, const NDArray &data, const NDArray &weights,
    bool has_bias, const NDArray &output) {
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
  mkldnn::memory::dims dilate{0, 0};
  if (param.dilate.ndim() == 2) {
    dilate[0] = param.dilate[0] - 1;
    dilate[1] = param.dilate[1] - 1;
  }
  auto bwd_pd = GetDeconvBwd_(data_md, weight_md, has_bias, out_md, engine,
      strides, padding, dilate);
  mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct,
      out_md, weight_md, data_md, strides, dilate, padding, padding,
      mkldnn::padding_kind::zero);
  return mkldnn::convolution_backward_data::primitive_desc(desc, engine, bwd_pd);
}

static mkldnn::convolution_forward::primitive_desc GetDeconvBwdData(
    const DeconvolutionParam &param, const NDArray &data, const NDArray &weights,
    bool has_bias, const NDArray &output) {
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
  mkldnn::memory::dims dilate{0, 0};
  if (param.dilate.ndim() == 2) {
    dilate[0] = param.dilate[0] - 1;
    dilate[1] = param.dilate[1] - 1;
  }
  return GetDeconvBwd_(data_md, weight_md, has_bias, out_md, engine,
      strides, padding, dilate);
}

static mkldnn::convolution_backward_weights::primitive_desc GetDeconvBwdWeights(
    const DeconvolutionParam& param, const NDArray &data, const NDArray &weights,
    bool has_bias, const NDArray &output,
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
  mkldnn::memory::dims dilate{0, 0};
  if (param.dilate.ndim() == 2) {
    dilate[0] = param.dilate[0] - 1;
    dilate[1] = param.dilate[1] - 1;
  }
  if (!has_bias) {
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        out_md, weight_md, data_md, strides, dilate, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
  } else {
    auto bias_md = GetBiasDesc(data_md);
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
        out_md, weight_md, bias_md, data_md, strides, dilate, padding, padding,
        mkldnn::padding_kind::zero);
    return mkldnn::convolution_backward_weights::primitive_desc(desc, engine, fwd_pd);
  }
}

void MKLDNNDeconvolution_Forward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
    const std::vector<NDArray> &in_data, const std::vector<OpReqType> &req,
    const std::vector<NDArray> &out_data) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  mkldnn::convolution_backward_data::primitive_desc deconvFwd_pd = GetDeconvFwd(
      param, in_data[deconv::kData], in_data[deconv::kWeight], false,
      out_data[deconv::kOut]);
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
  // add bias, broadcast bias to dim 1: channel
  if (!param.no_bias) {
    // MKLDNN only supports float right now.
    typedef float DType;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 1, DType> bias = in_data[deconv::kBias].data().get<cpu, 1, DType>(s);
    // If the output data is stored in a special MKLDNN format, data()
    // automatically converts its format to the default format.
    // Unfortunately, MKLDNN doesn't support broadcast.
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
      param, inputs[deconv::kData + 1], inputs[deconv::kWeight + 1], false,
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
          inputs[deconv::kWeight + 1], false, inputs[deconv::kOut], bwdData_pd);
    auto out_grad_mem = inputs[deconv::kOut].GetMKLDNNDataReorder(
        bwdWeights_pd.src_primitive_desc());
    auto data_mem = inputs[deconv::kData + 1].GetMKLDNNDataReorder(
        bwdWeights_pd.diff_dst_primitive_desc());
    auto in_grad_weight = CreateMKLDNNMem(in_grad[deconv::kWeight],
        bwdWeights_pd.diff_weights_primitive_desc(), req[deconv::kWeight]);
    MKLDNNStream::Instance().RegisterPrim(mkldnn::convolution_backward_weights(
          bwdWeights_pd, *out_grad_mem, *data_mem, *in_grad_weight.second));
    CommitOutput(in_grad[deconv::kWeight], in_grad_weight);
  }
  MKLDNNStream::Instance().Submit();
  if (!param.no_bias) {
    typedef float DType;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 1, DType> gbias = in_grad[deconv::kBias].data().get<cpu, 1, DType>(s);
    // If there is bias, the out grad has already been converted to the default
    // format, so this shouldn't cause any performance issues.
    Tensor<cpu, 4, DType> grad = inputs[deconv::kOut].data().get<cpu, 4, DType>(s);
    Assign(gbias, req[deconv::kBias], mshadow::expr::sumall_except_dim<1>(grad));
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
