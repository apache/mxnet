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
 * \author Da Zheng, Rong Zhang (rong.a.zhang@intel.com)
*/

#if MXNET_USE_MKLDNN == 1

#include "../deconvolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

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

static mkldnn::convolution_backward_data::primitive_desc GetDeconvFwdImpl(
    const DeconvolutionParam& param, const NDArray &data, const NDArray &weights,
    bool has_bias, const NDArray &output) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  } else if (param.stride.ndim() == 1) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[0];
  } else {
    LOG(FATAL) << "Unsupported stride dim";
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.pad.ndim() == 1) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[0];
  } else {
    LOG(FATAL) << "Unsupported pad dim";
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
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  } else if (param.stride.ndim() == 1) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[0];
  } else {
    LOG(FATAL) << "Unsupported stride dim";
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.pad.ndim() == 1) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[0];
  } else {
    LOG(FATAL) << "Unsupported pad dim";
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
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  } else if (param.stride.ndim() == 1) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[0];
  } else {
    LOG(FATAL) << "Unsupported stride dim";
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.pad.ndim() == 1) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[0];
  } else {
    LOG(FATAL) << "Unsupported pad dim";
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

class MKLDNNDeconvForward {
  std::shared_ptr<mkldnn::convolution_backward_data> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;
  OutDataOp data_op;

 public:
  MKLDNNDeconvForward(const DeconvolutionParam& param,
                      const NDArray &data,
                      const NDArray &weights,
                      bool has_bias,
                      const NDArray &output);
  void SetDataHandle(const DeconvolutionParam& param,
                     const OpContext &ctx,
                     const std::vector<NDArray> &in_data,
                     const std::vector<OpReqType> &req,
                     const std::vector<NDArray> &out_data);

  void Execute(const std::vector<NDArray> &out_data);

 private:
  mkldnn::convolution_backward_data::primitive_desc fwd_pd;
};  // class MKLDNNDeconvForward

MKLDNNDeconvForward::MKLDNNDeconvForward(const DeconvolutionParam& param,
                                const NDArray &data,
                                const NDArray &weights,
                                bool has_bias,
                                const NDArray &output)
                                :fwd_pd(GetDeconvFwdImpl(param, data, weights, has_bias, output)) {
  this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          fwd_pd.diff_dst_primitive_desc()));
  this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          fwd_pd.weights_primitive_desc()));
  this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
          fwd_pd.diff_src_primitive_desc()));
  this->fwd = std::shared_ptr<mkldnn::convolution_backward_data>(
    new mkldnn::convolution_backward_data(fwd_pd,
                                          mkldnn::primitive::at(*this->data),
                                          mkldnn::primitive::at(*this->weight),
                                          *this->out));
}

void MKLDNNDeconvForward::SetDataHandle(const DeconvolutionParam& param,
                                        const OpContext &ctx,
                                        const std::vector<NDArray> &in_data,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<NDArray> &out_data) {
  auto data_mem = in_data[deconv::kData].GetMKLDNNDataReorder(
      fwd_pd.diff_dst_primitive_desc());
  const mkldnn::memory *weight_mem;
  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (in_data[deconv::kWeight].IsMKLDNNData())
      const_cast<NDArray &>(in_data[deconv::kWeight]).Reorder2Default();
    weight_mem = GetWeights(in_data[deconv::kWeight],
                            fwd_pd.weights_primitive_desc(),
                            param.num_group);
  } else {
    // For inference, we want to reorder the weight array so we don't need to
    // reorder data every time.
    const_cast<NDArray &>(in_data[deconv::kWeight]).MKLDNNDataReorder(
        fwd_pd.weights_primitive_desc());
    weight_mem = in_data[deconv::kWeight].GetMKLDNNData();
  }
  auto out_mem = CreateMKLDNNMem(out_data[deconv::kOut],
      fwd_pd.diff_src_primitive_desc(), req[deconv::kOut]);
  auto output = out_mem.second;
  this->data->set_data_handle(data_mem->get_data_handle());
  this->weight->set_data_handle(weight_mem->get_data_handle());
  this->out->set_data_handle(output->get_data_handle());
  this->data_op = out_mem.first;
}

void MKLDNNDeconvForward::Execute(const std::vector<NDArray> &out_data) {
  MKLDNNStream::Get()->RegisterPrim(*fwd);
  CommitOutput(out_data[deconv::kOut], mkldnn_output_t(this->data_op, this->out.get()));
  MKLDNNStream::Get()->Submit();
}

static void MKLDNNDeconvFwdBiasPostProcess(const DeconvolutionParam& param,
                                           const OpContext &ctx,
                                           const std::vector<NDArray> &in_data,
                                           const std::vector<NDArray> &out_data) {
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

typedef MKLDNNParamOpSign<DeconvolutionParam> MKLDNNDeconvSignature;

static inline MKLDNNDeconvForward &GetDeconvFwd(
    const nnvm::NodeAttrs& attrs, const NDArray &data,
    const NDArray &weights, const NDArray *bias,
    const NDArray &output) {
  static thread_local
        std::unordered_map<MKLDNNDeconvSignature, MKLDNNDeconvForward, MKLDNNOpHash> fwds;
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  MKLDNNDeconvSignature key(param);
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
    bool has_bias = (bias != nullptr);
    MKLDNNDeconvForward fwd(param, data, weights, has_bias, output);
    auto ins_ret = fwds.insert(
        std::pair<MKLDNNDeconvSignature, MKLDNNDeconvForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNDeconvolutionForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  MKLDNNDeconvForward &deconvFwd = GetDeconvFwd(
      attrs, in_data[deconv::kData], in_data[deconv::kWeight],
      param.no_bias ? nullptr : &in_data[deconv::kBias], out_data[deconv::kOut]);

  deconvFwd.SetDataHandle(param, ctx, in_data, req, out_data);

  deconvFwd.Execute(out_data);

  MKLDNNDeconvFwdBiasPostProcess(param, ctx, in_data, out_data);
}

void MKLDNNDeconvolutionBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  CHECK_NE(req[deconv::kWeight], kWriteInplace) << "cannot write weight inplace";
  mkldnn::convolution_forward::primitive_desc bwdData_pd = GetDeconvBwdData(
      param, inputs[deconv::kData + 1], inputs[deconv::kWeight + 1], false,
      inputs[deconv::kOut]);
  auto out_grad_mem = inputs[deconv::kOut].GetMKLDNNDataReorder(
      bwdData_pd.src_primitive_desc());
  if (req[deconv::kData]) {
    auto weight_mem = GetWeights(inputs[deconv::kWeight + 1],
                                 bwdData_pd.weights_primitive_desc(),
                                 param.num_group);
    auto in_grad_mem = CreateMKLDNNMem(in_grad[deconv::kData],
                                       bwdData_pd.dst_primitive_desc(),
                                       req[deconv::kData]);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::convolution_forward(bwdData_pd,
          *out_grad_mem, *weight_mem, *in_grad_mem.second));
    CommitOutput(in_grad[deconv::kData], in_grad_mem);
  }
  if (req[deconv::kWeight]) {
    mkldnn::convolution_backward_weights::primitive_desc bwdWeights_pd
      = GetDeconvBwdWeights(param, inputs[deconv::kData + 1],
          inputs[deconv::kWeight + 1], false, inputs[deconv::kOut], bwdData_pd);
    if (bwdData_pd.src_primitive_desc() != bwdWeights_pd.src_primitive_desc())
      out_grad_mem = inputs[deconv::kOut].GetMKLDNNDataReorder(
          bwdWeights_pd.src_primitive_desc());
    auto data_mem = inputs[deconv::kData + 1].GetMKLDNNDataReorder(
        bwdWeights_pd.diff_dst_primitive_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(in_grad[deconv::kWeight],
                                                 bwdWeights_pd.diff_weights_primitive_desc(),
                                                 req[deconv::kWeight]);
    MKLDNNStream::Get()->RegisterPrim(mkldnn::convolution_backward_weights(
          bwdWeights_pd, *out_grad_mem, *data_mem, *in_grad_weight.second));
    CommitOutput(in_grad[deconv::kWeight], in_grad_weight);
  }
  MKLDNNStream::Get()->Submit();
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
