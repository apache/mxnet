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
 */

#if MXNET_USE_MKLDNN == 1

#include "../deconvolution-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNDeconv(const DeconvolutionParam &params,
                         const NDArray &input) {
  if (params.kernel.ndim() != 2) return false;
  return (input.dtype() == mshadow::kFloat32 || input.dtype() == mshadow::kBfloat16)
         && input.shape().ndim() == 4;
}

static inline mkldnn::memory::desc GetBiasDesc(mkldnn::memory::desc md) {
  mkldnn::memory::dims dims(1);
  // This is deconvolution on 4D data. The second dimension is the channel.
  dims[0] = md.data.dims[1];
  return mkldnn::memory::desc(
      dims, static_cast<mkldnn::memory::data_type>(md.data.data_type),
      mkldnn::memory::format_tag::any);
}

std::shared_ptr<mkldnn::convolution_forward::primitive_desc> GetDeconvBwd_(
    const mkldnn::memory::desc &data_md, const mkldnn::memory::desc &weights_md,
    bool has_bias, const mkldnn::memory::desc &out_md,
    const mkldnn::engine &engine, const mkldnn::memory::dims &strides,
    const mkldnn::memory::dims &padding, const mkldnn::memory::dims &dilates) {
  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // for computation compared with the actual tensor size. Currently, MKL-DNN
  // operators are still reusing those memory from memory planning and the
  // memory size may smaller than what MKL-DNN kernels require. So here we need
  // select suboptimal kernel for computation according to tensor sizes.
  if (!has_bias) {
    mkldnn::convolution_forward::desc desc(
        mkldnn::prop_kind::forward_training,
        mkldnn::algorithm::convolution_direct, out_md, weights_md, data_md,
        strides, dilates, padding, padding);
    auto deconv_pd =
        std::make_shared<mkldnn::convolution_forward::primitive_desc>(desc,
                                                                      engine);
    while (deconv_pd->dst_desc().get_size() != GetMemDescSize(data_md) ||
           deconv_pd->src_desc().get_size() != GetMemDescSize(out_md) ||
           deconv_pd->weights_desc().get_size() != GetMemDescSize(weights_md)) {
      CHECK(deconv_pd->next_impl()) << "No implementation";
    }
    return deconv_pd;
  } else {
    auto bias_md = GetBiasDesc(data_md);
    mkldnn::convolution_forward::desc desc(
        mkldnn::prop_kind::forward_training,
        mkldnn::algorithm::convolution_direct, out_md, weights_md, bias_md,
        data_md, strides, dilates, padding, padding);
    auto deconv_pd =
        std::make_shared<mkldnn::convolution_forward::primitive_desc>(desc,
                                                                      engine);
    while (deconv_pd->dst_desc().get_size() != GetMemDescSize(data_md) ||
           deconv_pd->src_desc().get_size() != GetMemDescSize(out_md) ||
           deconv_pd->weights_desc().get_size() != GetMemDescSize(weights_md)) {
      CHECK(deconv_pd->next_impl()) << "No implementation";
    }
    return deconv_pd;
  }
}

std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc>
GetDeconvFwdImpl(const DeconvolutionParam &param, const NDArray &data,
                 const NDArray &weights, bool has_bias, const NDArray &output) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  CHECK_GE(param.stride.ndim(), 2);
  CHECK_GE(param.pad.ndim(), 2);
  CHECK_GE(param.dilate.ndim(), 2);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];
  mkldnn::memory::dims dilate{0, 0};
  dilate[0] = param.dilate[0] - 1;
  dilate[1] = param.dilate[1] - 1;
  auto bwd_pd = GetDeconvBwd_(data_md, weight_md, has_bias, out_md, engine,
                              strides, padding, dilate);
  mkldnn::convolution_backward_data::desc desc(
      mkldnn::algorithm::convolution_direct, out_md, weight_md, data_md,
      strides, dilate, padding, padding);
  auto deconv_pd =
      std::make_shared<mkldnn::convolution_backward_data::primitive_desc>(
          desc, engine, *bwd_pd);
  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // for computation compared with the actual tensor size. Currently, MKL-DNN
  // operators are still reusing those memory from memory planning and the
  // memory size may smaller than what MKL-DNN kernels require. So here we need
  // select suboptimal kernel for computation according to tensor sizes.
  while (deconv_pd->diff_dst_desc().get_size() != GetMemDescSize(data_md) ||
         deconv_pd->diff_src_desc().get_size() != GetMemDescSize(out_md) ||
         deconv_pd->weights_desc().get_size() != GetMemDescSize(weight_md)) {
    CHECK(deconv_pd->next_impl()) << "No implementation";
  }
  return deconv_pd;
}

std::shared_ptr<mkldnn::convolution_forward::primitive_desc>
GetDeconvBwdDataImpl(const DeconvolutionParam &param, const NDArray &data,
                     const NDArray &weights, bool has_bias,
                     const NDArray &output) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  CHECK_GE(param.stride.ndim(), 2);
  CHECK_GE(param.pad.ndim(), 2);
  CHECK_GE(param.dilate.ndim(), 2);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];
  mkldnn::memory::dims dilate{0, 0};
  dilate[0] = param.dilate[0] - 1;
  dilate[1] = param.dilate[1] - 1;
  return GetDeconvBwd_(data_md, weight_md, has_bias, out_md, engine, strides,
                       padding, dilate);
}

std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc>
GetDeconvBwdWeightsImpl(
    const DeconvolutionParam &param, const NDArray &data,
    const NDArray &weights, bool has_bias, const NDArray &output,
    const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  CHECK_GE(param.stride.ndim(), 2);
  CHECK_GE(param.pad.ndim(), 2);
  CHECK_GE(param.dilate.ndim(), 2);
  mkldnn::memory::dims strides{0, 0};
  strides[0] = param.stride[0];
  strides[1] = param.stride[1];
  mkldnn::memory::dims padding{0, 0};
  padding[0] = param.pad[0];
  padding[1] = param.pad[1];
  mkldnn::memory::dims dilate{0, 0};
  dilate[0] = param.dilate[0] - 1;
  dilate[1] = param.dilate[1] - 1;

  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // for computation compared with the actual tensor size. Currently, MKL-DNN
  // operators are still reusing those memory from memory planning and the
  // memory size may smaller than what MKL-DNN kernels require. So here we need
  // select suboptimal kernel for computation according to tensor sizes.
  if (!has_bias) {
    mkldnn::convolution_backward_weights::desc desc(
        mkldnn::algorithm::convolution_direct, out_md, weight_md, data_md,
        strides, dilate, padding, padding);
    auto deconv_pd =
        std::make_shared<mkldnn::convolution_backward_weights::primitive_desc>(
            desc, engine, fwd_pd);
    while (deconv_pd->diff_dst_desc().get_size() != GetMemDescSize(data_md) ||
           deconv_pd->src_desc().get_size() != GetMemDescSize(out_md) ||
           deconv_pd->diff_weights_desc().get_size() !=
               GetMemDescSize(weight_md)) {
      CHECK(deconv_pd->next_impl()) << "No implementation";
    }
    return deconv_pd;
  } else {
    auto bias_md = GetBiasDesc(data_md);
    mkldnn::convolution_backward_weights::desc desc(
        mkldnn::algorithm::convolution_direct, out_md, weight_md, bias_md,
        data_md, strides, dilate, padding, padding);
    auto deconv_pd =
        std::make_shared<mkldnn::convolution_backward_weights::primitive_desc>(
            desc, engine, fwd_pd);
    while (deconv_pd->diff_dst_desc().get_size() != GetMemDescSize(data_md) ||
           deconv_pd->src_desc().get_size() != GetMemDescSize(out_md) ||
           deconv_pd->diff_weights_desc().get_size() !=
               GetMemDescSize(weight_md)) {
      CHECK(deconv_pd->next_impl()) << "No implementation";
    }
    return deconv_pd;
  }
}

class MKLDNNDeconvForward {
 public:
  MKLDNNDeconvForward(const DeconvolutionParam &param, const NDArray &data,
                      const NDArray &weights, bool has_bias,
                      const NDArray &output);
  const mkldnn::convolution_backward_data &GetFwd() const { return *fwd; }

  const mkldnn::convolution_backward_data::primitive_desc &GetPd() const {
    return *fwd_pd;
  }

 private:
  std::shared_ptr<mkldnn::convolution_backward_data> fwd;
  std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc> fwd_pd;
};  // class MKLDNNDeconvForward

MKLDNNDeconvForward::MKLDNNDeconvForward(const DeconvolutionParam &param,
                                         const NDArray &data,
                                         const NDArray &weights, bool has_bias,
                                         const NDArray &output)
    : fwd_pd(GetDeconvFwdImpl(param, data, weights, has_bias, output)) {
  fwd = std::make_shared<mkldnn::convolution_backward_data>(GetPd());
}

static void MKLDNNDeconvFwdBiasPostProcess(
    const DeconvolutionParam &param, const OpContext &ctx, const NDArray &bias,
    const std::vector<NDArray> &out_data) {
  // add bias, broadcast bias to dim 1: channel
  if (!param.no_bias) {
    // MKLDNN only supports float right now.
    typedef float DType;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 1, DType> b = bias.data().get<cpu, 1, DType>(s);
    // The output data is stored in a special MKLDNN format,
    // converts its format to the default format.
    // Unfortunately, MKLDNN doesn't support broadcast.
    auto out_data_def = out_data[deconv::kOut].Reorder2Default();
    Tensor<cpu, 4, DType> out_cpu = out_data_def.data().get<cpu, 4, DType>(s);
    out_cpu += mshadow::expr::broadcast<1>(b, out_cpu.shape_);
  }
}

MKLDNNDeconvForward &GetDeconvFwd(const nnvm::NodeAttrs &attrs,
                                  const NDArray &data, const NDArray &weights,
                                  const NDArray *bias, const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DeconvSignature, MKLDNNDeconvForward,
                                         OpHash>
      fwds;
#else
  static MX_THREAD_LOCAL
      std::unordered_map<DeconvSignature, MKLDNNDeconvForward, OpHash>
          fwds;
#endif
  const DeconvolutionParam &param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  DeconvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias) key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    bool has_bias = (bias != nullptr);
    auto fwd = MKLDNNDeconvForward(param, data, weights, has_bias, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNDeconvolutionForward(const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const DeconvolutionParam &param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  auto &data = in_data[deconv::kData];
  auto &weight = in_data[deconv::kWeight];
  const NDArray *bias = param.no_bias ? nullptr : &in_data[deconv::kBias];

  MKLDNNDeconvForward &fwd =
      GetDeconvFwd(attrs, data, weight, bias, out_data[deconv::kOut]);

  auto data_mem = data.GetMKLDNNDataReorder(fwd.GetPd().diff_dst_desc());
  const mkldnn::memory *weight_mem;
  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (weight.IsMKLDNNData())
      // This asks the engine to change the layout of the weight array after
      // it's used.
      weight.Reorder2DefaultAsync();
    weight_mem =
        GetWeights(weight, fwd.GetPd().weights_desc(), param.num_group);
  } else {
    // For inference, we want to reorder the weight array so we don't need to
    // reorder data every time.
    if (weight.IsDefaultData()) {
      // We also need to modify the layout on the original weight array. The
      // data conversion happens after the weight array is used.
      weight.MKLDNNDataReorderAsync(fwd.GetPd().weights_desc());
      weight_mem =
          GetWeights(weight, fwd.GetPd().weights_desc(), param.num_group);

    } else {
      weight_mem = weight.GetMKLDNNData();
      CHECK(weight_mem->get_desc() == fwd.GetPd().weights_desc());
    }
  }
  mkldnn_output_t out_mem;
  out_mem = CreateMKLDNNMem(out_data[deconv::kOut], fwd.GetPd().diff_src_desc(),
                            req[deconv::kOut]);

  mkldnn_args_map_t net_args;

  net_args.insert({MKLDNN_ARG_DIFF_DST, *data_mem});
  net_args.insert({MKLDNN_ARG_WEIGHTS, *weight_mem});
  net_args.insert({MKLDNN_ARG_DIFF_SRC, *out_mem.second});
  MKLDNNStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data[deconv::kOut], out_mem);
  MKLDNNStream::Get()->Submit();

  MKLDNNDeconvFwdBiasPostProcess(param, ctx, *bias, out_data);
}

class MKLDNNDeconvBackwardData {
  std::shared_ptr<mkldnn::convolution_forward> bwd;

 public:
  std::shared_ptr<mkldnn::convolution_forward::primitive_desc> bwd_pd;
  MKLDNNDeconvBackwardData(const DeconvolutionParam &param, const NDArray &data,
                           const NDArray &weights, const NDArray &output);

  const mkldnn::convolution_forward &GetBwd() const { return *bwd; }
  const mkldnn::convolution_forward::primitive_desc &GetDataPd() const {
    return *bwd_pd;
  }
};

MKLDNNDeconvBackwardData::MKLDNNDeconvBackwardData(
    const DeconvolutionParam &param, const NDArray &data,
    const NDArray &weights, const NDArray &output)
    : bwd_pd(GetDeconvBwdDataImpl(param, data, weights, false, output)) {
  bwd = std::make_shared<mkldnn::convolution_forward>(GetDataPd());
}

typedef ParamOpSign<DeconvolutionParam> MKLDNNDeconvSignature;

static inline MKLDNNDeconvBackwardData &GetDeconvBwdData(
    const DeconvolutionParam &param, const NDArray &data,
    const NDArray &weights, const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNDeconvSignature,
                                         MKLDNNDeconvBackwardData, OpHash>
      bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNDeconvSignature,
                                            MKLDNNDeconvBackwardData, OpHash>
      bwds;
#endif
  MKLDNNDeconvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto bwd = MKLDNNDeconvBackwardData(param, data, weights, output);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

class MKLDNNDeconvBackwardWeights {
  std::shared_ptr<mkldnn::convolution_backward_weights> bwd;

 public:
  std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc>
      bwd_data_pd;
  MKLDNNDeconvBackwardWeights(
      const DeconvolutionParam &param, const NDArray &data,
      const NDArray &weights, const NDArray &output,
      const mkldnn::convolution_forward::primitive_desc &bwd_data_pd);
  const mkldnn::convolution_backward_weights &GetBwd() const { return *bwd; }
  const mkldnn::convolution_backward_weights::primitive_desc &GetWeightsPd()
      const {
    return *bwd_data_pd;
  }
};

MKLDNNDeconvBackwardWeights::MKLDNNDeconvBackwardWeights(
    const DeconvolutionParam &param, const NDArray &data,
    const NDArray &weights, const NDArray &output,
    const mkldnn::convolution_forward::primitive_desc &bwd_data_pd)
    : bwd_data_pd(GetDeconvBwdWeightsImpl(param, data, weights, false, output,
                                          bwd_data_pd)) {
  bwd = std::make_shared<mkldnn::convolution_backward_weights>(GetWeightsPd());
}

static inline MKLDNNDeconvBackwardWeights &GetDeconvBwdWeights(
    const DeconvolutionParam &param, const NDArray &data,
    const NDArray &weights, const NDArray &output,
    const mkldnn::convolution_forward::primitive_desc &bwd_data_pd) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNDeconvSignature,
                                         MKLDNNDeconvBackwardWeights, OpHash>
      bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNDeconvSignature,
                                            MKLDNNDeconvBackwardWeights, OpHash>
      bwds;
#endif
  MKLDNNDeconvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto bwd =
        MKLDNNDeconvBackwardWeights(param, data, weights, output, bwd_data_pd);
    auto ins_ret = bwds.insert(
        std::pair<MKLDNNDeconvSignature, MKLDNNDeconvBackwardWeights>(key,
                                                                      bwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNDeconvolutionBackward(const nnvm::NodeAttrs &attrs,
                                 const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const DeconvolutionParam &param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  auto &data = inputs[deconv::kData + 1];
  auto &weight = inputs[deconv::kWeight + 1];
  auto &out_grad = inputs[deconv::kOut];

  CHECK_NE(req[deconv::kWeight], kWriteInplace)
      << "cannot write weight inplace";
  MKLDNNDeconvBackwardData &bwd_data =
      GetDeconvBwdData(param, data, weight, inputs[deconv::kOut]);
  auto out_grad_mem =
      out_grad.GetMKLDNNDataReorder(bwd_data.GetDataPd().src_desc());
  if (req[deconv::kData]) {
    auto weight_mem = GetWeights(weight, bwd_data.GetDataPd().weights_desc(),
                                 param.num_group);
    auto in_grad_mem =
        CreateMKLDNNMem(in_grad[deconv::kData], bwd_data.GetDataPd().dst_desc(),
                        req[deconv::kData]);
    mkldnn_args_map_t net_args = {{MKLDNN_ARG_SRC, *out_grad_mem},
                                  {MKLDNN_ARG_WEIGHTS, *weight_mem},
                                  {MKLDNN_ARG_DST, *in_grad_mem.second}};
    MKLDNNStream::Get()->RegisterPrimArgs(bwd_data.GetBwd(), net_args);
    CommitOutput(in_grad[deconv::kData], in_grad_mem);
  }
  if (req[deconv::kWeight]) {
    MKLDNNDeconvBackwardWeights &bwd_weights = GetDeconvBwdWeights(
        param, data, weight, inputs[deconv::kOut], bwd_data.GetDataPd());
    if (bwd_data.GetDataPd().src_desc() !=
        bwd_weights.GetWeightsPd().src_desc())
      out_grad_mem =
          out_grad.GetMKLDNNDataReorder(bwd_weights.GetWeightsPd().src_desc());
    auto data_mem =
        data.GetMKLDNNDataReorder(bwd_weights.GetWeightsPd().diff_dst_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(
        in_grad[deconv::kWeight],
        bwd_weights.GetWeightsPd().diff_weights_desc(), req[deconv::kWeight]);

    mkldnn_args_map_t net_args = {
        {MKLDNN_ARG_SRC, *out_grad_mem},
        {MKLDNN_ARG_DIFF_DST, *data_mem},
        {MKLDNN_ARG_DIFF_WEIGHTS, *in_grad_weight.second}};
    MKLDNNStream::Get()->RegisterPrimArgs(bwd_weights.GetBwd(), net_args);
    CommitOutput(in_grad[deconv::kWeight], in_grad_weight);
  }
  MKLDNNStream::Get()->Submit();

  if (!param.no_bias) {
    typedef float DType;
    Stream<cpu> *s = ctx.get_stream<cpu>();
    Tensor<cpu, 1, DType> gbias =
        in_grad[deconv::kBias].data().get<cpu, 1, DType>(s);

    NDArray temp = inputs[deconv::kOut];
    if (temp.IsMKLDNNData()) {
      temp = temp.Reorder2Default();
    }

    Tensor<cpu, 4, DType> grad = temp.data().get<cpu, 4, DType>(s);
    Assign(gbias, req[deconv::kBias],
           mshadow::expr::sumall_except_dim<1>(grad));
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
