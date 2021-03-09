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

using DeconvFwd = mkldnn::deconvolution_forward;
using DeconvFwdPD = mkldnn::deconvolution_forward::primitive_desc;

using DeconvBwdData = mkldnn::deconvolution_backward_data;
using DeconvBwdDataPD = mkldnn::deconvolution_backward_data::primitive_desc;

using DeconvBwdWeight = mkldnn::deconvolution_backward_weights;
using DeconvBwdWeightPD = mkldnn::deconvolution_backward_weights::primitive_desc;

bool SupportMKLDNNDeconv(const DeconvolutionParam &params, const NDArray &input) {
  if (params.kernel.ndim() != 2) return false;
  return (input.dtype() == mshadow::kFloat32 || input.dtype() == mshadow::kBfloat16) &&
         input.shape().ndim() == 4;
}

// Swaps the logical order of dimensions that in plain format would correspond to input and output
// channels (for example: oihw => iohw, iohw => oihw, goihw => giohw).
static inline mkldnn::memory::desc IOLogicalSwapDesc(mkldnn::memory::desc desc, int num_groups) {
  auto &d = desc.data;
  int offset = int(num_groups > 1);
  int dim0 = offset + 0;
  int dim1 = offset + 1;
  std::swap(d.dims[dim0], d.dims[dim1]);
  std::swap(d.padded_dims[dim0], d.padded_dims[dim1]);
  if (d.format_kind != dnnl_format_kind_any) {
    std::swap(d.format_desc.blocking.strides[dim0], d.format_desc.blocking.strides[dim1]);
    // as padding is not supported, these are always zeros?
    std::swap(d.padded_offsets[dim0], d.padded_offsets[dim1]);
    // for blocked format: change indices
    for (int i = 0; i < d.format_desc.blocking.inner_nblks; ++i) {
      auto &val = d.format_desc.blocking.inner_idxs[i];
      if (val == dim0) {
        val = dim1;
      } else if (val == dim1) {
        val = dim0;
      }
    }
  }
  return desc;
}

// Applies IOLogicalSwapDesc to arr
static inline void IOLogicalSwapMKLDNNMem(const NDArray &arr, int num_groups) {
  mkldnn::memory::desc desc;
  if (arr.IsMKLDNNData()) {
    desc = arr.GetMKLDNNData()->get_desc();
  } else {
    const auto &temp = GetWeightDesc(arr, num_groups);
    desc = mkldnn::memory::desc(
        temp.dims(), temp.data_type(),
        static_cast<mkldnn::memory::format_tag>(GetDefaultFormat(temp.data.ndims)));
  }
  const_cast<NDArray &>(arr).UpdateMKLDNNMemDesc(IOLogicalSwapDesc(desc, num_groups));
}

// Version of GetWeightDesc for deconvolution (with swap)
static inline mkldnn::memory::desc GetDeconvWeightDesc(const NDArray &weights, int num_groups) {
  return IOLogicalSwapDesc(GetWeightDesc(weights, num_groups), num_groups);
}

// Imposes the plain format on memory descriptors with padding
// Changing only one at a time, so maybe better implementations will be selected
// (than entirely plain one)
void ImposePlainWherePadding(mkldnn::memory::desc &src_md, mkldnn::memory::desc &dst_md,
                             mkldnn::memory::desc &weight_md, size_t src_size, size_t dst_size,
                             size_t wei_size) {
  if (src_size != GetMemDescSize(src_md)) {
    CHECK(src_md.data.format_kind == dnnl_format_kind_any) << "No implementation";
    src_md = GetDesc(src_md, GetDefaultFormat(src_md));
  } else if (dst_size != GetMemDescSize(dst_md)) {
    CHECK(dst_md.data.format_kind == dnnl_format_kind_any) << "No implementation";
    dst_md = GetDesc(dst_md, GetDefaultFormat(dst_md));
  } else if (wei_size != GetMemDescSize(weight_md)) {
    CHECK(weight_md.data.format_kind == dnnl_format_kind_any) << "No implementation";
    int num_groups = (weight_md.data.ndims > src_md.data.ndims) ? weight_md.data.dims[0] : 1;
    weight_md = IOLogicalSwapDesc(weight_md, num_groups);
    weight_md = IOLogicalSwapDesc(GetDesc(weight_md, GetDefaultFormat(weight_md)), num_groups);
  }
}

std::shared_ptr<DeconvFwdPD> GetDeconvFwdImpl(const DeconvolutionParam &param, const NDArray &data,
                                              const NDArray &weights, const NDArray *bias,
                                              const NDArray &output) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetDeconvWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto bias_md = bias ? GetMemDesc(*bias)
                      : mkldnn::memory::desc{
                            {}, mkldnn::memory::data_type::undef, mkldnn::memory::format_tag::any};
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
  auto desc = [&]() {
    return DeconvFwd::desc(
        mkldnn::prop_kind::forward_training,  // TODO: check if this should be constant
        mkldnn::algorithm::deconvolution_direct, data_md, weight_md, bias_md, out_md, strides,
        dilate, padding, padding);
  };
  auto deconv_pd = std::make_shared<DeconvFwdPD>(desc(), engine);
  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // compared to the actual size of the tensor. Currently, MKL-DNN operators
  // still reuse memory from memory planning, so here we need to select a
  // suboptimal kernel for computation that has the expected memory size requirements
  while (deconv_pd->dst_desc().get_size() != GetMemDescSize(out_md) ||
         deconv_pd->src_desc().get_size() != GetMemDescSize(data_md) ||
         deconv_pd->weights_desc().get_size() != GetMemDescSize(weight_md)) {
    // for deconvolution primitive next_impl always fails. Keep this?
    if (!deconv_pd->next_impl()) {
      ImposePlainWherePadding(data_md, out_md, weight_md, deconv_pd->dst_desc().get_size(),
                              deconv_pd->src_desc().get_size(),
                              deconv_pd->weights_desc().get_size());
      *deconv_pd = DeconvFwdPD(desc(), engine);
    }
  }

  return deconv_pd;
}

std::shared_ptr<DeconvBwdDataPD> GetDeconvBwdDataImpl(const DeconvolutionParam &param,
                                                      const NDArray &data, const NDArray &weights,
                                                      const NDArray &output,
                                                      const DeconvFwdPD &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetDeconvWeightDesc(weights, param.num_group);
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
  auto desc = [&]() {
    return DeconvBwdData::desc(mkldnn::algorithm::deconvolution_direct, data_md, weight_md, out_md,
                               strides, dilate, padding, padding);
  };
  auto deconv_pd = std::make_shared<DeconvBwdDataPD>(desc(), engine, fwd_pd);
  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // compared to the actual size of the tensor. Currently, MKL-DNN operators
  // still reuse memory from memory planning, so here we need to select a
  // suboptimal kernel for computation that has the expected memory size requirements
  while (deconv_pd->diff_dst_desc().get_size() != GetMemDescSize(out_md) ||
         deconv_pd->diff_src_desc().get_size() != GetMemDescSize(data_md) ||
         deconv_pd->weights_desc().get_size() != GetMemDescSize(weight_md)) {
    if (!deconv_pd->next_impl()) {
      ImposePlainWherePadding(data_md, out_md, weight_md, deconv_pd->diff_dst_desc().get_size(),
                              deconv_pd->diff_src_desc().get_size(),
                              deconv_pd->weights_desc().get_size());
      *deconv_pd = DeconvBwdDataPD(desc(), engine, fwd_pd);
    }
  }
  return deconv_pd;
}

std::shared_ptr<DeconvBwdWeightPD> GetDeconvBwdWeightsImpl(
    const DeconvolutionParam &param, const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output, const DeconvFwdPD &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetDeconvWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto bias_md = bias ? GetMemDesc(*bias)
                      : mkldnn::memory::desc{
                            {}, mkldnn::memory::data_type::undef, mkldnn::memory::format_tag::any};
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
  auto desc = [&]() {
    return DeconvBwdWeight::desc(mkldnn::algorithm::deconvolution_direct, data_md, weight_md,
                                 bias_md, out_md, strides, dilate, padding, padding);
  };
  auto deconv_pd = std::make_shared<DeconvBwdWeightPD>(desc(), engine, fwd_pd);

  // MKL-DNN introduced padded formats since 0.15 which require more memory
  // compared to the actual size of the tensor. Currently, MKL-DNN operators
  // still reuse memory from memory planning, so here we need to select a
  // suboptimal kernel for computation that has the expected memory size requirements
  while (deconv_pd->diff_dst_desc().get_size() != GetMemDescSize(out_md) ||
         deconv_pd->src_desc().get_size() != GetMemDescSize(data_md) ||
         deconv_pd->diff_weights_desc().get_size() != GetMemDescSize(weight_md)) {
    if (!deconv_pd->next_impl()) {
      ImposePlainWherePadding(data_md, out_md, weight_md, deconv_pd->diff_dst_desc().get_size(),
                              deconv_pd->src_desc().get_size(),
                              deconv_pd->diff_weights_desc().get_size());
      *deconv_pd = DeconvBwdWeightPD(desc(), engine, fwd_pd);
    }
  }
  return deconv_pd;
}

class MKLDNNDeconvForward {
 public:
  MKLDNNDeconvForward(const DeconvolutionParam &param, const NDArray &data, const NDArray &weights,
                      const NDArray *bias, const NDArray &output);
  const DeconvFwd &GetFwd() const { return *fwd; }

  const DeconvFwdPD &GetPd() const { return *fwd_pd; }

 private:
  std::shared_ptr<DeconvFwd> fwd;
  std::shared_ptr<DeconvFwdPD> fwd_pd;
};  // class MKLDNNDeconvForward

MKLDNNDeconvForward::MKLDNNDeconvForward(const DeconvolutionParam &param, const NDArray &data,
                                         const NDArray &weights, const NDArray *bias,
                                         const NDArray &output)
    : fwd_pd(GetDeconvFwdImpl(param, data, weights, bias, output)) {
  fwd = std::make_shared<DeconvFwd>(GetPd());
}

MKLDNNDeconvForward &GetDeconvFwd(const DeconvolutionParam &param, const NDArray &data,
                                  const NDArray &weights, const NDArray *bias,
                                  const NDArray &output) {
  using deconv_fwd_map = std::unordered_map<DeconvSignature, MKLDNNDeconvForward, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local deconv_fwd_map fwds;
#else
  static MX_THREAD_LOCAL deconv_fwd_map fwds;
#endif
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
    auto fwd = MKLDNNDeconvForward(param, data, weights, bias, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNDeconvolutionForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                                const std::vector<NDArray> &in_data,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const DeconvolutionParam &param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  auto &data = in_data[deconv::kData];
  auto &weight = in_data[deconv::kWeight];
  const NDArray *bias = param.no_bias ? nullptr : &in_data[deconv::kBias];

  MKLDNNDeconvForward &fwd = GetDeconvFwd(param, data, weight, bias, out_data[deconv::kOut]);

  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (weight.IsMKLDNNData())
      // This asks the engine to change the layout of the weight array after
      // it's used.
      weight.Reorder2DefaultAsync();
  } else {
    // For inference, we want to reorder the weight array so we don't need to
    // reorder data every time.
    if (weight.IsDefaultData()) {
      // We also need to modify the layout on the original weight array. The
      // data conversion happens after the weight array is used.
      weight.MKLDNNDataReorderAsync(IOLogicalSwapDesc(fwd.GetPd().weights_desc(), param.num_group));
    } else {
      CHECK(weight.GetMKLDNNData()->get_desc() ==
            IOLogicalSwapDesc(fwd.GetPd().weights_desc(), param.num_group));
    }
  }

  // MXNet (correctly) assumes that deconvolution is implemented using convolution primitives.
  // For that, we would pass input tensor in place of output and output tensor in place of
  // input (for appropriate convolution primitives: deconvolution forward = convolution backward
  // data, deconvolution backward data = convolution forward). Convolution primitive expects
  // weight tensor with shape (o, i, h, w), but because we swapped input and output tensors:
  // o = input_channels, i = output_channels. So in that case, deconvolution needs a weight
  // tensor with shape (input_channels, output_channels, h, w), which is (i, o, h, w) and MXNet
  // provides such tensor.

  // MKLDNN's deconvolution primitive also expects weight tensor with shape (o, i, h, w),
  // but this time we don't swap input and output tensors, so o = output_channels, i =
  // input_channels, so the current weight tensor won't fit (when oihw != iohw). But actually,
  // underneath deconvolution MKLDNN also uses convolution, so even though it expects the weight
  // tensor with shape (o, i, h, w), it wants it in iohw format, so it's physical representation
  // match current weight tensor.

  // So here we swap logical order of input and output dimensions for weight tensor just for MKLDNN
  // operations
  IOLogicalSwapMKLDNNMem(weight, param.num_group);

  auto data_mem = data.GetMKLDNNDataReorder(fwd.GetPd().src_desc());
  const mkldnn::memory *weight_mem =
      GetWeights(weight, fwd.GetPd().weights_desc(), param.num_group);
  mkldnn_output_t out_mem =
      CreateMKLDNNMem(out_data[deconv::kOut], fwd.GetPd().dst_desc(), req[deconv::kOut]);
  mkldnn_args_map_t net_args;
  if (bias) {
    const mkldnn::memory *bias_mem = in_data[deconv::kBias].GetMKLDNNData();
    net_args.insert({MKLDNN_ARG_BIAS, *bias_mem});
  }

  net_args.insert({MKLDNN_ARG_SRC, *data_mem});
  net_args.insert({MKLDNN_ARG_WEIGHTS, *weight_mem});
  net_args.insert({MKLDNN_ARG_DST, *out_mem.second});
  MKLDNNStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
  CommitOutput(out_data[deconv::kOut], out_mem);
  MKLDNNStream::Get()->Submit();

  // swap back from oihw to iohw
  IOLogicalSwapMKLDNNMem(weight, param.num_group);
}

class MKLDNNDeconvBackward {
  std::shared_ptr<DeconvBwdDataPD> bwd_data_pd_;
  std::shared_ptr<DeconvBwdWeightPD> bwd_weight_pd_;
  std::shared_ptr<DeconvBwdData> bwd_data_;
  std::shared_ptr<DeconvBwdWeight> bwd_weight_;

 public:
  MKLDNNDeconvBackward(const DeconvolutionParam &param, const NDArray &data, const NDArray &weights,
                       const NDArray *bias, const NDArray &output) {
    const auto fwd_pd = GetDeconvFwdImpl(param, data, weights, bias, output);
    bwd_data_pd_ = GetDeconvBwdDataImpl(param, data, weights, output, *fwd_pd);
    bwd_weight_pd_ = GetDeconvBwdWeightsImpl(param, data, weights, bias, output, *fwd_pd);
    bwd_data_ = std::make_shared<DeconvBwdData>(GetDataPd());
    bwd_weight_ = std::make_shared<DeconvBwdWeight>(GetWeightsPd());
  }

  const DeconvBwdData &GetBwdData() const { return *bwd_data_; }

  const DeconvBwdWeight &GetBwdWeights() const { return *bwd_weight_; }

  const DeconvBwdDataPD &GetDataPd() const { return *bwd_data_pd_; }

  const DeconvBwdWeightPD &GetWeightsPd() const { return *bwd_weight_pd_; }
};

typedef ParamOpSign<DeconvolutionParam> MKLDNNDeconvSignature;

static inline MKLDNNDeconvBackward &GetDeconvBwd(const DeconvolutionParam &param,
                                                 const NDArray &data, const NDArray &weights,
                                                 const NDArray *bias, const NDArray &output) {
  using mkldnn_deconv_bwd_map =
      std::unordered_map<MKLDNNDeconvSignature, MKLDNNDeconvBackward, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local mkldnn_deconv_bwd_map bwds;
#else
  static MX_THREAD_LOCAL mkldnn_deconv_bwd_map bwds;
#endif
  MKLDNNDeconvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias) key.AddSign(*bias);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto bwd = MKLDNNDeconvBackward(param, data, weights, bias, output);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNDeconvolutionBackward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  const DeconvolutionParam &param = nnvm::get<DeconvolutionParam>(attrs.parsed);

  auto &data = inputs[deconv::kData + 1];
  auto &weight = inputs[deconv::kWeight + 1];
  const auto *bias = param.no_bias ? nullptr : &inputs[deconv::kBias + 1];
  auto &out_grad = inputs[deconv::kOut];

  CHECK_NE(req[deconv::kWeight], kWriteInplace) << "cannot write weight inplace";
  MKLDNNDeconvBackward &deconvBwd = GetDeconvBwd(param, data, weight, bias, out_grad);
  auto out_grad_mem = out_grad.GetMKLDNNDataReorder(deconvBwd.GetDataPd().diff_dst_desc());
  if (req[deconv::kData]) {
    // swap is explained in MKLDNNDeconvolutionForward
    IOLogicalSwapMKLDNNMem(weight, param.num_group);
    auto weight_mem = GetWeights(weight, deconvBwd.GetDataPd().weights_desc(), param.num_group);
    auto in_grad_mem = CreateMKLDNNMem(in_grad[deconv::kData],
                                       deconvBwd.GetDataPd().diff_src_desc(), req[deconv::kData]);
    mkldnn_args_map_t net_args = {{MKLDNN_ARG_DIFF_DST, *out_grad_mem},
                                  {MKLDNN_ARG_WEIGHTS, *weight_mem},
                                  {MKLDNN_ARG_DIFF_SRC, *in_grad_mem.second}};
    MKLDNNStream::Get()->RegisterPrimArgs(deconvBwd.GetBwdData(), net_args);
    CommitOutput(in_grad[deconv::kData], in_grad_mem);
  }
  if (req[deconv::kWeight] || req[deconv::kBias]) {
    if (deconvBwd.GetDataPd().diff_dst_desc() != deconvBwd.GetWeightsPd().diff_dst_desc())
      out_grad_mem = out_grad.GetMKLDNNDataReorder(deconvBwd.GetWeightsPd().diff_dst_desc());
    auto data_mem = data.GetMKLDNNDataReorder(deconvBwd.GetWeightsPd().src_desc());
    mkldnn_output_t in_grad_weight;
    const mkldnn::memory::desc &wei_md = deconvBwd.GetWeightsPd().diff_weights_desc();
    // swaps are explained in MKLDNNDeconvolutionForward
    // CreateMKLDNNWeightGrad always creates a new tensor as IsDefaultFormat always fails (because
    // of logical swap) We try to reuse in_grad[deconv::kWeight] memory (which, when not swapped, is
    // always in default format), so here we check if after a swap, wei_md will have a default
    // format
    if (req[deconv::kWeight] == OpReqType::kWriteTo &&
        IsDefaultFormat(IOLogicalSwapDesc(wei_md, param.num_group))) {
      in_grad_weight = {OutDataOp::Noop,
                        const_cast<NDArray &>(in_grad[deconv::kWeight]).CreateMKLDNNData(wei_md)};
    } else {
      IOLogicalSwapMKLDNNMem(in_grad[deconv::kWeight], param.num_group);
      in_grad_weight =
          CreateMKLDNNWeightGrad(in_grad[deconv::kWeight], wei_md, req[deconv::kWeight]);
    }

    mkldnn_args_map_t net_args = {{MKLDNN_ARG_DIFF_DST, *out_grad_mem},
                                  {MKLDNN_ARG_SRC, *data_mem},
                                  {MKLDNN_ARG_DIFF_WEIGHTS, *in_grad_weight.second}};
    mkldnn_output_t in_grad_bias;
    if (!param.no_bias) {
      in_grad_bias = CreateMKLDNNMem(in_grad[deconv::kBias],
                                     deconvBwd.GetWeightsPd().diff_bias_desc(), req[deconv::kBias]);
      net_args.insert({MKLDNN_ARG_DIFF_BIAS, *in_grad_bias.second});
    }
    MKLDNNStream::Get()->RegisterPrimArgs(deconvBwd.GetBwdWeights(), net_args);
    CommitOutput(in_grad[deconv::kWeight], in_grad_weight);
    // CommitOutput Should run after RegisterPrimArgs for memory dependency
    if (!param.no_bias) CommitOutput(in_grad[deconv::kBias], in_grad_bias);
  }
  MKLDNNStream::Get()->Submit();

  // swap back from oihw to iohw
  if (req[deconv::kData]) IOLogicalSwapMKLDNNMem(weight, param.num_group);
  if (req[deconv::kWeight] || req[deconv::kBias])
    IOLogicalSwapMKLDNNMem(in_grad[deconv::kWeight], param.num_group);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
