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

#include "./mkldnn_deconvolution-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNDeconv(const DeconvolutionParam &params, const NDArray &input) {
  return params.kernel.ndim() == 2 && input.shape().ndim() == 4 &&
         (input.dtype() == mshadow::kFloat32 || input.dtype() == mshadow::kBfloat16);
}

// Forward

void MKLDNNDeconvolutionForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const auto &param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  const auto &tensors = MKLDNNDeconvFwd::Tensors(param.no_bias, inputs, outputs);
  MKLDNNDeconvFwd &fwd = MKLDNNDeconvFwd::GetCached(param, tensors);

  fwd.ControlWeightsFormat(param.num_group, ctx.is_train, tensors.weights);
  fwd.Execute(param.num_group, req, tensors);
}

MKLDNNDeconvFwd::Tensors::Tensors(const NDArray &data, const NDArray &weights,
                                  const NDArray *const bias, const NDArray &out)
    : data(data), weights(weights), bias(bias), out(out) {}

MKLDNNDeconvFwd::Tensors::Tensors(const bool no_bias, const std::vector<NDArray> &inputs,
                                  const std::vector<NDArray> &outputs)
    : data(inputs[deconv::kData]),
      weights(inputs[deconv::kWeight]),
      bias(no_bias ? nullptr : &inputs[deconv::kBias]),
      out(outputs[deconv::kOut]) {}

MKLDNNDeconvFwd &MKLDNNDeconvFwd::GetCached(const DeconvolutionParam &param,
                                            const Tensors &tensors) {
  using deconv_fwd_map = std::unordered_map<DeconvSignature, MKLDNNDeconvFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local deconv_fwd_map fwds;
#else
  static MX_THREAD_LOCAL deconv_fwd_map fwds;
#endif
  DeconvSignature key(param);
  key.AddSign(tensors.data);
  key.AddSign(tensors.weights);
  key.AddSign(tensors.out);
  if (tensors.bias) {
    key.AddSign(*tensors.bias);
  }

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const MKLDNNDeconvFwd fwd(param, tensors);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

std::shared_ptr<deconv_fwd_pd_t> MKLDNNDeconvFwd::MakePD(const DeconvolutionParam &param,
                                                         const Tensors &tensors) {
  DeconvDescCreator ddc(param, tensors.data, tensors.weights, tensors.bias, tensors.out);
  const auto pd = std::make_shared<deconv_fwd_pd_t>(ddc.MakeFwdDesc(), ddc.engine);
  const auto get_data_size = [&pd]() { return pd->src_desc().get_size(); };
  const auto get_weights_size = [&pd]() { return pd->weights_desc().get_size(); };
  const auto get_out_size = [&pd]() { return pd->dst_desc().get_size(); };

  while (!ddc.CheckImplSizeReq(get_data_size(), get_weights_size(), get_out_size())) {
    if (!pd->next_impl()) {
      // ImposePlainWherePadding fails when all memory descriptors already have plain formats
      // imposed, meaning there is no implementation with plain formats
      CHECK(ddc.ImposePlainWherePadding(get_data_size(), get_weights_size(), get_out_size()))
          << "No implementation of deconvolution forward propagation";
      *pd = deconv_fwd_pd_t(ddc.MakeFwdDesc(), ddc.engine);
    }
  }
  return pd;
}

MKLDNNDeconvFwd::MKLDNNDeconvFwd(const DeconvolutionParam &param, const Tensors &tensors)
    : fwd_pd(MakePD(param, tensors)) {
  fwd = std::make_shared<deconv_fwd_t>(*fwd_pd);
}

void MKLDNNDeconvFwd::ControlWeightsFormat(const uint32_t num_group, const bool is_train,
                                           const NDArray &weights) {
  if (is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (weights.IsMKLDNNData()) {
      // This asks the engine to change the layout of the weights array after it's used.
      weights.Reorder2DefaultAsync();
    }
  } else {
    // For inference, we want to reorder the weights array so we don't need to
    // reorder data every time.
    if (weights.IsDefaultData()) {
      // We also need to modify the layout on the original weights array.
      // The data conversion happens after the weights array is used.
      weights.MKLDNNDataReorderAsync(IOLogicalSwapDesc(fwd_pd->weights_desc(), num_group));
    } else {
      CHECK(weights.GetMKLDNNData()->get_desc() ==
            IOLogicalSwapDesc(fwd_pd->weights_desc(), num_group));
    }
  }
}

void MKLDNNDeconvFwd::Execute(const uint32_t num_group, const std::vector<OpReqType> &req,
                              const Tensors &tensors) {
  // MXNet (correctly) assumes that deconvolution is implemented using convolution primitives.
  // For that, we would pass input tensor in place of output and output tensor in place of input
  // (for appropriate convolution primitives: deconvolution forward = convolution backward data,
  // deconvolution backward data = convolution forward).
  // The convolution primitive expects weights tensor with the shape of
  // (primitive_out_channels, primitive_in_channels, h, w), but with swapped input and output:
  // primitive_out_channels = deconv_in_channels, primitive_in_channels = deconv_out_channels,
  // so it becomes (deconv_in_channels, deconv_out_channels, h, w) and MXNet provides such tensor.
  //
  // MKLDNN deconvolution primitive also (as convolution) expects weights tensor with the shape of
  // (primitive_out_channels, primitive_in_channels, h, w), but this time we don't swap input and
  // output tensors, so:
  // primitive_out_channels = deconv_out_channels, primitive_in_channels = deconv_in_channels,
  // thus the current weights tensor won't fit (when deconv_out_channels != deconv_in_channels).
  // However, underneath deconvolution MKLDNN also uses convolution, so even though it expects the
  // weights tensor with the logical order of oihw, it wants its physical representation to
  // match the order of iohw, which is the same as current weights tensor.
  //
  // So here we swap logical order of input and output dimensions for weights tensor just for
  // MKLDNN operations.
  IOLogicalSwapMKLDNNMem(tensors.weights, num_group);
  {
    mkldnn_args_map_t net_args;
    const auto &out_mem = OutMem(req[deconv::kOut], tensors.out);

    net_args.insert({MKLDNN_ARG_SRC, *DataMem(tensors.data)});
    net_args.insert({MKLDNN_ARG_WEIGHTS, *WeightsMem(num_group, tensors.weights)});
    net_args.insert({MKLDNN_ARG_DST, *out_mem.second});
    if (tensors.bias) {
      net_args.insert({MKLDNN_ARG_BIAS, *BiasMem(*tensors.bias)});
    }

    // CommitOutput should run after RegisterPrimArgs for memory dependency
    MKLDNNStream::Get()->RegisterPrimArgs(*fwd, net_args);
    CommitOutput(tensors.out, out_mem);
    MKLDNNStream::Get()->Submit();
  }
  IOLogicalSwapMKLDNNMem(tensors.weights, num_group);  // swap back from oihw to iohw
}

const mkldnn::memory *MKLDNNDeconvFwd::DataMem(const NDArray &data) const {
  return data.GetMKLDNNDataReorder(fwd_pd->src_desc());
}

const mkldnn::memory *MKLDNNDeconvFwd::WeightsMem(const uint32_t num_group,
                                                  const NDArray &weights) const {
  return GetWeights(weights, fwd_pd->weights_desc(), num_group);
}

const mkldnn::memory *MKLDNNDeconvFwd::BiasMem(const NDArray &bias) const {
  return bias.GetMKLDNNData();
}

mkldnn_output_t MKLDNNDeconvFwd::OutMem(const OpReqType req, const NDArray &out) const {
  return CreateMKLDNNMem(out, fwd_pd->dst_desc(), req);
}

// Backward

void MKLDNNDeconvolutionBackward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  CHECK_NE(req[deconv::kWeight], kWriteInplace) << "Cannot write weights inplace";

  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const auto &param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  const auto &read_tensors = MKLDNNDeconvBwd::ReadTensors(param.no_bias, inputs);
  const auto &write_tensors = MKLDNNDeconvBwd::WriteTensors(param.no_bias, outputs);
  MKLDNNDeconvBwd &bwd = MKLDNNDeconvBwd::GetCached(param, read_tensors);

  bwd.Execute(param.num_group, req, read_tensors, write_tensors);
}

MKLDNNDeconvBwd::ReadTensors::ReadTensors(const bool no_bias, const std::vector<NDArray> &inputs)
    : data(inputs[deconv::kData + 1]),
      weights(inputs[deconv::kWeight + 1]),
      bias(no_bias ? nullptr : &inputs[deconv::kBias + 1]),
      out_grad(inputs[deconv::kOut]) {}

MKLDNNDeconvBwd::WriteTensors::WriteTensors(const bool no_bias, const std::vector<NDArray> &outputs)
    : data_grad(outputs[deconv::kData]),
      weights_grad(outputs[deconv::kWeight]),
      bias_grad(no_bias ? nullptr : &outputs[deconv::kBias]) {}

MKLDNNDeconvBwd &MKLDNNDeconvBwd::GetCached(const DeconvolutionParam &param,
                                            const ReadTensors &read_tensors) {
  using deconv_bwd_map = std::unordered_map<DeconvSignature, MKLDNNDeconvBwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local deconv_bwd_map bwds;
#else
  static MX_THREAD_LOCAL deconv_bwd_map bwds;
#endif
  DeconvSignature key(param);
  key.AddSign(read_tensors.data);
  key.AddSign(read_tensors.weights);
  key.AddSign(read_tensors.out_grad);
  if (read_tensors.bias) {
    key.AddSign(*read_tensors.bias);
  }

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    const MKLDNNDeconvBwd bwd(param, read_tensors);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

std::shared_ptr<deconv_bwd_data_pd_t> MKLDNNDeconvBwd::MakeDataPD(const DeconvolutionParam &param,
                                                                  const ReadTensors &read_tensors,
                                                                  const deconv_fwd_pd_t &fwd_pd) {
  DeconvDescCreator ddc(param, read_tensors.data, read_tensors.weights, nullptr,
                        read_tensors.out_grad);
  const auto pd = std::make_shared<deconv_bwd_data_pd_t>(ddc.MakeBwdDataDesc(), ddc.engine, fwd_pd);
  const auto get_data_size = [&pd]() { return pd->diff_src_desc().get_size(); };
  const auto get_weights_size = [&pd]() { return pd->weights_desc().get_size(); };
  const auto get_out_size = [&pd]() { return pd->diff_dst_desc().get_size(); };

  while (!ddc.CheckImplSizeReq(get_data_size(), get_weights_size(), get_out_size())) {
    if (!pd->next_impl()) {
      // ImposePlainWherePadding fails when all memory descriptors already have plain formats
      // imposed, meaning there is no implementation with plain formats
      CHECK(ddc.ImposePlainWherePadding(get_data_size(), get_weights_size(), get_out_size()))
          << "No implementation of deconvolution backward propagation";
      *pd = deconv_bwd_data_pd_t(ddc.MakeBwdDataDesc(), ddc.engine, fwd_pd);
    }
  }
  return pd;
}

std::shared_ptr<deconv_bwd_weights_pd_t> MKLDNNDeconvBwd::MakeWeightsPD(
    const DeconvolutionParam &param, const ReadTensors &read_tensors,
    const deconv_fwd_pd_t &fwd_pd) {
  DeconvDescCreator ddc(param, read_tensors.data, read_tensors.weights, read_tensors.bias,
                        read_tensors.out_grad);
  const auto pd =
      std::make_shared<deconv_bwd_weights_pd_t>(ddc.MakeBwdWeightsDesc(), ddc.engine, fwd_pd);
  const auto get_data_size = [&pd]() { return pd->src_desc().get_size(); };
  const auto get_weights_size = [&pd]() { return pd->diff_weights_desc().get_size(); };
  const auto get_out_size = [&pd]() { return pd->diff_dst_desc().get_size(); };

  while (!ddc.CheckImplSizeReq(get_data_size(), get_weights_size(), get_out_size())) {
    if (!pd->next_impl()) {
      // ImposePlainWherePadding fails when all memory descriptors already have plain formats
      // imposed, meaning there is no implementation with plain formats
      CHECK(ddc.ImposePlainWherePadding(get_data_size(), get_weights_size(), get_out_size()))
          << "No implementation of calculating deconvolution weights gradient";
      *pd = deconv_bwd_weights_pd_t(ddc.MakeBwdWeightsDesc(), ddc.engine, fwd_pd);
    }
  }
  return pd;
}

MKLDNNDeconvBwd::MKLDNNDeconvBwd(const DeconvolutionParam &param, const ReadTensors &read_tensors) {
  const auto &fwd_pd = MKLDNNDeconvFwd::MakePD(
      param, MKLDNNDeconvFwd::Tensors(read_tensors.data, read_tensors.weights, read_tensors.bias,
                                      read_tensors.out_grad));
  bwd_data_pd = MakeDataPD(param, read_tensors, *fwd_pd);
  bwd_weights_pd = MakeWeightsPD(param, read_tensors, *fwd_pd);
  bwd_data = std::make_shared<deconv_bwd_t>(*bwd_data_pd);
  bwd_weights = std::make_shared<deconv_bwd_weights_t>(*bwd_weights_pd);
}

void MKLDNNDeconvBwd::Execute(const uint32_t num_group, const std::vector<OpReqType> &req,
                              const ReadTensors &read_tensors, const WriteTensors &write_tensors) {
  // swaps are explained in MKLDNNDeconvFwd::Execute
  IOSwapWeightsTensors(num_group, req, read_tensors.weights, write_tensors.weights_grad);
  {
    auto *const out_grad_mem = ScheduleBwdData(num_group, req, read_tensors, write_tensors);
    ScheduleBwdWeights(num_group, req, read_tensors, write_tensors, out_grad_mem);
    MKLDNNStream::Get()->Submit();
  }
  IOSwapWeightsTensors(num_group, req, read_tensors.weights, write_tensors.weights_grad);
}

void MKLDNNDeconvBwd::IOSwapWeightsTensors(const uint32_t num_group,
                                           const std::vector<OpReqType> &req,
                                           const NDArray &weights, const NDArray &weights_grad) {
  if (req[deconv::kData]) {
    IOLogicalSwapMKLDNNMem(weights, num_group);
  }
  if (req[deconv::kWeight] || req[deconv::kBias]) {
    IOLogicalSwapMKLDNNMem(weights_grad, num_group);
  }
}

const mkldnn::memory *MKLDNNDeconvBwd::ScheduleBwdData(const uint32_t num_group,
                                                       const std::vector<OpReqType> &req,
                                                       const ReadTensors &read_tensors,
                                                       const WriteTensors &write_tensors) {
  if (req[deconv::kData]) {
    mkldnn_args_map_t net_args;
    auto *const out_grad_mem = OutGradMem(read_tensors.out_grad);
    const auto &data_grad_mem = DataGradMem(req[deconv::kData], write_tensors.data_grad);

    net_args.insert({MKLDNN_ARG_DIFF_DST, *out_grad_mem});
    net_args.insert({MKLDNN_ARG_WEIGHTS, *WeightsMem(num_group, read_tensors.weights)});
    net_args.insert({MKLDNN_ARG_DIFF_SRC, *data_grad_mem.second});

    // CommitOutput should run after RegisterPrimArgs for memory dependency
    MKLDNNStream::Get()->RegisterPrimArgs(*bwd_data, net_args);
    CommitOutput(write_tensors.data_grad, data_grad_mem);
    return out_grad_mem;
  }
  return nullptr;
}

void MKLDNNDeconvBwd::ScheduleBwdWeights(const uint32_t num_group,
                                         const std::vector<OpReqType> &req,
                                         const ReadTensors &read_tensors,
                                         const WriteTensors &write_tensors,
                                         const mkldnn::memory *const out_grad_mem) {
  if (req[deconv::kWeight] || req[deconv::kBias]) {
    mkldnn_args_map_t net_args;
    const auto &weights_grad_mem =
        WeightsGradMem(num_group, req[deconv::kWeight], write_tensors.weights_grad);
    const auto &bias_grad_mem = BiasGradMem(req[deconv::kBias], write_tensors.bias_grad);

    net_args.insert({MKLDNN_ARG_DIFF_DST, *OutGradMem(read_tensors.out_grad, out_grad_mem)});
    net_args.insert({MKLDNN_ARG_SRC, *DataMem(read_tensors.data)});
    net_args.insert({MKLDNN_ARG_DIFF_WEIGHTS, *weights_grad_mem.second});
    if (bias_grad_mem.second) {
      net_args.insert({MKLDNN_ARG_DIFF_BIAS, *bias_grad_mem.second});
    }

    // CommitOutput should run after RegisterPrimArgs for memory dependency
    MKLDNNStream::Get()->RegisterPrimArgs(*bwd_weights, net_args);
    CommitOutput(write_tensors.weights_grad, weights_grad_mem);
    if (bias_grad_mem.second) {
      CommitOutput(*write_tensors.bias_grad, bias_grad_mem);
    }
  }
}

const mkldnn::memory *MKLDNNDeconvBwd::DataMem(const NDArray &data) const {
  return data.GetMKLDNNDataReorder(bwd_weights_pd->src_desc());
}

const mkldnn::memory *MKLDNNDeconvBwd::WeightsMem(const uint32_t num_group,
                                                  const NDArray &weights) const {
  return GetWeights(weights, bwd_data_pd->weights_desc(), num_group);
}

const mkldnn::memory *MKLDNNDeconvBwd::OutGradMem(const NDArray &out_grad) const {
  return out_grad.GetMKLDNNDataReorder(bwd_data_pd->diff_dst_desc());
}

const mkldnn::memory *MKLDNNDeconvBwd::OutGradMem(const NDArray &out_grad,
                                                  const mkldnn::memory *const out_grad_mem) const {
  if (!out_grad_mem || bwd_data_pd->diff_dst_desc() != bwd_weights_pd->diff_dst_desc()) {
    return out_grad.GetMKLDNNDataReorder(bwd_weights_pd->diff_dst_desc());
  }
  return out_grad_mem;
}

mkldnn_output_t MKLDNNDeconvBwd::DataGradMem(const OpReqType req, const NDArray &data_grad) const {
  return CreateMKLDNNMem(data_grad, bwd_data_pd->diff_src_desc(), req);
}

mkldnn_output_t MKLDNNDeconvBwd::WeightsGradMem(const uint32_t num_group, const OpReqType req,
                                                const NDArray &weights_grad) const {
  // CreateMKLDNNWeightGrad always creates a new tensor as IsDefaultFormat always fails (because
  // of the logical swap - explained in MKLDNNDeconvFwd::Execute). We try to reuse weights_grad
  // memory (which, when not swapped, is always in default format), so here we check if after a
  // swap, wei_md will have a default format
  const auto &wei_md = bwd_weights_pd->diff_weights_desc();
  if (req == OpReqType::kWriteTo && IsDefaultFormat(IOLogicalSwapDesc(wei_md, num_group))) {
    return {OutDataOp::Noop, const_cast<NDArray &>(weights_grad).CreateMKLDNNData(wei_md)};
  }
  return CreateMKLDNNWeightGrad(weights_grad, wei_md, req);
}

mkldnn_output_t MKLDNNDeconvBwd::BiasGradMem(const OpReqType req, const NDArray *const bias) const {
  return bias ? CreateMKLDNNMem(*bias, bwd_weights_pd->diff_bias_desc(), req)
              : mkldnn_output_t(OutDataOp::Noop, nullptr);
}

// DeconvDescCreator

DeconvDescCreator::DeconvDescCreator(const DeconvolutionParam &param, const NDArray &data,
                                     const NDArray &weights, const NDArray *const bias,
                                     const NDArray &out)
    : data_md(GetMemDesc(data)),
      weights_md(GetDeconvWeightsDesc(weights, param.num_group)),
      bias_md(bias ? GetMemDesc(*bias) : mkldnn::memory::desc()),
      out_md(GetMemDesc(out)),
      strides(param.stride.ndim()),
      padding(param.pad.ndim()),
      dilates(param.dilate.ndim()),
      engine(CpuEngine::Get()->get_engine()) {
  // assuming only deconv2D is supported for now
  CHECK(param.stride.ndim() == param.pad.ndim() && param.stride.ndim() == param.dilate.ndim());
  CHECK(param.stride.ndim() == 2);
  for (int i = 0; i < param.stride.ndim(); ++i) {
    strides[i] = param.stride[i];
    padding[i] = param.pad[i];
    dilates[i] = param.dilate[i] - 1;
  }
}

bool DeconvDescCreator::ImposePlainWherePadding(const size_t data_size, const size_t weights_size,
                                                const size_t out_size) {
  // Changing only one at a time, so maybe better implementations will be selected (than entirely
  // plain one)
  if (data_md.data.format_kind == dnnl_format_kind_any && data_size != GetMemDescSize(data_md)) {
    data_md = GetDesc(data_md, GetDefaultFormat(data_md));
    return true;
  } else if (out_md.data.format_kind == dnnl_format_kind_any &&
             out_size != GetMemDescSize(out_md)) {
    out_md = GetDesc(out_md, GetDefaultFormat(out_md));
    return true;
  } else if (weights_md.data.format_kind == dnnl_format_kind_any &&
             weights_size != GetMemDescSize(weights_md)) {
    const int num_gr = (weights_md.data.ndims > data_md.data.ndims) ? weights_md.data.dims[0] : 1;
    weights_md = IOLogicalSwapDesc(weights_md, num_gr);
    weights_md = IOLogicalSwapDesc(GetDesc(weights_md, GetDefaultFormat(weights_md)), num_gr);
    return true;
  }
  return false;
}

bool DeconvDescCreator::CheckImplSizeReq(const size_t data_size, const size_t weights_size,
                                         const size_t out_size) const {
  // MKLDNN introduced padded formats since 0.15 which require more memory
  // compared to the actual size of the tensor. Currently, MKLDNN operators
  // still reuse memory from memory planning, so here we need to accept only a
  // kernel that has the expected memory size requirements (which is suboptimal)
  return (data_size == GetMemDescSize(data_md) && weights_size == GetMemDescSize(weights_md) &&
          out_size == GetMemDescSize(out_md));
}

deconv_fwd_t::desc DeconvDescCreator::MakeFwdDesc() const {
  return deconv_fwd_t::desc(mkldnn::prop_kind::forward_training,
                            mkldnn::algorithm::deconvolution_direct, data_md, weights_md, bias_md,
                            out_md, strides, dilates, padding, padding);
}

deconv_bwd_t::desc DeconvDescCreator::MakeBwdDataDesc() const {
  return deconv_bwd_t::desc(mkldnn::algorithm::deconvolution_direct, data_md, weights_md, out_md,
                            strides, dilates, padding, padding);
}

deconv_bwd_weights_t::desc DeconvDescCreator::MakeBwdWeightsDesc() const {
  return deconv_bwd_weights_t::desc(mkldnn::algorithm::deconvolution_direct, data_md, weights_md,
                                    bias_md, out_md, strides, dilates, padding, padding);
}

// Utilities

mkldnn::memory::desc IOLogicalSwapDesc(const mkldnn::memory::desc &desc, const int num_groups) {
  std::vector<int> order(desc.data.ndims);
  std::iota(std::begin(order), std::end(order), 0);
  const int offset = int(num_groups > 1);
  std::swap(order[offset + 0], order[offset + 1]);
  return desc.permute_axes(order);
}

void IOLogicalSwapMKLDNNMem(const NDArray &arr, const int num_groups) {
  mkldnn::memory::desc desc;
  if (arr.IsMKLDNNData()) {
    desc = arr.GetMKLDNNData()->get_desc();
  } else {
    // GetMKLDNNData won't take groups into account when creating mkldnn::memory, we need to use
    // descriptor from GetWeightDesc but with default format
    const auto &temp = GetWeightDesc(arr, num_groups);
    desc = mkldnn::memory::desc(
        temp.dims(), temp.data_type(),
        static_cast<mkldnn::memory::format_tag>(GetDefaultFormat(temp.data.ndims)));
  }
  const_cast<NDArray &>(arr).UpdateMKLDNNMemDesc(IOLogicalSwapDesc(desc, num_groups));
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
