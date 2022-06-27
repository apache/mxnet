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
 * \file dnnl_deconvolution.cc
 */

#if MXNET_USE_ONEDNN == 1

#include "operator/nn/deconvolution-inl.h"
#include "dnnl_deconvolution-inl.h"

namespace mxnet {
namespace op {

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_convolution.html
bool SupportDNNLDeconv(const DeconvolutionParam& params, const NDArray& input) {
  return params.kernel.ndim() >= 1 && params.kernel.ndim() <= 3 &&
         SupportDNNL<3, 5, DNNLTypeMode::FloatTypes>(input);
}

DNNLDeconvFwd::Tensors::Tensors(const bool no_bias,
                                const std::vector<NDArray>& inputs,
                                const std::vector<NDArray>& outputs)
    : data(inputs[deconv::kData]),
      weights(inputs[deconv::kWeight]),
      bias(no_bias ? nullptr : &inputs[deconv::kBias]),
      out(outputs[deconv::kOut]) {}

DNNLDeconvFwd::Tensors::Tensors(const NDArray& data,
                                const NDArray& weights,
                                const NDArray* const bias,
                                const NDArray& out)
    : data(data), weights(weights), bias(bias), out(out) {}

DNNLDeconvBwd::ReadTensors::ReadTensors(const bool no_bias, const std::vector<NDArray>& inputs)
    : data(inputs[deconv::kData + 1]),
      weights(inputs[deconv::kWeight + 1]),
      bias(no_bias ? nullptr : &inputs[deconv::kBias + 1]),
      out_grad(inputs[deconv::kOut]) {}

DNNLDeconvBwd::WriteTensors::WriteTensors(const bool no_bias, const std::vector<NDArray>& outputs)
    : data_grad(outputs[deconv::kData]),
      weights_grad(outputs[deconv::kWeight]),
      bias_grad(no_bias ? nullptr : &outputs[deconv::kBias]) {}

void DNNLDeconvolutionForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const auto& param  = nnvm::get<DeconvolutionParam>(attrs.parsed);
  const auto tensors = DNNLDeconvFwd::Tensors(param.no_bias, inputs, outputs);
  const auto& fwd    = DNNLDeconvFwd::GetCached(param, tensors);

  fwd.ControlWeightsFormat(param.num_group, ctx.is_train, tensors.weights);
  fwd.Execute(param.num_group, req[deconv::kOut], tensors);
}

DNNLDeconvFwd::DNNLDeconvFwd(const DeconvolutionParam& param, const Tensors& tensors)
    : fwd_pd(CreatePrimitiveDesc(param, tensors)) {
  fwd = std::make_shared<deconv_fwd_t>(*fwd_pd);
}

DNNLDeconvFwd& DNNLDeconvFwd::GetCached(const DeconvolutionParam& param, const Tensors& tensors) {
  using deconv_fwd_map = std::unordered_map<DeconvSignature, DNNLDeconvFwd, OpHash>;
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
    const DNNLDeconvFwd fwd(param, tensors);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

std::shared_ptr<deconv_fwd_pd_t> DNNLDeconvFwd::CreatePrimitiveDesc(const DeconvolutionParam& param,
                                                                    const Tensors& tensors) {
  DeconvDescCreator ddc(param, tensors.data, tensors.weights, tensors.bias, tensors.out);
  auto fwd_desc = ddc.CreateFwdDesc();  // `fwd_desc` lifetime must be longer than `pd`
                                        // when using next_impl
  const auto& engine          = CpuEngine::Get()->get_engine();
  const auto pd               = std::make_shared<deconv_fwd_pd_t>(fwd_desc, engine);
  const auto get_data_size    = [&pd]() { return pd->src_desc().get_size(); };
  const auto get_weights_size = [&pd]() { return pd->weights_desc().get_size(); };
  const auto get_out_size     = [&pd]() { return pd->dst_desc().get_size(); };

  while (!ddc.CheckImplSizeReq(get_data_size(), get_weights_size(), get_out_size())) {
    if (!pd->next_impl()) {
      // ImposePlainWherePadding fails when all memory descriptors already have plain formats
      // imposed, meaning there is no implementation with plain formats
      CHECK(ddc.ImposePlainWherePadding(get_data_size(), get_weights_size(), get_out_size()))
          << "No implementation of deconvolution forward propagation";
      fwd_desc = ddc.CreateFwdDesc();
      *pd      = deconv_fwd_pd_t(fwd_desc, engine);
    }
  }
  return pd;
}

void DNNLDeconvFwd::ControlWeightsFormat(const uint32_t num_group,
                                         const bool is_train,
                                         const NDArray& weights) const {
  if (is_train) {
    // TODO(zhengda) kvstore doesn't handle DNNL correctly. Let's reorder it
    // to the default format for now.
    if (weights.IsDNNLData()) {
      // This asks the engine to change the layout of the weights array after it's used.
      weights.Reorder2DefaultAsync();
    }
  } else {
    // For inference, we want to reorder the weights array so we don't need to
    // reorder data every time.
    if (weights.IsDefaultData()) {
      // We also need to modify the layout on the original weights array.
      // The data conversion happens after the weights array is used.
      auto logical_swap_desc = IOLogicalSwapDesc(fwd_pd->weights_desc(), num_group);
      weights.DNNLDataReorderAsync(&logical_swap_desc);
    } else {
      CHECK(weights.GetDNNLData()->get_desc() ==
            IOLogicalSwapDesc(fwd_pd->weights_desc(), num_group));
    }
  }
}

void DNNLDeconvFwd::Execute(const uint32_t num_group,
                            const OpReqType req,
                            const Tensors& tensors) const {
  // MXNet (correctly) assumes that deconvolution is implemented using convolution primitives.
  // For that, we would pass input tensor in place of output and output tensor in place of input
  // (for appropriate convolution primitives: deconvolution forward = convolution backward data,
  // deconvolution backward data = convolution forward).
  // The convolution primitive expects weights tensor with the shape of
  // (primitive_out_channels, primitive_in_channels, h, w), but with swapped input and output:
  // primitive_out_channels = deconv_in_channels, primitive_in_channels = deconv_out_channels,
  // so it becomes (deconv_in_channels, deconv_out_channels, h, w) and MXNet provides such tensor.
  //
  // DNNL deconvolution primitive also (as convolution) expects weights tensor with the shape of
  // (primitive_out_channels, primitive_in_channels, h, w), but this time we don't swap input and
  // output tensors, so:
  // primitive_out_channels = deconv_out_channels, primitive_in_channels = deconv_in_channels,
  // thus the current weights tensor won't fit (when deconv_out_channels != deconv_in_channels).
  // However, underneath deconvolution DNNL also uses convolution, so even though it expects the
  // weights tensor with the logical order of oihw, it wants its physical representation to
  // match the order of iohw, which is the same as current weights tensor.
  //
  // So here we swap logical order of input and output dimensions for weights tensor just for
  // DNNL operations.
  IOLogicalSwapDNNLMem(tensors.weights, num_group);
  {
    dnnl_args_map_t net_args;
    const auto& out_mem = OutMem(req, tensors.out);

    net_args.insert({DNNL_ARG_SRC, *DataMem(tensors.data)});
    net_args.insert({DNNL_ARG_WEIGHTS, *WeightsMem(num_group, tensors.weights)});
    net_args.insert({DNNL_ARG_DST, *out_mem.second});
    if (tensors.bias) {
      net_args.insert({DNNL_ARG_BIAS, *BiasMem(*tensors.bias)});
    }

    // CommitOutput should run after RegisterPrimArgs for memory dependency
    DNNLStream::Get()->RegisterPrimArgs(*fwd, net_args);
    CommitOutput(tensors.out, out_mem);
    DNNLStream::Get()->Submit();
  }
  IOLogicalSwapDNNLMem(tensors.weights, num_group);  // swap back from oihw to iohw
}

void DNNLDeconvolutionBackward(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  CHECK_NE(req[deconv::kWeight], kWriteInplace) << "Cannot write weights inplace";

  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const auto& param        = nnvm::get<DeconvolutionParam>(attrs.parsed);
  const auto read_tensors  = DNNLDeconvBwd::ReadTensors(param.no_bias, inputs);
  const auto write_tensors = DNNLDeconvBwd::WriteTensors(param.no_bias, outputs);
  DNNLDeconvBwd& bwd       = DNNLDeconvBwd::GetCached(param, read_tensors);

  bwd.Execute(param.num_group, req, read_tensors, write_tensors);
}

DNNLDeconvBwd::DNNLDeconvBwd(const DeconvolutionParam& param, const ReadTensors& read_tensors) {
  const auto& fwd_pd = DNNLDeconvFwd::CreatePrimitiveDesc(
      param,
      DNNLDeconvFwd::Tensors(
          read_tensors.data, read_tensors.weights, read_tensors.bias, read_tensors.out_grad));
  bwd_data_pd    = CreateDataPrimitiveDesc(param, read_tensors, *fwd_pd);
  bwd_weights_pd = CreateWeightsPrimitiveDesc(param, read_tensors, *fwd_pd);
  bwd_data       = std::make_shared<deconv_bwd_data_t>(*bwd_data_pd);
  bwd_weights    = std::make_shared<deconv_bwd_weights_t>(*bwd_weights_pd);
}

DNNLDeconvBwd& DNNLDeconvBwd::GetCached(const DeconvolutionParam& param,
                                        const ReadTensors& read_tensors) {
  using deconv_bwd_map = std::unordered_map<DeconvSignature, DNNLDeconvBwd, OpHash>;
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
    const DNNLDeconvBwd bwd(param, read_tensors);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

std::shared_ptr<deconv_bwd_data_pd_t> DNNLDeconvBwd::CreateDataPrimitiveDesc(
    const DeconvolutionParam& param,
    const ReadTensors& read_tensors,
    const deconv_fwd_pd_t& fwd_pd) {
  DeconvDescCreator ddc(
      param, read_tensors.data, read_tensors.weights, nullptr, read_tensors.out_grad);
  auto bwd_d_desc = ddc.CreateBwdDataDesc();  // `bwd_d_desc` lifetime must be longer than `pd`
                                              // when using next_impl
  const auto& engine          = CpuEngine::Get()->get_engine();
  const auto pd               = std::make_shared<deconv_bwd_data_pd_t>(bwd_d_desc, engine, fwd_pd);
  const auto get_data_size    = [&pd]() { return pd->diff_src_desc().get_size(); };
  const auto get_weights_size = [&pd]() { return pd->weights_desc().get_size(); };
  const auto get_out_size     = [&pd]() { return pd->diff_dst_desc().get_size(); };

  while (!ddc.CheckImplSizeReq(get_data_size(), get_weights_size(), get_out_size())) {
    if (!pd->next_impl()) {
      // ImposePlainWherePadding fails when all memory descriptors already have plain formats
      // imposed, meaning there is no implementation with plain formats
      CHECK(ddc.ImposePlainWherePadding(get_data_size(), get_weights_size(), get_out_size()))
          << "No implementation of deconvolution backward propagation";
      bwd_d_desc = ddc.CreateBwdDataDesc();
      *pd        = deconv_bwd_data_pd_t(bwd_d_desc, engine, fwd_pd);
    }
  }
  return pd;
}

std::shared_ptr<deconv_bwd_weights_pd_t> DNNLDeconvBwd::CreateWeightsPrimitiveDesc(
    const DeconvolutionParam& param,
    const ReadTensors& read_tensors,
    const deconv_fwd_pd_t& fwd_pd) {
  DeconvDescCreator ddc(
      param, read_tensors.data, read_tensors.weights, read_tensors.bias, read_tensors.out_grad);
  auto bwd_w_desc = ddc.CreateBwdWeightsDesc();  // `bwd_w_desc` lifetime must be longer than `pd`
                                                 // when using next_impl
  const auto& engine       = CpuEngine::Get()->get_engine();
  const auto pd            = std::make_shared<deconv_bwd_weights_pd_t>(bwd_w_desc, engine, fwd_pd);
  const auto get_data_size = [&pd]() { return pd->src_desc().get_size(); };
  const auto get_weights_size = [&pd]() { return pd->diff_weights_desc().get_size(); };
  const auto get_out_size     = [&pd]() { return pd->diff_dst_desc().get_size(); };

  while (!ddc.CheckImplSizeReq(get_data_size(), get_weights_size(), get_out_size())) {
    if (!pd->next_impl()) {
      // ImposePlainWherePadding fails when all memory descriptors already have plain formats
      // imposed, meaning there is no implementation with plain formats
      CHECK(ddc.ImposePlainWherePadding(get_data_size(), get_weights_size(), get_out_size()))
          << "No implementation of calculating deconvolution weights gradient";
      bwd_w_desc = ddc.CreateBwdWeightsDesc();
      *pd        = deconv_bwd_weights_pd_t(bwd_w_desc, engine, fwd_pd);
    }
  }
  return pd;
}

void DNNLDeconvBwd::Execute(const uint32_t num_group,
                            const std::vector<OpReqType>& req,
                            const ReadTensors& read_tensors,
                            const WriteTensors& write_tensors) const {
  // swaps are explained in DNNLDeconvFwd::Execute
  IOSwapWeightsTensors(num_group, req, read_tensors.weights, write_tensors.weights_grad);
  {
    auto* const out_grad_mem =
        ScheduleBwdData(num_group, req[deconv::kData], read_tensors, write_tensors);
    ScheduleBwdWeights(num_group, req, read_tensors, write_tensors, out_grad_mem);
    DNNLStream::Get()->Submit();
  }
  IOSwapWeightsTensors(num_group, req, read_tensors.weights, write_tensors.weights_grad);
}

const dnnl::memory* DNNLDeconvBwd::ScheduleBwdData(const uint32_t num_group,
                                                   const OpReqType req,
                                                   const ReadTensors& read_tensors,
                                                   const WriteTensors& write_tensors) const {
  if (req) {
    dnnl_args_map_t net_args;
    auto* const out_grad_mem  = OutGradMem(read_tensors.out_grad);
    const auto& data_grad_mem = DataGradMem(req, write_tensors.data_grad);

    net_args.insert({DNNL_ARG_DIFF_DST, *out_grad_mem});
    net_args.insert({DNNL_ARG_WEIGHTS, *WeightsMem(num_group, read_tensors.weights)});
    net_args.insert({DNNL_ARG_DIFF_SRC, *data_grad_mem.second});

    // CommitOutput should run after RegisterPrimArgs for memory dependency
    DNNLStream::Get()->RegisterPrimArgs(*bwd_data, net_args);
    CommitOutput(write_tensors.data_grad, data_grad_mem);
    return out_grad_mem;
  }
  return nullptr;
}

void DNNLDeconvBwd::ScheduleBwdWeights(const uint32_t num_group,
                                       const std::vector<OpReqType>& req,
                                       const ReadTensors& read_tensors,
                                       const WriteTensors& write_tensors,
                                       const dnnl::memory* const out_grad_mem) const {
  OpReqType weight_req = req[deconv::kWeight];
  OpReqType bias_req   = req.size() > deconv::kBias ? req[deconv::kBias] : OpReqType::kNullOp;
  if (weight_req || bias_req) {
    dnnl_args_map_t net_args;
    const auto& weights_grad_mem =
        WeightsGradMem(num_group, weight_req, write_tensors.weights_grad);
    const auto& bias_grad_mem = BiasGradMem(bias_req, write_tensors.bias_grad);

    net_args.insert({DNNL_ARG_DIFF_DST, *OutGradMem(read_tensors.out_grad, out_grad_mem)});
    net_args.insert({DNNL_ARG_SRC, *DataMem(read_tensors.data)});
    net_args.insert({DNNL_ARG_DIFF_WEIGHTS, *weights_grad_mem.second});
    if (bias_grad_mem.second) {
      net_args.insert({DNNL_ARG_DIFF_BIAS, *bias_grad_mem.second});
    }

    // CommitOutput should run after RegisterPrimArgs for memory dependency
    DNNLStream::Get()->RegisterPrimArgs(*bwd_weights, net_args);
    CommitOutput(write_tensors.weights_grad, weights_grad_mem);
    if (bias_grad_mem.second) {
      CommitOutput(*write_tensors.bias_grad, bias_grad_mem);
    }
  }
}

DeconvDescCreator::DeconvDescCreator(const DeconvolutionParam& param,
                                     const NDArray& data,
                                     const NDArray& weights,
                                     const NDArray* const bias,
                                     const NDArray& out)
    : data_md(GetMemDesc(data)),
      weights_md(GetDeconvWeightsDesc(weights, param.num_group)),
      bias_md(bias ? GetMemDesc(*bias) : dnnl::memory::desc()),
      out_md(GetMemDesc(out)),
      strides(param.stride.ndim()),
      padding(param.pad.ndim()),
      dilates(param.dilate.ndim()) {
  CHECK_EQ(param.stride.ndim(), param.pad.ndim());
  CHECK_EQ(param.stride.ndim(), param.dilate.ndim());
  CHECK_GE(param.stride.ndim(), 1);
  CHECK_LE(param.stride.ndim(), 3);
  for (int i = 0; i < param.stride.ndim(); ++i) {
    strides[i] = param.stride[i];
    padding[i] = param.pad[i];
    dilates[i] = param.dilate[i] - 1;
  }
}

bool DeconvDescCreator::ImposePlainWherePadding(const size_t data_size,
                                                const size_t weights_size,
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
    weights_md       = IOLogicalSwapDesc(weights_md, num_gr);
    weights_md       = IOLogicalSwapDesc(GetDesc(weights_md, GetDefaultFormat(weights_md)), num_gr);
    return true;
  }
  return false;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
