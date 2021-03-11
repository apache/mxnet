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
  if (params.kernel.ndim() != 2) return false;
  return (input.dtype() == mshadow::kFloat32 || input.dtype() == mshadow::kBfloat16) &&
         input.shape().ndim() == 4;
}

/*############################### Forward ###############################*/

void MKLDNNDeconvolutionForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const auto &param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  const auto &tensors = MKLDNNDeconvFwd::Tensors(param.no_bias, inputs, outputs);
  MKLDNNDeconvFwd &fwd = MKLDNNDeconvFwd::GetCached(param, tensors);

  fwd.ControlWeightFormat(param.num_group, ctx.is_train, tensors.weight);
  fwd.Execute(param.num_group, req, tensors);
}

MKLDNNDeconvFwd::Tensors::Tensors(const NDArray &data, const NDArray &weight, const NDArray *bias,
                                  const NDArray &out)
    : data(data), weight(weight), bias(bias), out(out) {}

MKLDNNDeconvFwd::Tensors::Tensors(bool no_bias, const std::vector<NDArray> &inputs,
                                  const std::vector<NDArray> &outputs)
    : data(inputs[deconv::kData]),
      weight(inputs[deconv::kWeight]),
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
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(tensors.data);
  key.AddSign(tensors.weight);
  key.AddSign(tensors.out);
  if (tensors.bias) key.AddSign(*tensors.bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto fwd = MKLDNNDeconvFwd(param, tensors);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

std::shared_ptr<deconv_fwd_pd_t> MKLDNNDeconvFwd::MakePD(const DeconvolutionParam &param,
                                                         const Tensors &tensors) {
  DeconvDescCreator ddc(param, tensors.data, tensors.weight, tensors.bias, tensors.out);
  auto pd = std::make_shared<deconv_fwd_pd_t>(ddc.MakeFwdDesc(), ddc.engine);

  while (true) {
    size_t data_size = pd->src_desc().get_size();
    size_t weight_size = pd->weights_desc().get_size();
    size_t out_size = pd->dst_desc().get_size();
    if (ddc.CheckImpl(data_size, weight_size, out_size)) break;
    if (pd->next_impl()) continue;
    ddc.ImposePlainWherePadding(data_size, weight_size, out_size);
    *pd = deconv_fwd_pd_t(ddc.MakeFwdDesc(), ddc.engine);
  }
  return pd;
}

MKLDNNDeconvFwd::MKLDNNDeconvFwd(const DeconvolutionParam &param, const Tensors &tensors)
    : fwd_pd(MakePD(param, tensors)) {
  fwd = std::make_shared<deconv_fwd_t>(*fwd_pd);
}

void MKLDNNDeconvFwd::ControlWeightFormat(uint32_t num_group, bool is_train,
                                          const NDArray &weight) {
  if (is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it
    // to the default format for now.
    if (weight.IsMKLDNNData())
      // This asks the engine to change the layout of the weight array after it's used.
      weight.Reorder2DefaultAsync();
  } else {
    // For inference, we want to reorder the weight array so we don't need to
    // reorder data every time.
    if (weight.IsDefaultData()) {
      // We also need to modify the layout on the original weight array. The
      // data conversion happens after the weight array is used.
      weight.MKLDNNDataReorderAsync(IOLogicalSwapDesc(fwd_pd->weights_desc(), num_group));
    } else {
      CHECK(weight.GetMKLDNNData()->get_desc() ==
            IOLogicalSwapDesc(fwd_pd->weights_desc(), num_group));
    }
  }
}

void MKLDNNDeconvFwd::Execute(uint32_t num_group, const std::vector<OpReqType> &req,
                              const Tensors &tensors) {
  // MXNet (correctly) assumes that deconvolution is implemented using convolution primitives.
  // For that, we would pass input tensor in place of output and output tensor in place of
  // input (for appropriate convolution primitives: deconvolution forward = convolution backward
  // data, deconvolution backward data = convolution forward). Convolution primitive expects
  // weight tensor with shape (o, i, h, w), but because we swapped input and output tensors:
  // o = input_channels, i = output_channels. So in that case, deconvolution needs a weight
  // tensor with shape (input_channels, output_channels, h, w) and MXNet provides such tensor.
  //
  // MKLDNN's deconvolution primitive also expects weight tensor with shape (o, i, h, w),
  // but this time we don't swap input and output tensors, so o = output_channels, i =
  // input_channels, so the current weight tensor won't fit (when oihw != iohw). But actually,
  // underneath deconvolution MKLDNN also uses convolution, so even though it expects the weight
  // tensor with shape (o, i, h, w), it wants it in iohw format, so it's physical representation
  // match current weight tensor.
  //
  // So here we swap logical order of input and output dimensions for weight tensor just for
  // MKLDNN operations
  IOLogicalSwapMKLDNNMem(tensors.weight, num_group);
  {
    mkldnn_args_map_t net_args;
    auto out_mem = OutMem(req[deconv::kOut], tensors.out);

    net_args.insert({MKLDNN_ARG_SRC, *DataMem(tensors.data)});
    net_args.insert({MKLDNN_ARG_WEIGHTS, *WeightMem(num_group, tensors.weight)});
    net_args.insert({MKLDNN_ARG_DST, *out_mem.second});
    if (tensors.bias) net_args.insert({MKLDNN_ARG_BIAS, *BiasMem(*tensors.bias)});

    // CommitOutput Should run after RegisterPrimArgs for memory dependency
    MKLDNNStream::Get()->RegisterPrimArgs(*fwd, net_args);
    CommitOutput(tensors.out, out_mem);
    MKLDNNStream::Get()->Submit();
  }
  IOLogicalSwapMKLDNNMem(tensors.weight, num_group);  // swap back from oihw to iohw
}

const mkldnn::memory *MKLDNNDeconvFwd::DataMem(const NDArray &data) const {
  return data.GetMKLDNNDataReorder(fwd_pd->src_desc());
}

const mkldnn::memory *MKLDNNDeconvFwd::WeightMem(uint32_t num_group, const NDArray &weight) const {
  return GetWeights(weight, fwd_pd->weights_desc(), num_group);
}

const mkldnn::memory *MKLDNNDeconvFwd::BiasMem(const NDArray &bias) const {
  return bias.GetMKLDNNData();
}

mkldnn_output_t MKLDNNDeconvFwd::OutMem(OpReqType req, const NDArray &out) const {
  return CreateMKLDNNMem(out, fwd_pd->dst_desc(), req);
}

/*############################### Backward ###############################*/

void MKLDNNDeconvolutionBackward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  CHECK_NE(req[deconv::kWeight], kWriteInplace) << "cannot write weight inplace";

  TmpMemMgr::Get()->Init(ctx.requested[deconv::kTempSpace]);
  const auto &param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  const auto &rt = MKLDNNDeconvBwd::ReadTensors(param.no_bias, inputs);
  const auto &wt = MKLDNNDeconvBwd::WriteTensors(param.no_bias, outputs);
  MKLDNNDeconvBwd &bwd = MKLDNNDeconvBwd::GetCached(param, rt);

  bwd.Execute(param.num_group, req, rt, wt);
}

MKLDNNDeconvBwd::ReadTensors::ReadTensors(bool no_bias, const std::vector<NDArray> &inputs)
    : data(inputs[deconv::kData + 1]),
      weight(inputs[deconv::kWeight + 1]),
      bias(no_bias ? nullptr : &inputs[deconv::kBias + 1]),
      out_grad(inputs[deconv::kOut]) {}

MKLDNNDeconvBwd::WriteTensors::WriteTensors(bool no_bias, const std::vector<NDArray> &outputs)
    : data_grad(outputs[deconv::kData]),
      weight_grad(outputs[deconv::kWeight]),
      bias_grad(no_bias ? nullptr : &outputs[deconv::kBias]) {}

MKLDNNDeconvBwd &MKLDNNDeconvBwd::GetCached(const DeconvolutionParam &param,
                                            const ReadTensors &rt) {
  using mkldnn_deconv_bwd_map = std::unordered_map<DeconvSignature, MKLDNNDeconvBwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local mkldnn_deconv_bwd_map bwds;
#else
  static MX_THREAD_LOCAL mkldnn_deconv_bwd_map bwds;
#endif
  DeconvSignature key(param);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(rt.data);
  key.AddSign(rt.weight);
  key.AddSign(rt.out_grad);
  if (rt.bias) key.AddSign(*rt.bias);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto bwd = MKLDNNDeconvBwd(param, rt);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

std::shared_ptr<deconv_bwd_data_pd_t> MKLDNNDeconvBwd::MakeDataPD(const DeconvolutionParam &param,
                                                                  const ReadTensors &rt,
                                                                  const deconv_fwd_pd_t &fwd_pd) {
  DeconvDescCreator ddc(param, rt.data, rt.weight, nullptr, rt.out_grad);
  auto pd = std::make_shared<deconv_bwd_data_pd_t>(ddc.MakeBwdDataDesc(), ddc.engine, fwd_pd);

  while (true) {
    size_t data_size = pd->diff_src_desc().get_size();
    size_t weight_size = pd->weights_desc().get_size();
    size_t out_size = pd->diff_dst_desc().get_size();
    if (ddc.CheckImpl(data_size, weight_size, out_size)) break;
    if (pd->next_impl()) continue;
    ddc.ImposePlainWherePadding(data_size, weight_size, out_size);
    *pd = deconv_bwd_data_pd_t(ddc.MakeBwdDataDesc(), ddc.engine, fwd_pd);
  }
  return pd;
}

std::shared_ptr<deconv_bwd_weight_pd_t> MKLDNNDeconvBwd::MakeWeightsPD(
    const DeconvolutionParam &param, const ReadTensors &rt, const deconv_fwd_pd_t &fwd_pd) {
  DeconvDescCreator ddc(param, rt.data, rt.weight, rt.bias, rt.out_grad);
  auto pd = std::make_shared<deconv_bwd_weight_pd_t>(ddc.MakeBwdWeightDesc(), ddc.engine, fwd_pd);

  while (true) {
    size_t data_size = pd->src_desc().get_size();
    size_t weight_size = pd->diff_weights_desc().get_size();
    size_t out_size = pd->diff_dst_desc().get_size();
    if (ddc.CheckImpl(data_size, weight_size, out_size)) break;
    if (pd->next_impl()) continue;
    ddc.ImposePlainWherePadding(data_size, weight_size, out_size);
    *pd = deconv_bwd_weight_pd_t(ddc.MakeBwdWeightDesc(), ddc.engine, fwd_pd);
  }
  return pd;
}

MKLDNNDeconvBwd::MKLDNNDeconvBwd(const DeconvolutionParam &param, const ReadTensors &rt) {
  const auto fwd_pd = MKLDNNDeconvFwd::MakePD(  // TODO: use cached?
      param, MKLDNNDeconvFwd::Tensors(rt.data, rt.weight, rt.bias, rt.out_grad));
  bwd_data_pd = MakeDataPD(param, rt, *fwd_pd);
  bwd_weight_pd = MakeWeightsPD(param, rt, *fwd_pd);
  bwd_data = std::make_shared<deconv_bwd_t>(*bwd_data_pd);
  bwd_weight = std::make_shared<deconv_bwd_weight_t>(*bwd_weight_pd);
}

void MKLDNNDeconvBwd::Execute(uint32_t num_group, const std::vector<OpReqType> &req,
                              const ReadTensors &rt, const WriteTensors &wt) {
  // swaps are explained in MKLDNNDeconvFwd::Execute
  IOSwapWeightTensors(num_group, req, rt.weight, wt.weight_grad);
  {
    auto out_grad_mem = ScheduleBwdData(num_group, req, rt, wt);
    ScheduleBwdWeight(num_group, req, rt, wt, out_grad_mem);
    MKLDNNStream::Get()->Submit();
  }
  IOSwapWeightTensors(num_group, req, rt.weight, wt.weight_grad);
}

void MKLDNNDeconvBwd::IOSwapWeightTensors(uint32_t num_group, const std::vector<OpReqType> &req,
                                          const NDArray &weight, const NDArray &weight_grad) {
  if (req[deconv::kData]) IOLogicalSwapMKLDNNMem(weight, num_group);
  if (req[deconv::kWeight] || req[deconv::kBias]) IOLogicalSwapMKLDNNMem(weight_grad, num_group);
}

const mkldnn::memory *MKLDNNDeconvBwd::ScheduleBwdData(uint32_t num_group,
                                                       const std::vector<OpReqType> &req,
                                                       const ReadTensors &rt,
                                                       const WriteTensors &wt) {
  if (req[deconv::kData]) {
    mkldnn_args_map_t net_args;
    auto out_grad_mem = OutGradMem(rt.out_grad);
    auto data_grad_mem = DataGradMem(req[deconv::kData], wt.data_grad);

    net_args.insert({MKLDNN_ARG_DIFF_DST, *out_grad_mem});
    net_args.insert({MKLDNN_ARG_WEIGHTS, *WeightMem(num_group, rt.weight)});
    net_args.insert({MKLDNN_ARG_DIFF_SRC, *data_grad_mem.second});

    // CommitOutput Should run after RegisterPrimArgs for memory dependency
    MKLDNNStream::Get()->RegisterPrimArgs(*bwd_data, net_args);
    CommitOutput(wt.data_grad, data_grad_mem);
    return out_grad_mem;  // try reuse it in ScheduleBwdWeight
  }
  return nullptr;
}

void MKLDNNDeconvBwd::ScheduleBwdWeight(uint32_t num_group, const std::vector<OpReqType> &req,
                                        const ReadTensors &rt, const WriteTensors &wt,
                                        const mkldnn::memory *out_grad_mem) {
  if (req[deconv::kWeight] || req[deconv::kBias]) {
    mkldnn_args_map_t net_args;
    auto weight_grad_mem = WeightGradMem(num_group, req[deconv::kWeight], wt.weight_grad);
    auto bias_grad_mem = BiasGradMem(req[deconv::kBias], wt.bias_grad);

    net_args.insert({MKLDNN_ARG_DIFF_DST, *OutGradMem(rt.out_grad, out_grad_mem)});
    net_args.insert({MKLDNN_ARG_SRC, *DataMem(rt.data)});
    net_args.insert({MKLDNN_ARG_DIFF_WEIGHTS, *weight_grad_mem.second});
    if (bias_grad_mem.second) net_args.insert({MKLDNN_ARG_DIFF_BIAS, *bias_grad_mem.second});

    // CommitOutput Should run after RegisterPrimArgs for memory dependency
    MKLDNNStream::Get()->RegisterPrimArgs(*bwd_weight, net_args);
    CommitOutput(wt.weight_grad, weight_grad_mem);
    if (bias_grad_mem.second) CommitOutput(*wt.bias_grad, bias_grad_mem);
  }
}

const mkldnn::memory *MKLDNNDeconvBwd::DataMem(const NDArray &data) const {
  return data.GetMKLDNNDataReorder(bwd_weight_pd->src_desc());
}

const mkldnn::memory *MKLDNNDeconvBwd::WeightMem(uint32_t num_group, const NDArray &weight) const {
  return GetWeights(weight, bwd_data_pd->weights_desc(), num_group);
}

const mkldnn::memory *MKLDNNDeconvBwd::OutGradMem(const NDArray &out_grad) const {
  return out_grad.GetMKLDNNDataReorder(bwd_data_pd->diff_dst_desc());
}

const mkldnn::memory *MKLDNNDeconvBwd::OutGradMem(const NDArray &out_grad,
                                                  const mkldnn::memory *out_grad_mem) const {
  if (!out_grad_mem || bwd_data_pd->diff_dst_desc() != bwd_weight_pd->diff_dst_desc())
    return out_grad.GetMKLDNNDataReorder(bwd_weight_pd->diff_dst_desc());
  return out_grad_mem;
}

mkldnn_output_t MKLDNNDeconvBwd::DataGradMem(OpReqType req, const NDArray &data_grad) const {
  return CreateMKLDNNMem(data_grad, bwd_data_pd->diff_src_desc(), req);
}

mkldnn_output_t MKLDNNDeconvBwd::WeightGradMem(uint32_t num_group, OpReqType req,
                                               const NDArray &weight_grad) const {
  // CreateMKLDNNWeightGrad always creates a new tensor as IsDefaultFormat always fails (because
  // of the logical swap - explained in MKLDNNDeconvFwd::Execute). We try to reuse weight_grad
  // memory (which, when not swapped, is always in default format), so here we check if after a
  // swap, wei_md will have a default format
  const auto &wei_md = bwd_weight_pd->diff_weights_desc();
  if (req == OpReqType::kWriteTo && IsDefaultFormat(IOLogicalSwapDesc(wei_md, num_group)))
    return {OutDataOp::Noop, const_cast<NDArray &>(weight_grad).CreateMKLDNNData(wei_md)};
  return CreateMKLDNNWeightGrad(weight_grad, wei_md, req);
}

mkldnn_output_t MKLDNNDeconvBwd::BiasGradMem(OpReqType req, const NDArray *bias) const {
  return bias ? CreateMKLDNNMem(*bias, bwd_weight_pd->diff_bias_desc(), req)
              : mkldnn_output_t(OutDataOp::Noop, nullptr);
}

/*############################### DeconvDescCreator ###############################*/

DeconvDescCreator::DeconvDescCreator(const DeconvolutionParam &param, const NDArray &data,
                                     const NDArray &weight, const NDArray *bias, const NDArray &out)
    : data_md(GetMemDesc(data)),
      weight_md(GetDeconvWeightDesc(weight, param.num_group)),
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

void DeconvDescCreator::ImposePlainWherePadding(size_t data_size, size_t weight_size,
                                                size_t out_size) {
  if (data_size != GetMemDescSize(data_md)) {
    CHECK(data_md.data.format_kind == dnnl_format_kind_any) << "No implementation";
    data_md = GetDesc(data_md, GetDefaultFormat(data_md));
  } else if (out_size != GetMemDescSize(out_md)) {
    CHECK(out_md.data.format_kind == dnnl_format_kind_any) << "No implementation";
    out_md = GetDesc(out_md, GetDefaultFormat(out_md));
  } else if (weight_size != GetMemDescSize(weight_md)) {
    CHECK(weight_md.data.format_kind == dnnl_format_kind_any) << "No implementation";
    int num_groups = (weight_md.data.ndims > data_md.data.ndims) ? weight_md.data.dims[0] : 1;
    weight_md = IOLogicalSwapDesc(weight_md, num_groups);
    weight_md = IOLogicalSwapDesc(GetDesc(weight_md, GetDefaultFormat(weight_md)), num_groups);
  }
}

bool DeconvDescCreator::CheckImpl(size_t data_size, size_t weight_size, size_t out_size) const {
  // MKLDNN introduced padded formats since 0.15 which require more memory
  // compared to the actual size of the tensor. Currently, MKLDNN operators
  // still reuse memory from memory planning, so here we need to accept only a
  // kernel that has the expected memory size requirements (which is suboptimal)
  return (data_size == GetMemDescSize(data_md) && weight_size == GetMemDescSize(weight_md) &&
          out_size == GetMemDescSize(out_md));
}

deconv_fwd_t::desc DeconvDescCreator::MakeFwdDesc() const {
  // TODO: check if forward_training should be constant
  return deconv_fwd_t::desc(mkldnn::prop_kind::forward_training,
                            mkldnn::algorithm::deconvolution_direct, data_md, weight_md, bias_md,
                            out_md, strides, dilates, padding, padding);
}

deconv_bwd_t::desc DeconvDescCreator::MakeBwdDataDesc() const {
  return deconv_bwd_t::desc(mkldnn::algorithm::deconvolution_direct, data_md, weight_md, out_md,
                            strides, dilates, padding, padding);
}

deconv_bwd_weight_t::desc DeconvDescCreator::MakeBwdWeightDesc() const {
  return deconv_bwd_weight_t::desc(mkldnn::algorithm::deconvolution_direct, data_md, weight_md,
                                   bias_md, out_md, strides, dilates, padding, padding);
}

// Swaps the logical order of dimensions that in plain format would correspond to input and output
// channels (for example: oihw => iohw, iohw => oihw, goihw => giohw).
mkldnn::memory::desc IOLogicalSwapDesc(mkldnn::memory::desc desc, int num_groups) {
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

// Applies IOLogicalSwapDesc to MKLDNN memory of arr
void IOLogicalSwapMKLDNNMem(const NDArray &arr, int num_groups) {
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
