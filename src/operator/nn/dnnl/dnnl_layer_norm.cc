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
 * \file dnnl_layer_norm.cc
 * \author: Bartosz Kuncer, bartosz.kuncer@intel.com
 */

#if MXNET_USE_ONEDNN == 1

#include "dnnl_layer_norm-inl.h"

namespace mxnet {
namespace op {

// Support for https://oneapi-src.github.io/oneDNN/v2.6/dev_guide_layer_normalization.html
bool SupportDNNLLayerNorm(const LayerNormParam& param, const std::vector<NDArray>& inputs) {
  const mxnet::TShape& shape = inputs[layernorm::kData].shape();

  // Native implementation (which can be found in function LayerNormCPU) is faster than oneDNN's one
  // for small tensors. Below is the heuristic based on measurements on clx machine deciding whether
  // the shape is better for oneDNN or native implementation.
  auto ShapeBetterForDNNL = [](const mxnet::TShape& shape) {
    constexpr size_t shapeLimit = 1024;
    return shape.Size() / shape[0] >= shapeLimit && shape[0] >= shapeLimit;
  };

  return (ShapeBetterForDNNL(shape) && GetRealAxis(param.axis, shape.ndim()) == shape.ndim() - 1) &&
         SupportDNNL<2, 5, DNNLTypeMode::FloatTypes>(inputs[layernorm::kData]) &&
         inputs[layernorm::kGamma].dtype() == mshadow::kFloat32 &&
         inputs[layernorm::kBeta].dtype() == mshadow::kFloat32;
}

void DNNLLayerNormForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  const auto& fwd             = DNNLLayerNormFwd::GetCached(param, ctx, inputs[layernorm::kData]);
  fwd.Execute(param, ctx, inputs, req[layernorm::kOut], outputs);
}

DNNLLayerNormFwd& DNNLLayerNormFwd::GetCached(const LayerNormParam& param,
                                              const OpContext& ctx,
                                              const NDArray& data) {
  using layernorm_fwd_map = std::unordered_map<LayerNormSignature, DNNLLayerNormFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local layernorm_fwd_map layer_norm_fwds;
#else
  static MX_THREAD_LOCAL layernorm_fwd_map layer_norm_fwds;
#endif

  LayerNormSignature key(param);
  key.AddSign(data);

  auto it = layer_norm_fwds.find(key);
  if (it == layer_norm_fwds.end()) {
    DNNLLayerNormFwd fwd(param, data);
    it = AddToCache(&layer_norm_fwds, key, fwd);
  }
  return it->second;
}

DNNLLayerNormFwd::DNNLLayerNormFwd(const LayerNormParam& param, const NDArray& data) {
  const dnnl::memory::desc data_md = data.GetDNNLData()->get_desc();
  fwd_pd                           = CreatePrimitiveDesc(param, data_md);
  fwd                              = std::make_shared<layernorm_fwd_t>(*fwd_pd);
}

std::shared_ptr<layernorm_fwd_pd_t> DNNLLayerNormFwd::CreatePrimitiveDesc(
    const LayerNormParam& param,
    const dnnl::memory::desc& src_md) {
  layernorm_fwd_t::desc fwd_desc(dnnl::prop_kind::forward_training,
                                 src_md,
                                 param.eps,
                                 dnnl::normalization_flags::use_scale_shift);
  dnnl::engine& engine = CpuEngine::Get()->get_engine();
  return std::make_shared<layernorm_fwd_pd_t>(fwd_desc, engine);
}

inline dnnl::memory::desc GetMeanVarDesc(const dnnl::memory::data_type& dtype,
                                         const mxnet::TShape& _shape) {
  const auto ndim = _shape.ndim();

  dnnl::memory::dims shape(ndim, 1), strides(ndim, 1);
  shape[0] = _shape[0];
  for (int i = ndim - 1; i > 0; --i) {
    shape[i]       = _shape[i];
    strides[i - 1] = strides[i] * shape[i];
  }

  return dnnl::memory::desc{shape, dtype, strides};
}

inline dnnl::memory GetScaleShiftMem(const NDArray& gamma, const NDArray& beta) {
  // oneDNN takes gamma and beta as one SCALE_SHIFT tensor when both scale and shift are used. In
  // mxnet scale is called gamma and shift is called beta.
  constexpr size_t gammaAndBeta = 2;
  CHECK_EQ(gamma.shape()[0], beta.shape()[0]);
  const dnnl::memory::desc scale_shift_md(dnnl::memory::dims{gammaAndBeta, gamma.shape()[0]},
                                          get_dnnl_type(gamma.dtype()),
                                          dnnl::memory::format_tag::nc);
  auto scale_shift_mem = dnnl::memory(scale_shift_md, CpuEngine::Get()->get_engine());
  char* ptr            = reinterpret_cast<char*>(scale_shift_mem.get_data_handle());
  const size_t bytes   = scale_shift_md.get_size() / gammaAndBeta;
  memcpy(ptr, gamma.data().dptr_, bytes);
  memcpy(ptr + bytes, beta.data().dptr_, bytes);
  return scale_shift_mem;
}

void DNNLLayerNormFwd::Execute(const LayerNormParam& param,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const OpReqType& req,
                               const std::vector<NDArray>& outputs) const {
  auto mean_var_md = GetMeanVarDesc(get_dnnl_type(outputs[layernorm::kMean].dtype()),
                                    outputs[layernorm::kMean].shape());
  auto mean_mem =
      dnnl_output_t(OutDataOp::Noop,
                    const_cast<NDArray&>(outputs[layernorm::kMean]).CreateDNNLData(&mean_var_md));
  auto variance_mem = dnnl_output_t(
      OutDataOp::Noop, const_cast<NDArray&>(outputs[layernorm::kStd]).CreateDNNLData(&mean_var_md));

  auto output_mem      = CreateDNNLMem(outputs[layernorm::kOut], fwd_pd->dst_desc(), req);
  auto scale_shift_mem = GetScaleShiftMem(inputs[layernorm::kGamma], inputs[layernorm::kBeta]);

  dnnl_args_map_t args = {{DNNL_ARG_SRC, *inputs[layernorm::kData].GetDNNLData()},
                          {DNNL_ARG_DST, *output_mem.second},
                          {DNNL_ARG_MEAN, *mean_mem.second},
                          {DNNL_ARG_VARIANCE, *variance_mem.second},
                          {DNNL_ARG_SCALE_SHIFT, scale_shift_mem}};

  DNNLStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(outputs[layernorm::kOut], output_mem);
  CommitOutput(outputs[layernorm::kMean], mean_mem);
  CommitOutput(outputs[layernorm::kStd], variance_mem);
  DNNLStream::Get()->Submit();
}

DNNLLayerNormBwd::DNNLLayerNormBwd(const LayerNormParam& param,
                                   const std::vector<NDArray>& inputs,
                                   const dnnl::memory::desc& data_md,
                                   const dnnl::memory::desc& diff_md)
    : fwd_pd(DNNLLayerNormFwd::CreatePrimitiveDesc(param, data_md)),
      bwd_pd(CreatePrimitiveDesc(param, data_md, diff_md, *fwd_pd)) {
  bwd = std::make_shared<layernorm_bwd_t>(*bwd_pd);
}

std::shared_ptr<layernorm_bwd_pd_t> DNNLLayerNormBwd::CreatePrimitiveDesc(
    const LayerNormParam& param,
    const dnnl::memory::desc& data_md,
    const dnnl::memory::desc& diff_md,
    const layernorm_fwd_pd_t& layernorm_fwd_pd) {
  layernorm_bwd_t::desc layernorm_bwd_desc(dnnl::prop_kind::backward,
                                           diff_md,
                                           data_md,
                                           param.eps,
                                           dnnl::normalization_flags::use_scale_shift);
  dnnl::engine& engine = CpuEngine::Get()->get_engine();
  return std::make_shared<layernorm_bwd_pd_t>(layernorm_bwd_desc, engine, layernorm_fwd_pd);
}

void DNNLLayerNormBwd::Execute(const std::vector<NDArray>& inputs,
                               const std::vector<NDArray>& outputs,
                               const std::vector<OpReqType>& req) const {
  auto scale_shift_mem =
      GetScaleShiftMem(inputs[layernorm::kBwdGamma], inputs[layernorm::kBwdBeta]);
  auto scale_shift_mem_desc = scale_shift_mem.get_desc();
  auto diff_weights_ndarray = NDArray(&scale_shift_mem_desc);
  const auto bytes          = inputs[layernorm::kBwdGamma].shape()[0] *
                     mshadow::mshadow_sizeof(inputs[layernorm::kBwdGamma].dtype());
  const auto diff_weights_ndaray_data_ptr_plus_bytes = reinterpret_cast<void*>(
      reinterpret_cast<std::uintptr_t>(diff_weights_ndarray.data().dptr_) + bytes);
  if (req[layernorm::kBwdGammaGrad] == kAddTo) {
    memcpy(
        diff_weights_ndarray.data().dptr_, outputs[layernorm::kBwdGammaGrad].data().dptr_, bytes);
    memcpy(diff_weights_ndaray_data_ptr_plus_bytes,
           outputs[layernorm::kBwdBetaGrad].data().dptr_,
           bytes);
  }
  dnnl_output_t diff_src_mem = CreateDNNLMem(
      outputs[layernorm::kBwdDataGrad], bwd_pd->diff_src_desc(), req[layernorm::kBwdDataGrad]);
  dnnl_output_t diff_weights_mem = CreateDNNLMem(
      diff_weights_ndarray, bwd_pd->diff_weights_desc(), req[layernorm::kBwdGammaGrad]);
  dnnl_args_map_t args = {{DNNL_ARG_DIFF_DST, *inputs[layernorm::kBwdOutGrad].GetDNNLData()},
                          {DNNL_ARG_SRC, *inputs[layernorm::kBwdData].GetDNNLData()},
                          {DNNL_ARG_SCALE_SHIFT, scale_shift_mem},
                          {DNNL_ARG_MEAN, *inputs[layernorm::kBwdMean].GetDNNLData()},
                          {DNNL_ARG_VARIANCE, *inputs[layernorm::kBwdStd].GetDNNLData()},
                          {DNNL_ARG_DIFF_SRC, *diff_src_mem.second},
                          {DNNL_ARG_DIFF_SCALE_SHIFT, *diff_weights_mem.second}};
  DNNLStream::Get()->RegisterPrimArgs(*bwd, args);
  CommitOutput(outputs[layernorm::kBwdDataGrad], diff_src_mem);
  CommitOutput(diff_weights_ndarray, diff_weights_mem);
  DNNLStream::Get()->Submit();
  // Commit scale_shift diff
  memcpy(outputs[layernorm::kBwdGammaGrad].data().dptr_, diff_weights_ndarray.data().dptr_, bytes);
  memcpy(outputs[layernorm::kBwdBetaGrad].data().dptr_,
         diff_weights_ndaray_data_ptr_plus_bytes,
         bytes);
}

DNNLLayerNormBwd& DNNLLayerNormBwd::GetCached(const LayerNormParam& param,
                                              const std::vector<NDArray>& inputs) {
  using layernorm_bwd_map = std::unordered_map<LayerNormSignature, DNNLLayerNormBwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local layernorm_bwd_map layer_norm_bwds;
#else
  static MX_THREAD_LOCAL layernorm_bwd_map layer_norm_bwds;
#endif
  LayerNormSignature key(param);
  key.AddSign(inputs[layernorm::kBwdOutGrad]);
  key.AddSign(inputs[layernorm::kBwdData]);
  key.AddSign(inputs[layernorm::kBwdGamma]);
  key.AddSign(inputs[layernorm::kBwdMean]);
  key.AddSign(inputs[layernorm::kBwdStd]);
  key.AddSign(inputs[layernorm::kBwdBeta]);

  auto it = layer_norm_bwds.find(key);
  if (it == layer_norm_bwds.end()) {
    const dnnl::memory::desc data_md = inputs[layernorm::kBwdData].GetDNNLData()->get_desc();
    const dnnl::memory::desc diff_md = inputs[layernorm::kBwdOutGrad].GetDNNLData()->get_desc();
    DNNLLayerNormBwd bwd(param, inputs, data_md, diff_md);
    it = AddToCache(&layer_norm_bwds, key, bwd);
  }
  return it->second;
}

void DNNLLayerNormBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  DNNLLayerNormBwd& bwd       = DNNLLayerNormBwd::GetCached(param, inputs);
  bwd.Execute(inputs, outputs, req);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
