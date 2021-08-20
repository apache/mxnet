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
 * \file mkldnn_layer_norm.cc
 */

#if MXNET_USE_ONEDNN == 1

#include "./mkldnn_layer_norm-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNLayerNorm(const LayerNormParam& param, const std::vector<NDArray>& inputs) {
  const mxnet::TShape& shape = inputs[layernorm::kData].shape();

  // Native implementation (which can be found in function LayerNormCPU) is faster than oneDNN's one
  // for small tensors. Below is the heuristic based on measurements on clx machine deciding whether
  // the shape is better for oneDNN or native implementation.
  auto ShapeBetterForMKLDNN = [](const mxnet::TShape& shape) {
    constexpr size_t shapeLimit = 1024;
    return shape.Size() / shape[0] >= shapeLimit && shape[0] >= shapeLimit;
  };

  return (ShapeBetterForMKLDNN(shape) &&
          (GetRealAxis(param.axis, shape.ndim()) == shape.ndim() - 1) && (shape.ndim() >= 2) &&
          (shape.ndim() <= 5) &&
          (inputs[layernorm::kData].dtype() == mshadow::kFloat32 ||
           inputs[layernorm::kData].dtype() == mshadow::kBfloat16) &&
          inputs[layernorm::kGamma].dtype() == mshadow::kFloat32 &&
          inputs[layernorm::kBeta].dtype() == mshadow::kFloat32);
}

void MKLDNNLayerNormForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  const auto& fwd             = MKLDNNLayerNormFwd::GetCached(param, ctx, inputs[layernorm::kData]);
  fwd.Execute(param, ctx, inputs, req[layernorm::kOut], outputs);
}

MKLDNNLayerNormFwd& MKLDNNLayerNormFwd::GetCached(const LayerNormParam& param,
                                                  const OpContext& ctx,
                                                  const NDArray& data) {
  using layernorm_fwd_map = std::unordered_map<LayerNormSignature, MKLDNNLayerNormFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local layernorm_fwd_map layer_norm_fwds;
#else
  static MX_THREAD_LOCAL layernorm_fwd_map layer_norm_fwds;
#endif

  LayerNormSignature key(param);
  key.AddSign(data);
  key.AddSign(param.eps);

  auto it = layer_norm_fwds.find(key);
  if (it == layer_norm_fwds.end()) {
    MKLDNNLayerNormFwd fwd(param, data);
    it = AddToCache(&layer_norm_fwds, key, fwd);
  }
  return it->second;
}

MKLDNNLayerNormFwd::MKLDNNLayerNormFwd(const LayerNormParam& param, const NDArray& data) {
  const mkldnn::memory::desc data_md = data.GetMKLDNNData()->get_desc();
  fwd_pd                             = CreatePrimitiveDesc(param, data_md);
  fwd = std::make_shared<mkldnn::layer_normalization_forward>(*fwd_pd);
}

std::shared_ptr<layernorm_fwd_pd_t> MKLDNNLayerNormFwd::CreatePrimitiveDesc(
    const LayerNormParam& param,
    const mkldnn::memory::desc& src_md) {
  layernorm_fwd_t::desc fwd_desc(mkldnn::prop_kind::forward_training,
                                 src_md,
                                 param.eps,
                                 dnnl::normalization_flags::use_scale_shift);
  mkldnn::engine& engine = CpuEngine::Get()->get_engine();
  return std::make_shared<layernorm_fwd_pd_t>(fwd_desc, engine);
}

inline mkldnn::memory::desc GetMeanVarDesc(const mkldnn::memory::data_type& dtype,
                                           const mxnet::TShape& _shape) {
  const auto ndim = _shape.ndim();

  mkldnn::memory::dims shape(ndim, 1), strides(ndim, 1);
  shape[0] = _shape[0];
  for (int i = ndim - 1; i > 0; --i) {
    shape[i]       = _shape[i];
    strides[i - 1] = strides[i] * shape[i];
  }

  return mkldnn::memory::desc{shape, dtype, strides};
}

inline mkldnn::memory GetScaleShiftMem(const NDArray& gamma, const NDArray& beta) {
  // OneDNN takes gamma and beta as one SCALE_SHIFT tensor when both scale and shift are used. In
  // mxnet scale is called gamma and shift is called beta.
  constexpr size_t gammaAndBeta = 2;
  CHECK_EQ(gamma.shape()[0], beta.shape()[0]);
  const mkldnn::memory::desc scale_shift_md(mkldnn::memory::dims{gammaAndBeta, gamma.shape()[0]},
                                            get_mkldnn_type(gamma.dtype()),
                                            mkldnn::memory::format_tag::nc);
  auto scale_shift_mem = mkldnn::memory(scale_shift_md, CpuEngine::Get()->get_engine());
  char* ptr            = reinterpret_cast<char*>(scale_shift_mem.get_data_handle());
  const size_t bytes   = scale_shift_md.get_size() / gammaAndBeta;
  memcpy(ptr, gamma.data().dptr_, bytes);
  memcpy(ptr + bytes, beta.data().dptr_, bytes);
  return scale_shift_mem;
}

void MKLDNNLayerNormFwd::Execute(const LayerNormParam& param,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const OpReqType& req,
                                 const std::vector<NDArray>& outputs) const {
  auto mean_var_md = GetMeanVarDesc(get_mkldnn_type(outputs[layernorm::kMean].dtype()),
                                    outputs[layernorm::kMean].shape());
  auto mean_mem    = mkldnn_output_t(
      OutDataOp::Noop,
      const_cast<NDArray&>(outputs[layernorm::kMean]).CreateMKLDNNData(mean_var_md));
  auto variance_mem =
      mkldnn_output_t(OutDataOp::Noop,
                      const_cast<NDArray&>(outputs[layernorm::kStd]).CreateMKLDNNData(mean_var_md));

  auto output_mem      = CreateMKLDNNMem(outputs[layernorm::kOut], fwd_pd->dst_desc(), req);
  auto scale_shift_mem = GetScaleShiftMem(inputs[layernorm::kGamma], inputs[layernorm::kBeta]);

  mkldnn_args_map_t args = {{MKLDNN_ARG_SRC, *inputs[layernorm::kData].GetMKLDNNData()},
                            {MKLDNN_ARG_DST, *output_mem.second},
                            {MKLDNN_ARG_MEAN, *mean_mem.second},
                            {MKLDNN_ARG_VARIANCE, *variance_mem.second},
                            {MKLDNN_ARG_SCALE_SHIFT, scale_shift_mem}};

  MKLDNNStream::Get()->RegisterPrimArgs(*fwd, args);
  CommitOutput(outputs[layernorm::kOut], output_mem);
  CommitOutput(outputs[layernorm::kMean], mean_mem);
  CommitOutput(outputs[layernorm::kStd], variance_mem);
  MKLDNNStream::Get()->Submit();
}

MKLDNNLayerNormBwd::MKLDNNLayerNormBwd(const LayerNormParam& param,
                                       const std::vector<NDArray>& inputs,
                                       const mkldnn::memory::desc& data_md,
                                       const mkldnn::memory::desc& diff_md)
    : fwd_pd(MKLDNNLayerNormFwd::CreatePrimitiveDesc(param, data_md)),
      bwd_pd(CreatePrimitiveDesc(param, data_md, diff_md, *fwd_pd)) {
  bwd = std::make_shared<layernorm_bwd_t>(*bwd_pd);
}

std::shared_ptr<layernorm_bwd_pd_t> MKLDNNLayerNormBwd::CreatePrimitiveDesc(
    const LayerNormParam& param,
    const mkldnn::memory::desc& data_md,
    const mkldnn::memory::desc& diff_md,
    const layernorm_fwd_pd_t& layernorm_fwd_pd) {
  layernorm_bwd_t::desc layernorm_bwd_desc(dnnl::prop_kind::backward,
                                           diff_md,
                                           data_md,
                                           param.eps,
                                           dnnl::normalization_flags::use_scale_shift);
  mkldnn::engine& engine = CpuEngine::Get()->get_engine();
  return std::make_shared<layernorm_bwd_pd_t>(layernorm_bwd_desc, engine, layernorm_fwd_pd);
}

void MKLDNNLayerNormBwd::Execute(const std::vector<NDArray>& inputs,
                                 const std::vector<NDArray>& outputs,
                                 const std::vector<OpReqType>& req) const {
  auto scale_shift_mem =
      GetScaleShiftMem(inputs[layernorm::kBwdGamma], inputs[layernorm::kBwdBeta]);
  auto diff_weights_ndarray = NDArray(scale_shift_mem.get_desc());
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
  mkldnn_output_t diff_src_mem = CreateMKLDNNMem(
      outputs[layernorm::kBwdDataGrad], bwd_pd->diff_src_desc(), req[layernorm::kBwdDataGrad]);
  mkldnn_output_t diff_weights_mem = CreateMKLDNNMem(
      diff_weights_ndarray, bwd_pd->diff_weights_desc(), req[layernorm::kBwdGammaGrad]);
  mkldnn_args_map_t args = {{MKLDNN_ARG_DIFF_DST, *inputs[layernorm::kBwdOutGrad].GetMKLDNNData()},
                            {MKLDNN_ARG_SRC, *inputs[layernorm::kBwdData].GetMKLDNNData()},
                            {MKLDNN_ARG_SCALE_SHIFT, scale_shift_mem},
                            {MKLDNN_ARG_MEAN, *inputs[layernorm::kBwdMean].GetMKLDNNData()},
                            {MKLDNN_ARG_VARIANCE, *inputs[layernorm::kBwdStd].GetMKLDNNData()},
                            {MKLDNN_ARG_DIFF_SRC, *diff_src_mem.second},
                            {MKLDNN_ARG_DIFF_SCALE_SHIFT, *diff_weights_mem.second}};
  MKLDNNStream::Get()->RegisterPrimArgs(*bwd, args);
  CommitOutput(outputs[layernorm::kBwdDataGrad], diff_src_mem);
  CommitOutput(diff_weights_ndarray, diff_weights_mem);
  MKLDNNStream::Get()->Submit();
  // Commit scale_shift diff
  memcpy(outputs[layernorm::kBwdGammaGrad].data().dptr_, diff_weights_ndarray.data().dptr_, bytes);
  memcpy(outputs[layernorm::kBwdBetaGrad].data().dptr_,
         diff_weights_ndaray_data_ptr_plus_bytes,
         bytes);
}

MKLDNNLayerNormBwd& MKLDNNLayerNormBwd::GetCached(const LayerNormParam& param,
                                                  const std::vector<NDArray>& inputs) {
  using layernorm_bwd_map = std::unordered_map<LayerNormSignature, MKLDNNLayerNormBwd, OpHash>;
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
  key.AddSign(param.eps);

  auto it = layer_norm_bwds.find(key);
  if (it == layer_norm_bwds.end()) {
    const mkldnn::memory::desc data_md = inputs[layernorm::kBwdData].GetMKLDNNData()->get_desc();
    const mkldnn::memory::desc diff_md = inputs[layernorm::kBwdOutGrad].GetMKLDNNData()->get_desc();
    MKLDNNLayerNormBwd bwd(param, inputs, data_md, diff_md);
    it = AddToCache(&layer_norm_bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNLayerNormBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<NDArray>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<NDArray>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  MKLDNNLayerNormBwd& bwd     = MKLDNNLayerNormBwd::GetCached(param, inputs);
  bwd.Execute(inputs, outputs, req);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
