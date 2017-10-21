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
 * \file mkldnn_batch_norm.cc
 * \brief
 * \author Tao Lv
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BATCH_NORM_INL_H_

#if MXNET_USE_MKLDNN == 1
#include <vector>
#include <utility>
#include <mkldnn.hpp>
#include "../batch_norm-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))
namespace mxnet {
namespace op {

typedef mkldnn::batch_normalization_forward::primitive_desc     t_bn_f_pdesc;
typedef mkldnn::batch_normalization_forward::desc               t_bn_f_desc;
typedef mkldnn::batch_normalization_backward::primitive_desc    t_bn_b_pdesc;
typedef mkldnn::batch_normalization_backward::desc              t_bn_b_desc;

using mkldnn::use_global_stats;
using mkldnn::use_scale_shift;
using mkldnn::forward_training;
using mkldnn::forward_inference;

inline static unsigned _GetFlags(const std::vector<NDArray> &in_data,
                                 const std::vector<NDArray> &aux_states,
                                 const BatchNormParam &param, bool is_train) {
  unsigned flags = 0U;
  if (in_data.size() == 3U) {
    flags |= use_scale_shift;
  }

  // aux_states[0]: inMean
  // aux_states[1]: inVariance
  if (aux_states.size() == 2U && !is_train) {
    flags |= use_global_stats;
  }
  return flags;
}

template <typename DType>
inline static t_bn_f_pdesc _GetFwd(const mkldnn::memory &data_mem,
                                   bool is_train,
                                   DType eps,
                                   unsigned flags) {
  auto data_mpd   = data_mem.get_primitive_desc();
  auto data_md    = data_mpd.desc();
  auto engine     = CpuEngine::Get()->get_engine();

  if (is_train) {
    t_bn_f_desc bnFwd_desc(forward_training, data_md, eps, flags);
    return t_bn_f_pdesc(bnFwd_desc, engine);
  } else {
    t_bn_f_desc bnFwd_desc(forward_inference, data_md, eps, flags);
    return t_bn_f_pdesc(bnFwd_desc, engine);
  }
}

template <typename DType>
inline static t_bn_b_pdesc _GetBwd(const mkldnn::memory &data_mem,
                                   const mkldnn::memory &diff_mem,
                                   DType eps,
                                   unsigned flags) {
  auto data_mpd   = data_mem.get_primitive_desc();
  auto data_md    = data_mpd.desc();
  auto diff_mpd   = diff_mem.get_primitive_desc();
  auto diff_md    = diff_mpd.desc();
  auto engine     = CpuEngine::Get()->get_engine();

  t_bn_b_desc  bnBwd_desc(mkldnn::prop_kind::backward, diff_md, data_md, eps, flags);
  return t_bn_b_pdesc(bnBwd_desc, engine, _GetFwd(data_mem, true, eps, flags));
}

typedef MKLDNNParamOpSign<BatchNormParam> MKLDNNBNSignature;

class MKLDNNBNForward {
  std::shared_ptr<const mkldnn::memory> data_m;
  std::shared_ptr<const mkldnn::memory> weight_m;
  std::shared_ptr<const mkldnn::memory> out_m;
  std::shared_ptr<const mkldnn::memory> mean_m;
  std::shared_ptr<const mkldnn::memory> var_m;
  std::shared_ptr<mkldnn::batch_normalization_forward> fwd;
  bool is_train;
  t_bn_f_pdesc pd;

 public:
  MKLDNNBNForward(const t_bn_f_pdesc &_pd, bool is_train): pd(_pd) {
    weight_m.reset(new mkldnn::memory(pd.weights_primitive_desc()));
    this->is_train = is_train;
  }

  const mkldnn::memory &GetWeight() const {
    return *weight_m;
  }

  const t_bn_f_pdesc &GetPd() const {
    return pd;
  }

  const mkldnn::memory &GetMean() const {
    return *mean_m;
  }

  const mkldnn::memory &GetVar() const {
    return *var_m;
  }

  void SetDataHandle(const NDArray &data, const NDArray &mean,
                     const NDArray &var, const mkldnn::memory &out) {
    auto _data = data.GetMKLDNNData();
    if (data_m) {
      data_m->set_data_handle(_data->get_data_handle());
    } else {
      data_m.reset(new mkldnn::memory(_data->get_primitive_desc(),
                                      _data->get_data_handle()));
    }
    if (out_m) {
      out_m->set_data_handle(out.get_data_handle());
    } else {
      out_m.reset(new mkldnn::memory(out.get_primitive_desc(),
                                     out.get_data_handle()));
    }
    auto mean_ptr = mean.data().dptr_;
    if (mean_m) {
      mean_m->set_data_handle(mean_ptr);
    } else {
      mean_m.reset(new mkldnn::memory(pd.mean_primitive_desc(),
                                      mean_ptr));
    }
    auto var_ptr = var.data().dptr_;
    if (var_m) {
      var_m->set_data_handle(var_ptr);
    } else {
      var_m.reset(new mkldnn::memory(pd.variance_primitive_desc(),
                                     var_ptr));
    }

    if (fwd == nullptr) {
      if (!is_train)
        fwd.reset(new mkldnn::batch_normalization_forward(
                pd, *data_m, mkldnn::primitive::at(*mean_m),
                mkldnn::primitive::at(*var_m), *weight_m, *out_m));
      else
        fwd.reset(new mkldnn::batch_normalization_forward(
                pd, mkldnn::primitive::at(*data_m),
                mkldnn::primitive::at(*weight_m), *out_m,
                *mean_m, *var_m));
    }
  }

  const mkldnn::batch_normalization_forward &GetFwd() const {
    return *fwd;
  }
};

template<typename DType>
static MKLDNNBNForward &GetBNForward(const BatchNormParam& param,
                                     const OpContext &ctx, const NDArray &in_data,
                                     unsigned flags) {
  static thread_local std::unordered_map<MKLDNNBNSignature, MKLDNNBNForward, MKLDNNOpHash> fwds;
  MKLDNNBNSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(in_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto fwd_pd = _GetFwd(*in_data.GetMKLDNNData(), ctx.is_train,
                          (DType) param.eps, flags);
    MKLDNNBNForward fwd(fwd_pd, ctx.is_train);
    auto ins_ret = fwds.insert(std::pair<MKLDNNBNSignature, MKLDNNBNForward>(
            key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

template <typename DType>
void MKLDNNBatchNormForward(const OpContext &ctx, const BatchNormParam &param,
                            const std::vector<NDArray>   &in_data,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray>   &out_data,
                            const std::vector<NDArray>   &aux_states) {
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  unsigned flags      = _GetFlags(in_data, aux_states, param, ctx.is_train);
  const NDArray &data = in_data[batchnorm::kData];

  auto &fwd = GetBNForward<DType>(param, ctx, data, flags);
  const NDArray &out  = out_data[batchnorm::kOut];

  // for output memory
  auto out_mem = const_cast<NDArray &>(out).CreateMKLDNNData(fwd.GetPd().dst_primitive_desc());

  // mxnet will always use scale shift.
  // But if fix_gamma is true, then all scale elements will be set to 1.0f
  if (flags & use_scale_shift) {
    const NDArray &gamma    = in_data[batchnorm::kGamma];
    const NDArray &beta     = in_data[batchnorm::kBeta];
    CHECK_EQ(gamma.storage_type(), mxnet::kDefaultStorage);
    CHECK_EQ(beta.storage_type(), mxnet::kDefaultStorage);

    const mkldnn::memory &weight_mem = fwd.GetWeight();
    DType* weight_buf = reinterpret_cast<DType *>(weight_mem.get_data_handle());

    nnvm::dim_t channels_ = data.shape()[1];
    CHECK(weight_mem.get_primitive_desc().get_size() == channels_ * sizeof(DType) * 2);
    DType* weight_ptr = gamma.data().dptr<DType>();
    DType* bias_ptr = beta.data().dptr<DType>();
    if (!param.fix_gamma) {
#pragma omp parallel for simd
      for (int i = 0; i < channels_; i++) {
        weight_buf[i] = weight_ptr[i];
        weight_buf[channels_ + i] = bias_ptr[i];  // bias
      }
    } else if (IsBNWriting(req[batchnorm::kGamma])) {
#pragma omp parallel for simd
      for (int i = 0; i < channels_; i++) {
        weight_buf[i] = (DType)1.0f;
        weight_ptr[i] = (DType)1.0f;
        weight_buf[channels_ + i] = bias_ptr[i];  // bias
      }
    } else {
#pragma omp parallel for simd
      for (int i = 0; i < channels_; i++) {
        weight_buf[i] = (DType)1.0f;
        weight_buf[channels_ + i] = bias_ptr[i];  // bias
      }
    }

    if (!ctx.is_train) {
      DType* omean    = out_data[batchnorm::kMean].data().dptr<DType>();
      DType* ovar     = out_data[batchnorm::kVar].data().dptr<DType>();
      DType* inmean   = aux_states[batchnorm::kMovingMean].data().dptr<DType>();
      DType* invar    = aux_states[batchnorm::kMovingVar].data().dptr<DType>();
      // to align with origin implmentation: batch_norm.cc: L164
#pragma omp parallel for simd
      for (int i = 0; i < channels_; i++) {
        omean[i] = inmean[i];
        ovar[i] = VARIANCE_TO_INVSTD(invar[i], param.eps);
      }

      fwd.SetDataHandle(data, aux_states[batchnorm::kMovingMean],
                        aux_states[batchnorm::kMovingVar],
                        *out_mem);
      MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());
      MKLDNNStream::Get()->Submit();
    } else {  // training
      const NDArray &outMean  = out_data[batchnorm::kMean];
      const NDArray &outVar   = out_data[batchnorm::kVar];
      DType* omean    = outMean.data().dptr<DType>();
      DType* ovar     = outVar.data().dptr<DType>();

      fwd.SetDataHandle(data, outMean, outVar, *out_mem);
      MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());
      MKLDNNStream::Get()->Submit();
      DType* mean_mem_ptr = reinterpret_cast<DType*>(fwd.GetMean().get_data_handle());
      DType* var_mem_ptr  = reinterpret_cast<DType*>(fwd.GetVar().get_data_handle());
#pragma omp parallel for simd
      for (int i = 0; i < channels_; i++) {
        omean[i] = mean_mem_ptr[i];
        ovar[i]  = VARIANCE_TO_INVSTD(var_mem_ptr[i], param.eps);
      }
    }
  } else {  // no input gamma and beta
      LOG(FATAL) << "MKLDNN batch normalization: should not reach here ...";
  }
}

template <typename DType>
void MKLDNNBatchNormBackward(const OpContext &ctx, const BatchNormParam &param,
                             const std::vector<NDArray>    &out_grad,
                             const std::vector<NDArray>    &in_data,
                             const std::vector<NDArray>    &out_data,
                             const std::vector<OpReqType>  &req,
                             const std::vector<NDArray>    &in_grad,
                             const std::vector<NDArray>    &aux_states) {
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  CHECK_EQ(out_grad.size(), param.output_mean_var ? 3U : 1U);
  CHECK_EQ(in_data.size(), 3U);
  CHECK_EQ(out_data.size(), 3U);
  CHECK_EQ(in_grad.size(), 3U);
  unsigned flags = _GetFlags(in_data, aux_states, param, ctx.is_train);

  const NDArray &data         = in_data[batchnorm::kData];
  const NDArray &diff         = out_grad[batchnorm::kOut];
  const NDArray &gradIn       = in_grad[batchnorm::kData];
  const NDArray &moving_mean  = aux_states[batchnorm::kMovingMean];
  const NDArray &moving_var   = aux_states[batchnorm::kMovingVar];
  const NDArray &out_mean     = out_data[batchnorm::kMean];
  const NDArray &out_var      = out_data[batchnorm::kVar];

  CHECK(out_mean.IsDefault());
  CHECK(out_var.IsDefault());
  CHECK(moving_mean.IsDefault());
  CHECK(moving_var.IsDefault());

  auto data_mem  = data.GetMKLDNNData();
  auto diff_mem  = diff.GetMKLDNNData();
  // MKLDNN batchnorm should run on special layouts. If one of them isn't, we
  // should reorder them.
  if (data.IsDefault())
    data_mem = data.GetMKLDNNDataReorder(diff_mem->get_primitive_desc());
  else if (diff.IsDefault())
    diff_mem = diff.GetMKLDNNDataReorder(data_mem->get_primitive_desc());
  auto bwd_pd = _GetBwd(*data_mem, *diff_mem, param.eps, flags);
  auto gradi_mem = const_cast<NDArray &>(gradIn).CreateMKLDNNData(data_mem->get_primitive_desc());

  if (flags & use_scale_shift) {
    const NDArray &gamma    = in_data[batchnorm::kGamma];
    const NDArray &beta     = in_data[batchnorm::kBeta];
    // TODO(tao): how to reuse this memory?
    std::shared_ptr<const mkldnn::memory> weight_mem(
                    new mkldnn::memory(bwd_pd.weights_primitive_desc()));

    DType* weight_buf = reinterpret_cast<DType *>(weight_mem->get_data_handle());
    nnvm::dim_t channels_ = data.shape()[1];
    for (int i = 0; i < channels_; i++) {
      if (!param.fix_gamma)
        weight_buf[i] = (gamma.data().dptr<DType>())[i];   // weight
      else
        weight_buf[i] = (DType)1.0f;
    }

    for (int i = 0; i < channels_; i++) {
      weight_buf[channels_ + i] = (beta.data().dptr<DType>())[i];  // bias
    }

    std::shared_ptr<const mkldnn::memory> gradw_mem(
                    new mkldnn::memory(bwd_pd.diff_weights_primitive_desc()));
    // training but no input mean and variance
    if (ctx.is_train && !param.use_global_stats) {
      DType* moving_mean_ptr  = reinterpret_cast<DType *>(moving_mean.data().dptr<DType>());
      DType* moving_var_ptr   = reinterpret_cast<DType *>(moving_var.data().dptr<DType>());
      DType* out_mean_ptr     = reinterpret_cast<DType *>(out_mean.data().dptr<DType>());
      DType* out_var_ptr      = reinterpret_cast<DType *>(out_var.data().dptr<DType>());
      mkldnn::memory var_mem(bwd_pd.variance_primitive_desc());
      DType *tmp_var_ptr = reinterpret_cast<DType *>(var_mem.get_data_handle());

      DType minus_mom = (1.0f - param.momentum);
      for (int i = 0; i < channels_; i++) {
        moving_mean_ptr[i] = moving_mean_ptr[i] * param.momentum +
                             out_mean_ptr[i] * minus_mom;
        float variance = INVSTD_TO_VARIANCE(out_var_ptr[i], param.eps);
        tmp_var_ptr[i] = variance;
        moving_var_ptr[i] = moving_var_ptr[i] * param.momentum +
                            variance * minus_mom;
      }

      std::shared_ptr<const mkldnn::memory> out_mean_mem(
                      new mkldnn::memory(bwd_pd.mean_primitive_desc(), out_mean_ptr));
      std::shared_ptr<const mkldnn::memory> out_var_mem(
                      new mkldnn::memory(bwd_pd.variance_primitive_desc(), out_var_ptr));

      auto bn_bwd = mkldnn::batch_normalization_backward(bwd_pd,
                                                         *data_mem,
                                                         mkldnn::primitive::at(*out_mean_mem),
                                                         mkldnn::primitive::at(var_mem),
                                                         *diff_mem,
                                                         *weight_mem,
                                                         *gradi_mem,
                                                         *gradw_mem);

      MKLDNNStream::Get()->RegisterPrim(bn_bwd);
      MKLDNNStream::Get()->Submit();
    } else {
      std::shared_ptr<const mkldnn::memory> imean_mem(
                      new mkldnn::memory(bwd_pd.mean_primitive_desc(),
                      moving_mean.data().dptr<DType>()));
      std::shared_ptr<const mkldnn::memory> ivar_mem(
                      new mkldnn::memory(bwd_pd.variance_primitive_desc(),
                      moving_var.data().dptr<DType>()));
      auto bn_bwd = mkldnn::batch_normalization_backward(bwd_pd,
                                                         *data_mem,
                                                         mkldnn::primitive::at(*imean_mem),
                                                         mkldnn::primitive::at(*ivar_mem),
                                                         *diff_mem,
                                                         *weight_mem,
                                                         *gradi_mem,
                                                         *gradw_mem);

      MKLDNNStream::Get()->RegisterPrim(bn_bwd);
      MKLDNNStream::Get()->Submit();
    }

    // copy data from gradw_mem to in_grad[1] and in_grad[2]
    DType* gw_buf = reinterpret_cast<DType *>(gradw_mem->get_data_handle());
    for (int i = 0; i < channels_; i++) {
      if (!param.fix_gamma)
        (in_grad[1].data().dptr<DType>())[i] = gw_buf[i];
      else
        (in_grad[1].data().dptr<DType>())[i] = 0.0f;
    }

    for (int i = 0; i < channels_; i++) {
      (in_grad[2].data().dptr<DType>())[i] = gw_buf[i + channels_];
    }
  } else {
    LOG(FATAL) << "MKLDNN batch normalization backward: should not reach here ...";
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BATCH_NORM_INL_H_
