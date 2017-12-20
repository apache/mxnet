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

template <typename DType>
void MKLDNNBatchNormForward(const OpContext &ctx, const BatchNormParam &param,
                            const std::vector<NDArray>   &in_data,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray>   &out_data,
                            const std::vector<NDArray>   &aux_states) {
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  unsigned flags      = _GetFlags(in_data, aux_states, param, ctx.is_train);
  const NDArray &data = in_data[batchnorm::kData];

  auto data_mem       = data.GetMKLDNNData();
  auto fwd_pd         = _GetFwd(*data_mem, ctx.is_train, (DType) param.eps, flags);
  const NDArray &out  = out_data[batchnorm::kOut];

  // for output memory
  auto out_mem = const_cast<NDArray &>(out).CreateMKLDNNData(fwd_pd.dst_primitive_desc());

  // mxnet will always use scale shift.
  // But if fix_gamma is true, then all scale elements will be set to 1.0f
  if (flags & use_scale_shift) {
    const NDArray &gamma    = in_data[batchnorm::kGamma];
    const NDArray &beta     = in_data[batchnorm::kBeta];
    CHECK_EQ(gamma.storage_type(), mxnet::kDefaultStorage);
    CHECK_EQ(beta.storage_type(), mxnet::kDefaultStorage);

    // TODO(tao): how to reuse this memory?
    std::shared_ptr<const mkldnn::memory> weight_mem(
                        new mkldnn::memory(fwd_pd.weights_primitive_desc()));
    DType* weight_buf = reinterpret_cast<DType *>(weight_mem->get_data_handle());

    nnvm::dim_t channels_ = data.shape()[1];
    for (int i = 0; i < channels_; i++) {
      if (!param.fix_gamma) {
        weight_buf[i] = (gamma.data().dptr<DType>())[i];   // weight
      } else {
        weight_buf[i] = (DType)1.0f;
        if (IsBNWriting(req[batchnorm::kGamma]))
          (gamma.data().dptr<DType>())[i] = (DType)1.0f;
      }
    }

    for (int i = 0; i < channels_; i++) {
      weight_buf[channels_ + i] = (beta.data().dptr<DType>())[i];  // bias
    }

    if (!ctx.is_train) {
      DType* omean    = out_data[batchnorm::kMean].data().dptr<DType>();
      DType* ovar     = out_data[batchnorm::kVar].data().dptr<DType>();
      DType* inmean   = aux_states[batchnorm::kMovingMean].data().dptr<DType>();
      DType* invar    = aux_states[batchnorm::kMovingVar].data().dptr<DType>();
      // to align with origin implmentation: batch_norm.cc: L164
      for (int i = 0; i < channels_; i++) {
        omean[i] = (aux_states[batchnorm::kMovingMean].data().dptr<DType>())[i];
        ovar[i] = VARIANCE_TO_INVSTD(
                    (aux_states[batchnorm::kMovingVar].data().dptr<DType>())[i], param.eps);
      }
      std::shared_ptr<const mkldnn::memory> mean_m(
                      new mkldnn::memory(fwd_pd.mean_primitive_desc(), inmean));
      std::shared_ptr<const mkldnn::memory> var_m(
                      new mkldnn::memory(fwd_pd.variance_primitive_desc(), invar));
      auto bn = mkldnn::batch_normalization_forward(fwd_pd,
                                                    *data_mem,
                                                    mkldnn::primitive::at(*mean_m),
                                                    mkldnn::primitive::at(*var_m),
                                                    *weight_mem,
                                                    *out_mem);
      MKLDNNStream::Get()->RegisterPrim(bn);
      MKLDNNStream::Get()->Submit();
    } else {  // training
      const NDArray &outMean  = out_data[batchnorm::kMean];
      const NDArray &outVar   = out_data[batchnorm::kVar];
      CHECK_EQ(outMean.storage_type(), mxnet::kDefaultStorage);
      CHECK_EQ(outVar.storage_type(), mxnet::kDefaultStorage);
      DType* omean    = out_data[batchnorm::kMean].data().dptr<DType>();
      DType* ovar     = out_data[batchnorm::kVar].data().dptr<DType>();

      std::shared_ptr<const mkldnn::memory> mean_mem(
                      new mkldnn::memory(fwd_pd.mean_primitive_desc(), omean));
      std::shared_ptr<const mkldnn::memory> var_mem(
                      new mkldnn::memory(fwd_pd.variance_primitive_desc(), ovar));

      auto bn = mkldnn::batch_normalization_forward(fwd_pd,
                                                    mkldnn::primitive::at(*data_mem),
                                                    mkldnn::primitive::at(*weight_mem),
                                                    *out_mem,
                                                    *mean_mem,
                                                    *var_mem);
      MKLDNNStream::Get()->RegisterPrim(bn);
      MKLDNNStream::Get()->Submit();
      for (int i = 0; i < channels_; i++) {
        omean[i] = (reinterpret_cast<DType*>(mean_mem->get_data_handle()))[i];
        ovar[i]  = VARIANCE_TO_INVSTD(
                   (reinterpret_cast<DType*>(var_mem->get_data_handle()))[i], param.eps);
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

  CHECK_EQ(out_mean.storage_type(), mxnet::kDefaultStorage);
  CHECK_EQ(out_var.storage_type(), mxnet::kDefaultStorage);
  CHECK_EQ(moving_mean.storage_type(), mxnet::kDefaultStorage);
  CHECK_EQ(moving_var.storage_type(), mxnet::kDefaultStorage);

  auto data_mem  = data.GetMKLDNNData();
  auto diff_mem  = diff.GetMKLDNNData();
  if (diff_mem->get_primitive_desc() != data_mem->get_primitive_desc()) {
    data_mem = data.GetMKLDNNDataReorder(diff_mem->get_primitive_desc());
  }
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
