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
 * \file mkldnn_batch_norm-inl.h
 * \brief
 * \author Tao Lv (tao.a.lv@intel.com)
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
typedef MKLDNNParamOpSign<BatchNormParam>                       MKLDNNBNSignature;

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
class MKLDNNBatchNormFwd {
 public:
  MKLDNNBatchNormFwd(const mxnet::NDArray &data, DType eps,
                     bool is_train, bool scale_shift,
                     bool global_stats, bool fix_gamma) :
                     _out_mean(nullptr), _out_var(nullptr),
                     _flag(0U), _fix_gamma(fix_gamma), _is_train(is_train),
                     _channels(data.shape()[1]), _eps(eps),
                     fwd(nullptr), data(nullptr), weight(nullptr),
                     out(nullptr), mean(nullptr), variance(nullptr) {
    _Init(data, scale_shift, global_stats);
  }

  ~MKLDNNBatchNormFwd() {}

  void SetDataHandle(const std::vector<OpReqType> &req,
                     const mxnet::NDArray         &data,
                     const mxnet::NDArray         &output,
                     const mxnet::TBlob           &moving_mean,
                     const mxnet::TBlob           &moving_var,
                     const mxnet::TBlob           *out_mean,
                     const mxnet::TBlob           *out_var,
                     const mxnet::TBlob           *gamma        = nullptr,
                     const mxnet::TBlob           *beta         = nullptr);

  void Execute();

 private:
  DType *_out_mean;
  DType *_out_var;
  unsigned _flag;
  bool _fix_gamma;
  bool _is_train;
  nnvm::dim_t _channels;
  DType _eps;

  std::shared_ptr<mkldnn::batch_normalization_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> out;
  std::shared_ptr<mkldnn::memory> mean;
  std::shared_ptr<mkldnn::memory> variance;

 private:
  void _Init(const mxnet::NDArray &data, bool scale_shift, bool global_stats);
  void _SetWeight(const mxnet::TBlob &gamma,
                  const mxnet::TBlob &beta,
                  const OpReqType    &req);
  void _SetMeanVar(const DType *imean,
                   const DType *ivar,
                   DType *omean,
                   DType *ovar);
};

template <typename DType>
static inline MKLDNNBatchNormFwd<DType> &GetBatchNormFwd(const BatchNormParam &param,
                                                         bool is_train,
                                                         const NDArray &data) {
  static thread_local std::unordered_map<MKLDNNBNSignature,
                                         MKLDNNBatchNormFwd<DType>,
                                         MKLDNNOpHash> bn_fwds;
  MKLDNNBNSignature key(param);
  key.AddSign(is_train);
  key.AddSign(data);

  auto it = bn_fwds.find(key);
  if (it == bn_fwds.end()) {
    MKLDNNBatchNormFwd<DType> fwd(data, param.eps, is_train, true,
                               param.use_global_stats, param.fix_gamma);
    auto ins_ret = bn_fwds.insert(
                std::pair<MKLDNNBNSignature, MKLDNNBatchNormFwd<DType> >(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

template <typename DType>
void MKLDNNBatchNormCompute(const OpContext &ctx, const BatchNormParam &param,
                            const std::vector<NDArray>   &in_data,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray>   &out_data,
                            const std::vector<NDArray>   &aux_states) {
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  const NDArray &data  = in_data[batchnorm::kData];
  CHECK(in_data[batchnorm::kGamma].IsDefault());
  CHECK(in_data[batchnorm::kBeta].IsDefault());
  auto gamma           = in_data[batchnorm::kGamma].data();
  auto beta            = in_data[batchnorm::kBeta].data();
  CHECK(aux_states[batchnorm::kMovingMean].IsDefault());
  CHECK(aux_states[batchnorm::kMovingVar].IsDefault());
  auto moving_mean     = aux_states[batchnorm::kMovingMean].data();
  auto moving_var      = aux_states[batchnorm::kMovingVar].data();
  const NDArray &out   = out_data[batchnorm::kOut];

  CHECK(out_data[batchnorm::kMean].IsDefault());
  CHECK(out_data[batchnorm::kVar].IsDefault());
  TBlob out_mean, out_var;
  TBlob *omean_ptr = nullptr;
  TBlob *ovar_ptr = nullptr;
  if (req[batchnorm::kMean] != kNullOp) {
    out_mean        = out_data[batchnorm::kMean].data();
    omean_ptr = &out_mean;
  }
  if (req[batchnorm::kVar] != kNullOp) {
    out_var         = out_data[batchnorm::kVar].data();
    ovar_ptr = &out_var;
  }

  MKLDNNBatchNormFwd<DType> &fwd = GetBatchNormFwd<DType>(param, ctx.is_train, data);
  fwd.SetDataHandle(req, data, out, moving_mean, moving_var,
                    omean_ptr, ovar_ptr, &gamma, &beta);
  fwd.Execute();
}

template <typename DType>
void MKLDNNBatchNormGradCompute(const OpContext &ctx, const BatchNormParam &param,
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
