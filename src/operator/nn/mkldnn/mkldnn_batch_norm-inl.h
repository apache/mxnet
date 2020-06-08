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

#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/std::sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))
namespace mxnet {
namespace op {

typedef mkldnn::batch_normalization_forward::primitive_desc     t_bn_f_pdesc;
typedef mkldnn::batch_normalization_forward::desc               t_bn_f_desc;
typedef mkldnn::batch_normalization_backward::primitive_desc    t_bn_b_pdesc;
typedef mkldnn::batch_normalization_backward::desc              t_bn_b_desc;

inline static mkldnn::normalization_flags _GetFlags(const std::vector<NDArray> &in_data,
                                 const std::vector<NDArray> &aux_states,
                                 const BatchNormParam &param, bool is_train_and_not_global_stats) {
  mkldnn::normalization_flags flags = static_cast<mkldnn::normalization_flags>(0U);
  if (in_data.size() == 3U) {
    flags |=  mkldnn::normalization_flags::use_scale_shift;
  }

  // aux_states[0]: inMean
  // aux_states[1]: inVariance
  if (aux_states.size() == 2U && !is_train_and_not_global_stats) {
    flags |=  mkldnn::normalization_flags::use_global_stats;
  }
  return flags;
}

inline static t_bn_f_pdesc _GetFwd(const mkldnn::memory &data_mem,
                                   bool is_train,
                                   float eps,
                                   mkldnn::normalization_flags flags) {
  auto data_md   = data_mem.get_desc();
  auto engine    = CpuEngine::Get()->get_engine();

  if (is_train) {
    t_bn_f_desc bnFwd_desc(mkldnn::prop_kind::forward_training, data_md, eps, flags);
    return t_bn_f_pdesc(bnFwd_desc, engine);
  } else {
    t_bn_f_desc bnFwd_desc(mkldnn::prop_kind::forward_inference, data_md, eps, flags);
    return t_bn_f_pdesc(bnFwd_desc, engine);
  }
}

inline static t_bn_b_pdesc _GetBwd(const mkldnn::memory &data_mem,
                                   const mkldnn::memory &diff_mem,
                                   float eps,
                                   mkldnn::normalization_flags flags) {
  auto data_md    = data_mem.get_desc();
  auto diff_md    = diff_mem.get_desc();
  auto engine     = CpuEngine::Get()->get_engine();

  t_bn_b_desc  bnBwd_desc(mkldnn::prop_kind::backward, diff_md, data_md, eps, flags);
  return t_bn_b_pdesc(bnBwd_desc, engine, _GetFwd(data_mem, true, eps, flags));
}

typedef ParamOpSign<BatchNormParam> MKLDNNBNSignature;

class MKLDNNBNForward {
  std::shared_ptr<const mkldnn::memory> weight_m;
  std::shared_ptr<mkldnn::batch_normalization_forward> fwd;
  bool is_train_and_not_global_stats;
  t_bn_f_pdesc pd;

 public:
  MKLDNNBNForward(const t_bn_f_pdesc &_pd, bool is_train_and_not_global_stats): pd(_pd) {
    weight_m.reset(new mkldnn::memory(pd.weights_desc(), CpuEngine::Get()->get_engine()));
    fwd.reset(new mkldnn::batch_normalization_forward(pd));
    this->is_train_and_not_global_stats = is_train_and_not_global_stats;
  }

  const mkldnn::memory &GetWeight() const {
    return *weight_m;
  }

  const t_bn_f_pdesc &GetPd() const {
    return pd;
  }

  const mkldnn::batch_normalization_forward &GetFwd() const {
    return *fwd;
  }
};

template<typename DType>
static MKLDNNBNForward &GetBNForward(const BatchNormParam& param,
                                     const OpContext &ctx, const mkldnn::memory *data_mem,
                                     mkldnn::normalization_flags flags) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNBNSignature, MKLDNNBNForward, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNBNSignature, MKLDNNBNForward, OpHash> fwds;
#endif
  MKLDNNBNSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(*data_mem);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto fwd_pd = _GetFwd(*data_mem, ctx.is_train,
                          param.eps, flags);
    MKLDNNBNForward fwd(fwd_pd, ctx.is_train && !param.use_global_stats);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

template<typename DType>
static MKLDNNBNForward &GetBNForward(const BatchNormParam& param,
                                     const OpContext &ctx, const NDArray &in_data,
                                     mkldnn::normalization_flags flags) {
  return GetBNForward<DType>(param, ctx, in_data.GetMKLDNNData(), flags);
}

template <typename DType>
void MKLDNNBatchNormForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                            const std::vector<NDArray> &inputs, const std::vector<OpReqType> &req,
                            const std::vector<NDArray> &outputs) {
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  const std::vector<NDArray> in_data(inputs.begin(), inputs.begin() + batchnorm::kInMovingMean);
  const std::vector<NDArray> aux_states(inputs.begin() + batchnorm::kInMovingMean, inputs.end());
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  mkldnn::normalization_flags flags = _GetFlags(in_data,
                                                aux_states,
                                                param,
                                                ctx.is_train && !param.use_global_stats);
  const NDArray &data = in_data[batchnorm::kData];
  auto &fwd = GetBNForward<DType>(param, ctx, data, flags);
  const NDArray &out = outputs[batchnorm::kOut];

  // for output memory
  auto out_mem = const_cast<NDArray &>(out).CreateMKLDNNData(fwd.GetPd().dst_desc());

  // mxnet will always use scale shift.
  // But if fix_gamma is true, then all scale elements will be set to 1.0f
  if (static_cast<int>(flags) & static_cast<int>(mkldnn::normalization_flags::use_scale_shift)) {
    const NDArray &gamma    = in_data[batchnorm::kGamma];
    const NDArray &beta     = in_data[batchnorm::kBeta];
    CHECK_EQ(gamma.storage_type(), mxnet::kDefaultStorage);
    CHECK_EQ(beta.storage_type(), mxnet::kDefaultStorage);

    const mkldnn::memory &weight_mem = fwd.GetWeight();
    float* weight_buf = reinterpret_cast<float *>(weight_mem.get_data_handle());

    nnvm::dim_t channels_ = data.shape()[1];
    CHECK(weight_mem.get_desc().get_size() == channels_ * sizeof(float) * 2);
    float* weight_ptr = gamma.data().dptr<float>();
    float* bias_ptr = beta.data().dptr<float>();
    const size_t copy_size = sizeof(weight_buf[0]) * channels_;
    if (!param.fix_gamma) {
      memcpy(weight_buf, weight_ptr, copy_size);
      memcpy(&weight_buf[channels_], bias_ptr, copy_size);
    } else if (IsBNWriting(req[batchnorm::kGamma])) {
      for (int i = 0; i < channels_; i++) {
        weight_buf[i] = 1.0f;
        weight_ptr[i] = 1.0f;
        weight_buf[channels_ + i] = bias_ptr[i];  // bias
      }
    } else {
      for (int i = 0; i < channels_; i++) {
        weight_buf[i] = 1.0f;
        weight_buf[channels_ + i] = bias_ptr[i];  // bias
      }
    }

    mkldnn_args_map_t net_args;
    net_args[MKLDNN_ARG_SRC] = *data.GetMKLDNNData();
    net_args[MKLDNN_ARG_SCALE_SHIFT] = weight_mem;
    net_args[MKLDNN_ARG_DST] = *out_mem;

    if (!ctx.is_train || param.use_global_stats) {
      float* omean    = outputs[batchnorm::kMean].data().dptr<float>();
      float* ovar     = outputs[batchnorm::kVar].data().dptr<float>();
      float* inmean   = aux_states[batchnorm::kMovingMean].data().dptr<float>();
      float* invar    = aux_states[batchnorm::kMovingVar].data().dptr<float>();
      // to align with origin implmentation: batch_norm.cc: L164
      for (int i = 0; i < channels_; i++) {
        omean[i] = inmean[i];
        ovar[i] = VARIANCE_TO_INVSTD(invar[i], param.eps);
      }
      net_args[MKLDNN_ARG_MEAN] = *(aux_states[batchnorm::kMovingMean].GetMKLDNNData());
      net_args[MKLDNN_ARG_VARIANCE] = *(aux_states[batchnorm::kMovingVar].GetMKLDNNData());
      MKLDNNStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
      MKLDNNStream::Get()->Submit();
    } else {  // training
      const NDArray &outMean  = outputs[batchnorm::kMean];
      const NDArray &outVar   = outputs[batchnorm::kVar];
      net_args[MKLDNN_ARG_MEAN] = *(outMean.GetMKLDNNData());
      net_args[MKLDNN_ARG_VARIANCE] = *(outVar.GetMKLDNNData());
      MKLDNNStream::Get()->RegisterPrimArgs(fwd.GetFwd(), net_args);
      MKLDNNStream::Get()->Submit();

      float* ovar = outVar.data().dptr<float>();
      for (int i = 0; i < channels_; i++) {
        ovar[i] = VARIANCE_TO_INVSTD(ovar[i], param.eps);
      }
    }
  } else {  // no input gamma and beta
    LOG(FATAL) << "MKLDNN batch normalization: should not reach here ...";
  }
}

class MKLDNNBNBackward {
  std::shared_ptr<mkldnn::batch_normalization_backward> bwd;
  const std::shared_ptr<mkldnn::memory> weight_m;
  const std::shared_ptr<mkldnn::memory> gradw_m;

 public:
  const t_bn_b_pdesc pd;

  explicit MKLDNNBNBackward(const t_bn_b_pdesc &_pd)
      : weight_m(new mkldnn::memory(_pd.weights_desc(), CpuEngine::Get()->get_engine())),
        gradw_m(new mkldnn::memory(_pd.diff_weights_desc(), CpuEngine::Get()->get_engine())),
        pd(_pd) {
    bwd.reset(new mkldnn::batch_normalization_backward(pd));
  }

  const mkldnn::memory &GetWeight() const { return *weight_m; }

  const mkldnn::memory &GetGradw() const { return *gradw_m; }

  const mkldnn::batch_normalization_backward &GetBwd() const { return *bwd; }
};

template <typename DType>
static MKLDNNBNBackward &GetBNBackward(
    const BatchNormParam &param, const OpContext &ctx, const NDArray &in_data,
    const mkldnn::memory &in_mem, const NDArray &diff_data,
    const mkldnn::memory &diff_mem, mkldnn::normalization_flags flags) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNBNSignature, MKLDNNBNBackward, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNBNSignature, MKLDNNBNBackward, OpHash> bwds;
#endif
  MKLDNNBNSignature key(param);
  key.AddSign(in_data);
  key.AddSign(diff_data);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto bwd_pd = _GetBwd(in_mem, diff_mem, param.eps, flags);
    MKLDNNBNBackward bwd(bwd_pd);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

template <typename DType>
void MKLDNNBatchNormBackward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                             const std::vector<NDArray> &inputs, const std::vector<OpReqType> &req,
                             const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 8U);
  const BatchNormParam &param = nnvm::get<BatchNormParam>(attrs.parsed);
  std::vector<NDArray> out_grad(1);
  std::vector<NDArray> out_data(3);
  std::vector<NDArray> in_data(3);
  std::vector<NDArray> aux_states(2);
  out_grad[0] = inputs[0];
  out_data[batchnorm::kMean] = inputs[1];
  out_data[batchnorm::kVar] = inputs[2];
  in_data[batchnorm::kData] = inputs[3];
  in_data[batchnorm::kGamma] = inputs[4];
  in_data[batchnorm::kBeta] = inputs[5];
  aux_states[batchnorm::kMovingMean] = inputs[6];
  aux_states[batchnorm::kMovingVar] = inputs[7];
  const std::vector<NDArray> &in_grad = outputs;
  TmpMemMgr::Get()->Init(ctx.requested[batchnorm::kTempSpace]);
  mkldnn::normalization_flags flags = _GetFlags(in_data,
                                                aux_states,
                                                param,
                                                ctx.is_train && !param.use_global_stats);

  const NDArray &data         = in_data[batchnorm::kData];
  const NDArray &diff         = out_grad[batchnorm::kOut];
  const NDArray &gradIn       = in_grad[batchnorm::kData];
  const NDArray &moving_mean  = aux_states[batchnorm::kMovingMean];
  const NDArray &moving_var   = aux_states[batchnorm::kMovingVar];
  const NDArray &out_mean     = out_data[batchnorm::kMean];
  const NDArray &out_var      = out_data[batchnorm::kVar];

  CHECK(out_mean.IsDefaultData());
  CHECK(out_var.IsDefaultData());
  CHECK(moving_mean.IsDefaultData());
  CHECK(moving_var.IsDefaultData());

  auto data_mem  = data.GetMKLDNNData();
  auto diff_mem  = diff.GetMKLDNNData();
  // MKLDNN batchnorm should run on special layouts. If one of them isn't, we
  // should reorder them.
  if (data.IsDefaultData())
    data_mem = data.GetMKLDNNDataReorder(diff_mem->get_desc());
  else if (diff.IsDefaultData())
    diff_mem = diff.GetMKLDNNDataReorder(data_mem->get_desc());
  auto &bwd = GetBNBackward<DType>(param, ctx, data, *data_mem, diff, *diff_mem, flags);
  auto gradi_mem = CreateMKLDNNMem(const_cast<NDArray &>(gradIn),
      bwd.pd.diff_src_desc(), req[batchnorm::kData]);

  if (static_cast<int>(flags) & static_cast<int>(mkldnn::normalization_flags::use_scale_shift)) {
    const NDArray &gamma    = in_data[batchnorm::kGamma];
    const NDArray &beta     = in_data[batchnorm::kBeta];
    DType *weight_buf = reinterpret_cast<DType *>(bwd.GetWeight().get_data_handle());
    nnvm::dim_t channels_ = data.shape()[1];
    DType *weight_ptr = gamma.data().dptr<DType>();
    DType* bias_ptr = beta.data().dptr<DType>();
    const size_t copy_size = sizeof(DType) * channels_;
    if (!param.fix_gamma) {
      memcpy(weight_buf, weight_ptr, copy_size);
      memcpy(&weight_buf[channels_], bias_ptr, copy_size);
    } else {
      for (int i = 0; i < channels_; i++) {
        weight_buf[i] = static_cast<DType>(1.0f);
      }
      memcpy(&weight_buf[channels_], bias_ptr, copy_size);
    }
    mkldnn_args_map_t net_args;
    net_args[MKLDNN_ARG_SRC] = *data_mem;
    net_args[MKLDNN_ARG_DIFF_SRC] = *gradi_mem.second;
    net_args[MKLDNN_ARG_SCALE_SHIFT] = bwd.GetWeight();
    net_args[MKLDNN_ARG_DIFF_SCALE_SHIFT] = bwd.GetGradw();
    net_args[MKLDNN_ARG_DIFF_DST] = *diff_mem;

    // training but no input mean and variance
    if (ctx.is_train && !param.use_global_stats) {
      DType* moving_mean_ptr  = moving_mean.data().dptr<DType>();
      DType* moving_var_ptr   = moving_var.data().dptr<DType>();
      DType* out_mean_ptr     = out_mean.data().dptr<DType>();
      DType* out_var_ptr      = out_var.data().dptr<DType>();
      mkldnn::memory var_mem(bwd.pd.variance_desc(), CpuEngine::Get()->get_engine());
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
      net_args[MKLDNN_ARG_MEAN] = *(out_mean.GetMKLDNNData());
      net_args[MKLDNN_ARG_VARIANCE] = var_mem;
    } else {
      net_args[MKLDNN_ARG_MEAN] =  *(moving_mean.GetMKLDNNData());
      net_args[MKLDNN_ARG_VARIANCE] = *(moving_var.GetMKLDNNData());
    }
    MKLDNNStream::Get()->RegisterPrimArgs(bwd.GetBwd(), net_args);
    CommitOutput(gradIn, gradi_mem);
    MKLDNNStream::Get()->Submit();

    // copy data from gradw_mem to in_grad[1] and in_grad[2]
    DType *gw_buf = reinterpret_cast<DType *>(bwd.GetGradw().get_data_handle());
    DType *w_grad_1 = in_grad[batchnorm::kGamma].data().dptr<DType>();
    DType *w_grad_2 = in_grad[batchnorm::kBeta].data().dptr<DType>();

    // the gradient of gamma
    if (!param.fix_gamma) {
      if (req[batchnorm::kGamma] != kNullOp) {
        if (req[batchnorm::kGamma] != kAddTo) {
          memcpy(w_grad_1, gw_buf, copy_size);
        } else {
          for (int i = 0; i < channels_; i++) {
            w_grad_1[i] += gw_buf[i];
          }
        }
      }
    } else {
      for (int i = 0; i < channels_; i++) {
        (in_grad[1].data().dptr<DType>())[i] = 0.0f;
      }
    }

    // the gradient of beta
    if (req[batchnorm::kBeta] != kNullOp) {
      if (req[batchnorm::kBeta] != kAddTo) {
        memcpy(w_grad_2, &gw_buf[channels_], copy_size);
      } else {
        DType *grad_beta = &gw_buf[channels_];
        for (int i = 0; i < channels_; i++) {
          w_grad_2[i] += grad_beta[i];
        }
      }
    }
  } else {
    LOG(FATAL) << "MKLDNN batch normalization backward: should not reach here ...";
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BATCH_NORM_INL_H_
