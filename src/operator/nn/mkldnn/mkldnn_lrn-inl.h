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
 * \file mkldnn_lrn-inl.h 
 * \brief
 * \Author: Patric Zhao, patric.zhao@intel.com
*/
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H_

#if MXNET_USE_MKLDNN == 1
#include <utility>
#include <mkldnn.hpp>
#include "../lrn-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

static inline bool SupportMKLDNNLRN(const LRNParam &param) {
    return (param.nsize % 2 == 1) ? 1 : 0;
}

inline static lrn_forward::primitive_desc GetLRNFwdDesc(
    const LRNParam &param, bool is_train, const mkldnn::memory::desc &in_md) {
  auto engine = CpuEngine::Get()->get_engine();
  auto alg_   = algorithm::lrn_across_channels;
  auto alpha_ = param.alpha;
  auto beta_  = param.beta;
  auto nsize_ = param.nsize;
  auto k_     = param.knorm;
  auto kind_  = prop_kind::forward_training;
  if (is_train) {
    kind_ = prop_kind::forward_training;
  } else {
    kind_ = prop_kind::forward_scoring;
  }
  lrn_forward::desc fwd_desc_(kind_, alg_, in_md, nsize_, alpha_, beta_, k_);
  return mkldnn::lrn_forward::primitive_desc(fwd_desc_, engine);
}

inline static mkldnn::lrn_backward::primitive_desc GetLRNBwd(
    const LRNParam &param, const mkldnn::memory::desc &diff_in_md,
    const mkldnn::memory::desc &diff_md,
    const lrn_forward::primitive_desc &lrnFwd_desc) {
  auto engine = CpuEngine::Get()->get_engine();
  auto alg_   = algorithm::lrn_across_channels;
  auto alpha_ = param.alpha;
  auto beta_  = param.beta;
  int  nsize_ = param.nsize;
  auto k_ = param.knorm;

  lrn_backward::desc lrnBwd_desc(alg_, diff_in_md,
                diff_md, nsize_, alpha_, beta_, k_);
  return mkldnn::lrn_backward::primitive_desc(lrnBwd_desc,
                               engine, lrnFwd_desc);
}

typedef MKLDNNParamOpSign<LRNParam> MKLDNNLRNSignature;

// LRN Forward Class
class MKLDNNLRNFwd {
 public:
  MKLDNNLRNFwd(const LRNParam& param,
               bool  is_train,
               const NDArray &in_data,
               const NDArray *workspace):
               is_train(is_train) {
    _Init(param, is_train, in_data, workspace);
  }

  ~MKLDNNLRNFwd() {}

  void SetDataHandle(const NDArray &data,
                     const NDArray &output,
                     const NDArray *workspace);

  void Execute();

 private:
  std::shared_ptr<mkldnn::lrn_forward> fwd;
  std::shared_ptr<mkldnn::memory> in_mem;
  std::shared_ptr<mkldnn::memory> out_mem;
  std::shared_ptr<mkldnn::memory> ws_mem;
  bool is_train;

 private:
  void _Init(const LRNParam &param, bool is_train,
             const NDArray &in_data, const NDArray *workspace);
};  // End of LRN Forword Class

void MKLDNNLRNFwd::_Init(const LRNParam &param,
                         bool is_train,
                         const NDArray &in_data,
                         const NDArray *workspace) {
  auto in_data_md = in_data.GetMKLDNNData()->get_primitive_desc().desc();
  auto fwd_pd = GetLRNFwdDesc(param, is_train, in_data_md);

  this->in_mem.reset(new mkldnn::memory(in_data.GetMKLDNNData()
                     ->get_primitive_desc()));
  this->out_mem.reset(new mkldnn::memory(fwd_pd.dst_primitive_desc()));

  if (this->is_train) {
    if (workspace == nullptr) {
      this->ws_mem.reset(new mkldnn::memory(fwd_pd.workspace_primitive_desc()));
    } else {
      this->ws_mem.reset(new mkldnn::memory(workspace->GetMKLDNNData()
                         ->get_primitive_desc()));
    }  // end workspace
    this->fwd = std::shared_ptr<mkldnn::lrn_forward>(
                new mkldnn::lrn_forward(fwd_pd,
                                        mkldnn::primitive::at(*(this->in_mem)),
                                        *(this->ws_mem), *(this->out_mem)));
  } else {
    this->fwd = std::shared_ptr<mkldnn::lrn_forward>(
                new mkldnn::lrn_forward(fwd_pd,
                                        mkldnn::primitive::at(*(this->in_mem)),
                                        *(this->out_mem)));
  }  // end is_train
}

void MKLDNNLRNFwd::SetDataHandle(const NDArray &in_data,
                                 const NDArray &out_data,
                                 const NDArray *workspace) {
  auto in_data_mem   = in_data.GetMKLDNNData();
  auto out_data_mem  = const_cast<NDArray&>(out_data).CreateMKLDNNData(
                       this->out_mem->get_primitive_desc());
  if (this->is_train) {
    auto workspace_mem = workspace->GetMKLDNNData();
    this->ws_mem->set_data_handle(workspace_mem->get_data_handle());
  }
  this->in_mem->set_data_handle(in_data_mem->get_data_handle());
  this->out_mem->set_data_handle(out_data_mem->get_data_handle());
}

void MKLDNNLRNFwd::Execute() {
  MKLDNNStream::Get()->RegisterPrim(*(this->fwd));
  MKLDNNStream::Get()->Submit();
}
// End of LRN Class and its functions

static MKLDNNLRNFwd &GetLRNFwd(const LRNParam& param,
                               const OpContext &ctx,
                               const NDArray &in_data,
                               const NDArray *workspace) {
  static thread_local std::unordered_map<MKLDNNLRNSignature,
                                         MKLDNNLRNFwd,
                                         MKLDNNOpHash> lrn_fwds;
  auto alg_ = algorithm::lrn_across_channels;
  auto kind_ = prop_kind::forward_training;
  if (ctx.is_train) {
    kind_ = prop_kind::forward_training;
  } else {
    kind_ = prop_kind::forward_scoring;
  }

  MKLDNNLRNSignature key(param);
  key.AddSign(alg_);
  key.AddSign(kind_);
  key.AddSign(in_data);

  auto it = lrn_fwds.find(key);
  if (it == lrn_fwds.end()) {
    MKLDNNLRNFwd fwd(param, ctx.is_train ? 1:0, in_data, workspace);
    auto ins_ret = lrn_fwds.insert(std::pair<MKLDNNLRNSignature, MKLDNNLRNFwd>
                               (key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNLRNCompute(const OpContext &ctx, const LRNParam &param,
                      const NDArray &in_data, const OpReqType &req,
                      const NDArray &out_data, const NDArray *workspace) {
  MKLDNNLRNFwd fwd = GetLRNFwd(param, ctx, in_data, workspace);
  fwd.SetDataHandle(in_data, out_data, workspace);
  fwd.Execute();
}

void MKLDNNLRNGradCompute(const OpContext &ctx, const LRNParam &param,
                          const NDArray &out_grad,
                          const NDArray &in_data,
                          const OpReqType &req,
                          const NDArray &in_grad,
                          const NDArray *workspace) {
  if (req == kNullOp) {
    return;
  }

  // Get FW primitive
  auto data_mem = in_data.GetMKLDNNData();
  auto data_md = data_mem->get_primitive_desc().desc();
  auto pdesc_fwd = GetLRNFwdDesc(param, ctx.is_train, data_md);

  // Create data and diff descirption and primitive
  auto data_in_md = pdesc_fwd.src_primitive_desc().desc();
  auto diff_mem = out_grad.GetMKLDNNData();
  auto diff_md = diff_mem->get_primitive_desc().desc();
  auto pdesc_bwd = GetLRNBwd(param, data_in_md, diff_md, pdesc_fwd);
  auto diff_src_mem = CreateMKLDNNMem(in_grad,
          pdesc_bwd.diff_src_primitive_desc(), req);

  // Re-compute in case the FW OP doesn't pass the workspace;
  // otherwise, use the FW workspace.
  if (workspace == nullptr) {
    std::shared_ptr<const mkldnn::memory> ws_mem(
            new mkldnn::memory(pdesc_fwd.workspace_primitive_desc()));
    std::shared_ptr<const mkldnn::memory> dst_temp(
            new mkldnn::memory(pdesc_fwd.dst_primitive_desc()));
    MKLDNNStream::Get()->RegisterPrim(
            lrn_forward(pdesc_fwd, mkldnn::primitive::at(*data_mem),
            *ws_mem, *dst_temp));
    MKLDNNStream::Get()->RegisterPrim(
          lrn_backward(pdesc_bwd, mkldnn::primitive::at(*data_mem),
          mkldnn::primitive::at(*diff_mem), *ws_mem, *diff_src_mem.second));
    MKLDNNStream::Get()->Submit();
  } else {
    auto ws_mem = workspace->GetMKLDNNData();
    MKLDNNStream::Get()->RegisterPrim(
        lrn_backward(pdesc_bwd, mkldnn::primitive::at(*data_mem),
        mkldnn::primitive::at(*diff_mem), *ws_mem,
                              *diff_src_mem.second));
    MKLDNNStream::Get()->Submit();
  }
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H__
