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

inline mkldnn::algorithm GetMKLDNNLRNAlgo(const LRNParam &param) {
  // TODO(Patric): lrn_within_channel will cause core dump in MKLDNN backward
  //               Need to confirm with MKLDNN team and fix later
  return mkldnn::algorithm::lrn_across_channels;
}

inline mkldnn::lrn_forward::primitive_desc GetLRNFwdDesc(
    const LRNParam &param, const bool is_train, const mkldnn::memory::desc &src_md) {
  mkldnn::engine &engine = CpuEngine::Get()->get_engine();
  const mkldnn::algorithm  alg = GetMKLDNNLRNAlgo(param);
  const float alpha = param.alpha;
  const float beta = param.beta;
  const int   nsize = param.nsize;
  const float k = param.knorm;
  auto kind = mkldnn::prop_kind::forward_training;
  if (is_train) {
    kind = mkldnn::prop_kind::forward_training;
  } else {
    kind = mkldnn::prop_kind::forward_scoring;
  }
  mkldnn::lrn_forward::desc fwd_desc(kind, alg, src_md, nsize, alpha, beta, k);
  return mkldnn::lrn_forward::primitive_desc(fwd_desc, engine);
}

inline mkldnn::lrn_backward::primitive_desc GetLRNBwdDesc(
    const LRNParam &param, const mkldnn::memory::desc &data_in_md,
    const mkldnn::memory::desc &diff_md,
    const mkldnn::lrn_forward::primitive_desc &lrnFwd_desc) {
  mkldnn::engine &engine = CpuEngine::Get()->get_engine();
  const mkldnn::algorithm alg = GetMKLDNNLRNAlgo(param);
  const float alpha = param.alpha;
  const float beta = param.beta;
  const int nsize = param.nsize;
  const float k = param.knorm;

  mkldnn::lrn_backward::desc lrnBwd_desc(alg, data_in_md,
                diff_md, nsize, alpha, beta, k);
  return mkldnn::lrn_backward::primitive_desc(lrnBwd_desc,
                               engine, lrnFwd_desc);
}


typedef ParamOpSign<LRNParam> MKLDNNLRNSignature;

// LRN Forward Class
class MKLDNNLRNFwd {
 public:
  MKLDNNLRNFwd(const LRNParam& param,
               bool  is_train,
               const NDArray &in_data) {
    _Init(param, is_train, in_data);
  }

  ~MKLDNNLRNFwd() {}

  void Execute(const OpContext &ctx,
               const NDArray &in_data,
               const OpReqType req,
               const NDArray &out_data);

  mkldnn::lrn_forward &GetFwd();
  const mkldnn::memory *GetWs();
  mkldnn::lrn_forward::primitive_desc &GetFwdPd();

 private:
  std::shared_ptr<mkldnn::lrn_forward> fwd;
  mkldnn::lrn_forward::primitive_desc fwd_pd;

 private:
  void _Init(const LRNParam &param, bool is_train, const NDArray &in_data);
};  // End of LRN Forword Class

void MKLDNNLRNFwd::_Init(const LRNParam &param,
                         bool is_train,
                         const NDArray &in_data) {
  mkldnn::memory::desc in_data_md =
      in_data.GetMKLDNNData()->get_desc();
  this->fwd_pd =
      GetLRNFwdDesc(param, is_train, in_data_md);

  this->fwd = std::shared_ptr<mkldnn::lrn_forward>(new mkldnn::lrn_forward(this->fwd_pd));
}

void MKLDNNLRNFwd::Execute(const OpContext &ctx,
                            const NDArray &in_data,
                            const OpReqType req,
                            const NDArray &out_data) {
  auto output_mem_t = CreateMKLDNNMem(out_data, (this->fwd_pd).dst_desc(), req);

  mkldnn_args_map_t args = {
    { MKLDNN_ARG_SRC, *in_data.GetMKLDNNData()},
    { MKLDNN_ARG_DST, *output_mem_t.second },
  };
  std::shared_ptr<mkldnn::memory> workspace;
  if (ctx.is_train) {
    auto engine = CpuEngine::Get()->get_engine();
    workspace = std::make_shared<mkldnn::memory>((this->fwd_pd).workspace_desc(), engine);
    args[MKLDNN_ARG_WORKSPACE] = *(workspace);
  }
  MKLDNNStream::Get()->RegisterPrimArgs(*(this->fwd), args);
  CommitOutput(out_data, output_mem_t);
  MKLDNNStream::Get()->Submit();
}

mkldnn::lrn_forward &MKLDNNLRNFwd::GetFwd() { return *this->fwd; }
mkldnn::lrn_forward::primitive_desc &MKLDNNLRNFwd::GetFwdPd() { return this->fwd_pd; }

// End of LRN Class and its functions

static MKLDNNLRNFwd &GetLRNFwd(const LRNParam& param,
                               const OpContext &ctx,
                               const NDArray &in_data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNLRNSignature,
                                         MKLDNNLRNFwd,
                                         OpHash> lrn_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNLRNSignature,
                                            MKLDNNLRNFwd,
                                            OpHash> lrn_fwds;
#endif
  auto kind_ =
      ctx.is_train ? mkldnn::prop_kind::forward_training
                   : mkldnn::prop_kind::forward_scoring;

  MKLDNNLRNSignature key(param);
  key.AddSign(static_cast<int>(kind_));
  key.AddSign(in_data);

  auto it = lrn_fwds.find(key);
  if (it == lrn_fwds.end()) {
    MKLDNNLRNFwd fwd(param, ctx.is_train, in_data);
    it = AddToCache(&lrn_fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNLRNForward(const OpContext &ctx, const LRNParam &param,
                      const NDArray &in_data, const OpReqType req,
                      const NDArray &out_data) {
  auto in_buffer = in_data;
  if (in_buffer.IsView() && in_buffer.IsMKLDNNData())
    in_buffer = in_buffer.Reorder2Default();
  MKLDNNLRNFwd fwd = GetLRNFwd(param, ctx, in_buffer);
  fwd.Execute(ctx, in_buffer, req, out_data);
}

// LRN Backward Class
class MKLDNNLRNBwd {
  std::shared_ptr<mkldnn::lrn_backward> bwd;

 public:
  const mkldnn::lrn_forward::primitive_desc fwd_pd;
  const mkldnn::lrn_backward::primitive_desc bwd_pd;

  ~MKLDNNLRNBwd() {}

  MKLDNNLRNBwd(const LRNParam &param, const mkldnn::memory::desc in_data_md,
               const mkldnn::memory::desc diff_md)
      : fwd_pd(GetLRNFwdDesc(param, true, in_data_md)),
        bwd_pd(GetLRNBwdDesc(param, in_data_md, diff_md, this->fwd_pd)) {
          bwd = std::make_shared<mkldnn::lrn_backward>(bwd_pd);
        }

  const mkldnn::lrn_backward &GetBwd() const { return *bwd; }

  void Execute(const NDArray &out_grad,
                  const NDArray &in_data,
                  const NDArray &in_grad,
                  const mkldnn_output_t &diff_src_mem) {
    auto engine = CpuEngine::Get()->get_engine();
    auto workspace = std::make_shared<mkldnn::memory>((this->fwd_pd).workspace_desc(), engine);
    mkldnn_args_map_t args = {
      { MKLDNN_ARG_SRC, *in_data.GetMKLDNNData() },
      { MKLDNN_ARG_DIFF_DST, *out_grad.GetMKLDNNData()},
      { MKLDNN_ARG_WORKSPACE, *workspace },
      { MKLDNN_ARG_DIFF_SRC, *diff_src_mem.second }
    };
    MKLDNNStream::Get()->RegisterPrimArgs(*(this->bwd), args);
    CommitOutput(in_grad, diff_src_mem);
    MKLDNNStream::Get()->Submit();
  }
};  // End of LRN Class

static MKLDNNLRNBwd &GetLRNBwd(const LRNParam &param, const NDArray &in_data,
                               const NDArray &in_grad, const NDArray &out_grad) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local
      std::unordered_map<MKLDNNLRNSignature, MKLDNNLRNBwd, OpHash> lrn_bwds;
#else
  static MX_THREAD_LOCAL
      std::unordered_map<MKLDNNLRNSignature, MKLDNNLRNBwd, OpHash> lrn_bwds;
#endif
  MKLDNNLRNSignature key(param);
  key.AddSign(in_data);
  key.AddSign(in_grad);
  key.AddSign(out_grad);

  auto it = lrn_bwds.find(key);
  if (it == lrn_bwds.end()) {
    const mkldnn::memory::desc in_data_md =
        in_data.GetMKLDNNData()->get_desc();
    const mkldnn::memory::desc diff_md =
        out_grad.GetMKLDNNData()->get_desc();
    MKLDNNLRNBwd bwd(param, in_data_md, diff_md);
    it = AddToCache(&lrn_bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNLRNBackward(const OpContext &ctx, const LRNParam &param,
                       const NDArray &out_grad,
                       const NDArray &in_data,
                       const OpReqType req,
                       const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }
  // TODO(alex): (MXNET-846) figure out why in_grad output incorrect when in_data is nchw8c
  auto in_buffer = in_data;
  if (in_buffer.IsMKLDNNData()) {
    in_buffer = in_data.Reorder2Default();
  }
  MKLDNNLRNBwd &bwd = GetLRNBwd(param, in_buffer, in_grad, out_grad);
  mkldnn_output_t diff_src_mem =
      CreateMKLDNNMem(in_grad, bwd.bwd_pd.diff_src_desc(), req);

  bwd.Execute(out_grad, in_buffer, in_grad, diff_src_mem);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H__

