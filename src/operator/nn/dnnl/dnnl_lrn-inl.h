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
 * \file dnnl_lrn-inl.h
 * \brief
 * \Author: Patric Zhao, patric.zhao@intel.com
 */
#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_LRN_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_LRN_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <dnnl.hpp>
#include <utility>
#include <vector>

#include "operator/nn/lrn-inl.h"
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

inline dnnl::algorithm GetDNNLLRNAlgo(const LRNParam& param) {
  // TODO(Patric): lrn_within_channel will cause core dump in DNNL backward
  //               Need to confirm with DNNL team and fix later
  return dnnl::algorithm::lrn_across_channels;
}

inline dnnl::lrn_forward::primitive_desc GetLRNFwdDesc(const LRNParam& param,
                                                       const bool is_train,
                                                       const dnnl::memory::desc& src_md) {
  dnnl::engine& engine      = CpuEngine::Get()->get_engine();
  const dnnl::algorithm alg = GetDNNLLRNAlgo(param);
  const float alpha         = param.alpha;
  const float beta          = param.beta;
  const int nsize           = param.nsize;
  const float k             = param.knorm;
  auto kind                 = dnnl::prop_kind::forward_training;
  if (is_train) {
    kind = dnnl::prop_kind::forward_training;
  } else {
    kind = dnnl::prop_kind::forward_scoring;
  }
  dnnl::lrn_forward::desc fwd_desc(kind, alg, src_md, nsize, alpha, beta, k);
  return dnnl::lrn_forward::primitive_desc(fwd_desc, engine);
}

inline dnnl::lrn_backward::primitive_desc GetLRNBwdDesc(
    const LRNParam& param,
    const dnnl::memory::desc& data_in_md,
    const dnnl::memory::desc& diff_md,
    const dnnl::lrn_forward::primitive_desc& lrnFwd_desc) {
  dnnl::engine& engine      = CpuEngine::Get()->get_engine();
  const dnnl::algorithm alg = GetDNNLLRNAlgo(param);
  const float alpha         = param.alpha;
  const float beta          = param.beta;
  const int nsize           = param.nsize;
  const float k             = param.knorm;

  dnnl::lrn_backward::desc lrnBwd_desc(alg, data_in_md, diff_md, nsize, alpha, beta, k);
  return dnnl::lrn_backward::primitive_desc(lrnBwd_desc, engine, lrnFwd_desc);
}

typedef ParamOpSign<LRNParam> DNNLLRNSignature;

// LRN Forward Class
class DNNLLRNFwd {
 public:
  DNNLLRNFwd(const LRNParam& param, bool is_train, const NDArray& in_data) {
    _Init(param, is_train, in_data);
  }

  ~DNNLLRNFwd() {}

  void Execute(const OpContext& ctx,
               const NDArray& in_data,
               const OpReqType req,
               const NDArray& out_data);

  dnnl::lrn_forward& GetFwd();
  const dnnl::memory* GetWs();
  dnnl::lrn_forward::primitive_desc& GetFwdPd();

 private:
  std::shared_ptr<dnnl::lrn_forward> fwd;
  dnnl::lrn_forward::primitive_desc fwd_pd;

 private:
  void _Init(const LRNParam& param, bool is_train, const NDArray& in_data);
};  // End of LRN Forword Class

void DNNLLRNFwd::_Init(const LRNParam& param, bool is_train, const NDArray& in_data) {
  dnnl::memory::desc in_data_md = in_data.GetDNNLData()->get_desc();
  this->fwd_pd                  = GetLRNFwdDesc(param, is_train, in_data_md);

  this->fwd = std::shared_ptr<dnnl::lrn_forward>(new dnnl::lrn_forward(this->fwd_pd));
}

void DNNLLRNFwd::Execute(const OpContext& ctx,
                         const NDArray& in_data,
                         const OpReqType req,
                         const NDArray& out_data) {
  auto output_mem_t = CreateDNNLMem(out_data, (this->fwd_pd).dst_desc(), req);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC, *in_data.GetDNNLData()},
      {DNNL_ARG_DST, *output_mem_t.second},
  };
  std::shared_ptr<dnnl::memory> workspace;
  if (ctx.is_train) {
    auto engine = CpuEngine::Get()->get_engine();
    workspace   = std::make_shared<dnnl::memory>((this->fwd_pd).workspace_desc(), engine);
    args[DNNL_ARG_WORKSPACE] = *(workspace);
  }
  DNNLStream::Get()->RegisterPrimArgs(*(this->fwd), args);
  CommitOutput(out_data, output_mem_t);
  DNNLStream::Get()->Submit();
}

dnnl::lrn_forward& DNNLLRNFwd::GetFwd() {
  return *this->fwd;
}
dnnl::lrn_forward::primitive_desc& DNNLLRNFwd::GetFwdPd() {
  return this->fwd_pd;
}

// End of LRN Class and its functions

static DNNLLRNFwd& GetLRNFwd(const LRNParam& param, const OpContext& ctx, const NDArray& in_data) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLLRNSignature, DNNLLRNFwd, OpHash> lrn_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLLRNSignature, DNNLLRNFwd, OpHash> lrn_fwds;
#endif
  auto kind_ = ctx.is_train ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_scoring;

  DNNLLRNSignature key(param);
  key.AddSign(static_cast<int>(kind_));
  key.AddSign(in_data);

  auto it = lrn_fwds.find(key);
  if (it == lrn_fwds.end()) {
    DNNLLRNFwd fwd(param, ctx.is_train, in_data);
    it = AddToCache(&lrn_fwds, key, fwd);
  }
  return it->second;
}

void DNNLLRNForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const NDArray& in_data,
                    const OpReqType req,
                    const NDArray& out_data) {
  const LRNParam& param = nnvm::get<LRNParam>(attrs.parsed);
  auto in_buffer        = in_data;
  if (in_buffer.IsView() && in_buffer.IsDNNLData())
    in_buffer = in_buffer.Reorder2Default();
  DNNLLRNFwd fwd = GetLRNFwd(param, ctx, in_buffer);
  fwd.Execute(ctx, in_buffer, req, out_data);
}

// LRN Backward Class
class DNNLLRNBwd {
  std::shared_ptr<dnnl::lrn_backward> bwd;

 public:
  const dnnl::lrn_forward::primitive_desc fwd_pd;
  const dnnl::lrn_backward::primitive_desc bwd_pd;

  ~DNNLLRNBwd() {}

  DNNLLRNBwd(const LRNParam& param,
             const dnnl::memory::desc in_data_md,
             const dnnl::memory::desc diff_md)
      : fwd_pd(GetLRNFwdDesc(param, true, in_data_md)),
        bwd_pd(GetLRNBwdDesc(param, in_data_md, diff_md, this->fwd_pd)) {
    bwd = std::make_shared<dnnl::lrn_backward>(bwd_pd);
  }

  const dnnl::lrn_backward& GetBwd() const {
    return *bwd;
  }

  void Execute(const NDArray& out_grad,
               const NDArray& in_data,
               const NDArray& in_grad,
               const dnnl_output_t& diff_src_mem) {
    auto engine          = CpuEngine::Get()->get_engine();
    auto workspace       = std::make_shared<dnnl::memory>((this->fwd_pd).workspace_desc(), engine);
    dnnl_args_map_t args = {{DNNL_ARG_SRC, *in_data.GetDNNLData()},
                            {DNNL_ARG_DIFF_DST, *out_grad.GetDNNLData()},
                            {DNNL_ARG_WORKSPACE, *workspace},
                            {DNNL_ARG_DIFF_SRC, *diff_src_mem.second}};
    DNNLStream::Get()->RegisterPrimArgs(*(this->bwd), args);
    CommitOutput(in_grad, diff_src_mem);
    DNNLStream::Get()->Submit();
  }
};  // End of LRN Class

static DNNLLRNBwd& GetLRNBwd(const LRNParam& param,
                             const NDArray& in_data,
                             const NDArray& in_grad,
                             const NDArray& out_grad) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLLRNSignature, DNNLLRNBwd, OpHash> lrn_bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLLRNSignature, DNNLLRNBwd, OpHash> lrn_bwds;
#endif
  DNNLLRNSignature key(param);
  key.AddSign(in_data);
  key.AddSign(in_grad);
  key.AddSign(out_grad);

  auto it = lrn_bwds.find(key);
  if (it == lrn_bwds.end()) {
    const dnnl::memory::desc in_data_md = in_data.GetDNNLData()->get_desc();
    const dnnl::memory::desc diff_md    = out_grad.GetDNNLData()->get_desc();
    DNNLLRNBwd bwd(param, in_data_md, diff_md);
    it = AddToCache(&lrn_bwds, key, bwd);
  }
  return it->second;
}

void DNNLLRNBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp) {
    return;
  }
  const LRNParam& param   = nnvm::get<LRNParam>(attrs.parsed);
  const NDArray& out_grad = inputs[0];
  const NDArray& in_data  = inputs[1];
  const NDArray& in_grad  = outputs[0];
  // TODO(alex): (MXNET-846) figure out why in_grad output incorrect when in_data is nchw8c
  const auto in_buffer       = in_data.Reorder2Default();
  DNNLLRNBwd& bwd            = GetLRNBwd(param, in_buffer, in_grad, out_grad);
  dnnl_output_t diff_src_mem = CreateDNNLMem(in_grad, bwd.bwd_pd.diff_src_desc(), req[0]);

  bwd.Execute(out_grad, in_buffer, in_grad, diff_src_mem);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_LRN_INL_H__
