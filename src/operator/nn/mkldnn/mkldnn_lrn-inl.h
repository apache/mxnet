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

inline algorithm GetMKLDNNLRNAlgo(const LRNParam &param) {
  // TODO(Patric): lrn_within_channel will cause core dump in MKLDNN backward
  //               Need to confirm with MKLDNN team and fix later
  return algorithm::lrn_across_channels;
}

inline lrn_forward::primitive_desc GetLRNFwdDesc(const LRNParam &param,
                                                 const bool is_train,
                                                 const memory::desc &src_md) {
  mkldnn::engine &engine = CpuEngine::Get()->get_engine();
  const algorithm  alg = GetMKLDNNLRNAlgo(param);
  const float alpha = param.alpha;
  const float beta = param.beta;
  const int   nsize = param.nsize;
  const float k = param.knorm;
  auto kind = prop_kind::forward_training;
  if (is_train) {
    kind = prop_kind::forward_training;
  } else {
    kind = prop_kind::forward_scoring;
  }
  lrn_forward::desc fwd_desc(kind, alg, src_md, nsize, alpha, beta, k);
  return mkldnn::lrn_forward::primitive_desc(fwd_desc, engine);
}

inline mkldnn::lrn_backward::primitive_desc
GetLRNBwd(const LRNParam &param,
          const mkldnn::memory::desc &data_in_md,
          const mkldnn::memory::desc &diff_md,
          const lrn_forward::primitive_desc &lrnFwd_desc) {
  mkldnn::engine &engine = CpuEngine::Get()->get_engine();
  const algorithm alg = GetMKLDNNLRNAlgo(param);
  const float alpha = param.alpha;
  const float beta = param.beta;
  const int nsize = param.nsize;
  const float k = param.knorm;

  lrn_backward::desc lrnBwd_desc(alg, data_in_md,
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
               const NDArray &in_data):
               is_train(is_train) {
    _Init(param, is_train, in_data);
  }

  ~MKLDNNLRNFwd() {}

  void SetNewMem(const NDArray &data,
                 const NDArray &output,
                 const OpReqType req);

  void Execute(const NDArray &out_data);

 private:
  std::shared_ptr<mkldnn::lrn_forward> fwd;
  std::shared_ptr<mkldnn::memory> in_mem;
  std::shared_ptr<mkldnn::memory> out_mem;
  std::shared_ptr<mkldnn::memory> ws_mem;
  mkldnn_output_t output_mem_t;
  bool is_train;

 private:
  void _Init(const LRNParam &param, bool is_train, const NDArray &in_data);
};  // End of LRN Forword Class

void MKLDNNLRNFwd::_Init(const LRNParam &param,
                         bool is_train,
                         const NDArray &in_data) {
  mkldnn::memory::desc in_data_md = in_data.GetMKLDNNData()->get_primitive_desc().desc();
  lrn_forward::primitive_desc fwd_pd = GetLRNFwdDesc(param, is_train, in_data_md);

  this->in_mem.reset(new mkldnn::memory(in_data.GetMKLDNNData()
                     ->get_primitive_desc()));
  this->out_mem.reset(new mkldnn::memory(fwd_pd.dst_primitive_desc()));
  if (is_train) {
    // If it's training, we have to create a workspace memory. Otherwise, MKLDNN
    // will have segmentation fault.
    ws_mem.reset(new mkldnn::memory(fwd_pd.workspace_primitive_desc()));
    this->fwd = std::shared_ptr<mkldnn::lrn_forward>(
        new mkldnn::lrn_forward(fwd_pd, mkldnn::primitive::at(*this->in_mem),
                                *this->ws_mem, *this->out_mem));
  } else {
    this->fwd = std::shared_ptr<mkldnn::lrn_forward>(
        new mkldnn::lrn_forward(fwd_pd, mkldnn::primitive::at(*(this->in_mem)),
                                *(this->out_mem)));
  }
}

void MKLDNNLRNFwd::SetNewMem(const NDArray &in_data,
                             const NDArray &out_data,
                             const OpReqType req) {
  const mkldnn::memory *in_data_mem = in_data.GetMKLDNNData();
  output_mem_t = CreateMKLDNNMem(out_data, this->out_mem->get_primitive_desc(), req);
  this->in_mem->set_data_handle(in_data_mem->get_data_handle());
  this->out_mem->set_data_handle(output_mem_t.second->get_data_handle());
}

void MKLDNNLRNFwd::Execute(const NDArray &out_data) {
  MKLDNNStream::Get()->RegisterPrim(*(this->fwd));
  CommitOutput(out_data, output_mem_t);
  MKLDNNStream::Get()->Submit();
}
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
    MKLDNNLRNFwd fwd(param, ctx.is_train, in_data);
    auto ins_ret = lrn_fwds.insert(std::pair<MKLDNNLRNSignature, MKLDNNLRNFwd>
                                   (key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNLRNForward(const OpContext &ctx,
                      const LRNParam &param,
                      const NDArray &in_data,
                      const OpReqType req,
                      const NDArray &out_data) {
  auto in_buffer = in_data;
  if (in_buffer.IsView() && in_buffer.IsMKLDNNData())
    in_buffer = in_buffer.Reorder2Default();
  MKLDNNLRNFwd fwd = GetLRNFwd(param, ctx, in_buffer);
  fwd.SetNewMem(in_buffer, out_data, req);
  fwd.Execute(out_data);
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

  // Repeat FW for getting workspace
  const mkldnn::memory *data_mem = in_buffer.GetMKLDNNData();
  const mkldnn::memory::desc data_md = data_mem->get_primitive_desc().desc();
  const lrn_forward::primitive_desc pdesc_fwd = GetLRNFwdDesc(param, ctx.is_train,
                                                              data_md);

  // TODO(Patric): To keep the function stateless, we can't pass workspace
  //               from LRN forward to backward. We have to re-compute
  //               LRN forward to get the workspace.
  //               Will refine this code later.
  std::shared_ptr<const mkldnn::memory> ws_mem(
          new mkldnn::memory(pdesc_fwd.workspace_primitive_desc()));
  std::shared_ptr<const mkldnn::memory> dst_temp(
          new mkldnn::memory(pdesc_fwd.dst_primitive_desc()));
  MKLDNNStream::Get()->RegisterPrim(
          lrn_forward(pdesc_fwd, mkldnn::primitive::at(*data_mem),
          *ws_mem, *dst_temp));

  const mkldnn::memory *diff_mem = out_grad.GetMKLDNNData();
  const mkldnn::memory::desc diff_md = diff_mem->get_primitive_desc().desc();
  const mkldnn::lrn_backward::primitive_desc pdesc_bwd = GetLRNBwd(param, data_md,
                                                                   diff_md, pdesc_fwd);
  mkldnn_output_t diff_src_mem = CreateMKLDNNMem(in_grad,
                                                 pdesc_bwd.diff_src_primitive_desc(), req);

  MKLDNNStream::Get()->RegisterPrim(
        lrn_backward(pdesc_bwd, mkldnn::primitive::at(*data_mem),
        mkldnn::primitive::at(*diff_mem), *ws_mem, *diff_src_mem.second));
  CommitOutput(in_grad, diff_src_mem);
  MKLDNNStream::Get()->Submit();
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H__
