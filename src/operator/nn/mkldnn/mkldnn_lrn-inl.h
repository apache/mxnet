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

inline mkldnn::lrn_forward::primitive_desc GetLRNFwdDesc(
    const LRNParam &param, const bool is_train, const memory::desc &src_md) {
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

inline mkldnn::lrn_backward::primitive_desc GetLRNBwdDesc(
    const LRNParam &param, const mkldnn::memory::desc &data_in_md,
    const mkldnn::memory::desc &diff_md,
    const mkldnn::lrn_forward::primitive_desc &lrnFwd_desc) {
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

  void SetNewMem(const NDArray &in_data,
                 const mkldnn::memory *out_mem);

  void Execute(const NDArray &out_data);

  mkldnn::lrn_forward &GetFwd();

  const mkldnn::memory *GetWs();

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
  mkldnn::memory::desc in_data_md =
      in_data.GetMKLDNNData()->get_primitive_desc().desc();
  mkldnn::lrn_forward::primitive_desc fwd_pd =
      GetLRNFwdDesc(param, is_train, in_data_md);

  this->in_mem.reset(new mkldnn::memory(in_data.GetMKLDNNData()
                     ->get_primitive_desc()));
  this->out_mem.reset(new mkldnn::memory(fwd_pd.dst_primitive_desc()));
  if (is_train) {
    // If it's training, we have to create a workspace memory. Otherwise,
    // MKLDNN will have segmentation fault.
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

void MKLDNNLRNFwd::SetNewMem(const NDArray &in_data,
                             const mkldnn::memory *out_mem) {
  const mkldnn::memory *in_data_mem = in_data.GetMKLDNNData();
  this->in_mem->set_data_handle(in_data_mem->get_data_handle());
  this->out_mem->set_data_handle(out_mem->get_data_handle());
}

void MKLDNNLRNFwd::Execute(const NDArray &out_data) {
  MKLDNNStream::Get()->RegisterPrim(*(this->fwd));
  CommitOutput(out_data, output_mem_t);
  MKLDNNStream::Get()->Submit();
}

mkldnn::lrn_forward &MKLDNNLRNFwd::GetFwd() { return *this->fwd; }

const mkldnn::memory *MKLDNNLRNFwd::GetWs() { return this->ws_mem.get(); }
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
      ctx.is_train ? prop_kind::forward_training : prop_kind::forward_scoring;

  MKLDNNLRNSignature key(param);
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

void MKLDNNLRNForward(const OpContext &ctx, const LRNParam &param,
                      const NDArray &in_data, const OpReqType req,
                      const NDArray &out_data) {
  auto in_buffer = in_data;
  if (in_buffer.IsView() && in_buffer.IsMKLDNNData())
    in_buffer = in_buffer.Reorder2Default();
  MKLDNNLRNFwd fwd = GetLRNFwd(param, ctx, in_buffer);
  fwd.SetNewMem(in_buffer, out_data, req);
  fwd.Execute(out_data);
}

// LRN Backward Class
class MKLDNNLRNBwd {
  std::shared_ptr<mkldnn::lrn_backward> bwd;
  std::shared_ptr<mkldnn::memory> in_data_mem;
  std::shared_ptr<mkldnn::memory> diff_dst_mem;
  std::shared_ptr<mkldnn::memory> ws_mem;
  std::shared_ptr<mkldnn::memory> diff_src_mem;

 public:
  const mkldnn::lrn_forward::primitive_desc fwd_pd;
  const mkldnn::lrn_backward::primitive_desc bwd_pd;

  ~MKLDNNLRNBwd() {}

  MKLDNNLRNBwd(const LRNParam &param, const mkldnn::memory::desc in_data_md,
               const mkldnn::memory::desc diff_md)
      : fwd_pd(GetLRNFwdDesc(param, true, in_data_md)),
        bwd_pd(GetLRNBwdDesc(param, in_data_md, diff_md, this->fwd_pd)) {}

  void SetNewMem(const NDArray &in_data, const NDArray &out_grad,
                 const mkldnn::memory *ws, const mkldnn::memory *diff_src_mem) {
    if (bwd == nullptr) {
      this->in_data_mem.reset(
          new mkldnn::memory(this->fwd_pd.src_primitive_desc(),
                             in_data.GetMKLDNNData()->get_data_handle()));
      this->diff_dst_mem.reset(
          new mkldnn::memory(this->fwd_pd.dst_primitive_desc(),
                             out_grad.GetMKLDNNData()->get_data_handle()));
      this->ws_mem.reset(
          new mkldnn::memory(this->fwd_pd.workspace_primitive_desc(),
                             ws->get_data_handle()));
      this->diff_src_mem.reset(
          new mkldnn::memory(this->bwd_pd.diff_src_primitive_desc(),
                             diff_src_mem->get_data_handle()));
      this->bwd.reset(new mkldnn::lrn_backward(
          this->bwd_pd, mkldnn::primitive::at(*this->in_data_mem),
          mkldnn::primitive::at(*this->diff_dst_mem), *this->ws_mem,
          *this->diff_src_mem));
    } else {
      this->in_data_mem->set_data_handle(
          in_data.GetMKLDNNData()->get_data_handle());
      this->diff_dst_mem->set_data_handle(
          out_grad.GetMKLDNNData()->get_data_handle());
      this->ws_mem->set_data_handle(ws->get_data_handle());
      this->diff_src_mem->set_data_handle(diff_src_mem->get_data_handle());
    }
  }

  void Execute(const NDArray &in_grad, const mkldnn_output_t &diff_src_mem_) {
    MKLDNNStream::Get()->RegisterPrim(*(this->bwd));
    CommitOutput(in_grad, diff_src_mem_);
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
        in_data.GetMKLDNNData()->get_primitive_desc().desc();
    const mkldnn::memory::desc diff_md =
        out_grad.GetMKLDNNData()->get_primitive_desc().desc();
    MKLDNNLRNBwd bwd(param, in_data_md, diff_md);
    auto ins_ret =
        lrn_bwds.insert(std::pair<MKLDNNLRNSignature, MKLDNNLRNBwd>(key, bwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
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
  // Repeat FW for getting workspace
  // TODO(Patric): To keep the function stateless, we can't pass workspace
  //               from LRN forward to backward. We have to re-compute
  //               LRN forward to get the workspace.
  //               Will refine this code later.
  MKLDNNLRNFwd fwd = GetLRNFwd(param, ctx, in_buffer);
  std::shared_ptr<const mkldnn::memory> dst_temp(
      new mkldnn::memory(bwd.fwd_pd.dst_primitive_desc()));
  fwd.SetNewMem(in_buffer, dst_temp.get());
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());

  mkldnn_output_t diff_src_mem =
      CreateMKLDNNMem(in_grad, bwd.bwd_pd.diff_src_primitive_desc(), req);
  bwd.SetNewMem(in_buffer, out_grad, fwd.GetWs(), diff_src_mem.second);
  bwd.Execute(in_grad, diff_src_mem);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H__
