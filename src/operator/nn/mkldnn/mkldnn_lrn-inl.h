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
#include <mkldnn.hpp>
#include "../lrn-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

static inline algorithm GetMKLDNNLRNAlgo(const LRNParam &param) {
  // TODO(Patric): lrn_within_channel will cause core dump in MKLDNN backward
  //       Need to fix from MKLDNN
  return algorithm::lrn_across_channels;
}

inline static lrn_forward::primitive_desc GetLRNFwd(
    const LRNParam &param, bool is_train, const memory::desc &src_md) {
  auto engine = CpuEngine::Get()->get_engine();
  auto alg_ = GetMKLDNNLRNAlgo(param);
  auto alpha_ = param.alpha;
  auto beta_ = param.beta;
  auto nsize_ = param.nsize;
  auto k_ = param.knorm;
  auto kind_ = prop_kind::forward_training;
  if (is_train) {
    kind_ = prop_kind::forward_training;
  } else {
    kind_ = prop_kind::forward_scoring;
  }
  lrn_forward::desc fwd_desc_(kind_, alg_, src_md, nsize_, alpha_, beta_, k_);
  return mkldnn::lrn_forward::primitive_desc(fwd_desc_, engine);
}

inline static mkldnn::lrn_backward::primitive_desc GetLRNBwd(
    const LRNParam &param, const mkldnn::memory::desc &diff_in_md,
    const mkldnn::memory::desc &diff_md,
    const lrn_forward::primitive_desc &lrnFwd_desc) {
  auto engine = CpuEngine::Get()->get_engine();
  auto alg_ = GetMKLDNNLRNAlgo(param);
  auto alpha_ = param.alpha;
  auto  beta_ = param.beta;
  int nsize_ = param.nsize;
  auto k_ = param.knorm;

  lrn_backward::desc lrnBwd_desc(alg_, diff_in_md,
                diff_md, nsize_, alpha_, beta_, k_);
  return mkldnn::lrn_backward::primitive_desc(lrnBwd_desc,
                               engine, lrnFwd_desc);
}

void MKLDNNLRN_Forward(const OpContext &ctx, const LRNParam &param,
                           const NDArray &in_data, const OpReqType &req,
                           const NDArray &out_data) {
  auto src_mem = in_data.GetMKLDNNData();
  auto src_md =  src_mem->get_primitive_desc().desc();
  auto pdesc = GetLRNFwd(param, ctx.is_train, src_md);
  auto dst_mem =  const_cast<NDArray &>(out_data).CreateMKLDNNData(
          pdesc.dst_primitive_desc());
  if (ctx.is_train) {
    std::shared_ptr<const mkldnn::memory> ws_mem(
            new mkldnn::memory(pdesc.workspace_primitive_desc()));
    MKLDNNStream::Get()->RegisterPrim(
        lrn_forward(pdesc, mkldnn::primitive::at(*src_mem),
            *ws_mem, *dst_mem));
    MKLDNNStream::Get()->Submit();
  } else {
    MKLDNNStream::Get()->RegisterPrim(
        lrn_forward(pdesc, mkldnn::primitive::at(*src_mem), *dst_mem));
    MKLDNNStream::Get()->Submit();
  }
}

void MKLDNNLRN_Backward(const OpContext &ctx, const LRNParam &param,
                            const NDArray &out_grad,
                            const NDArray &in_data,
                            const OpReqType &req,
                            const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }
  // Repeat FW for getting workspace
  auto data_mem = in_data.GetMKLDNNData();
  auto data_md = data_mem->get_primitive_desc().desc();
  auto pdesc_fwd = GetLRNFwd(param, ctx.is_train, data_md);

  // workspace to share
  std::shared_ptr<const mkldnn::memory> ws_mem(
          new mkldnn::memory(pdesc_fwd.workspace_primitive_desc()));
  std::shared_ptr<const mkldnn::memory> dst_temp(
          new mkldnn::memory(pdesc_fwd.dst_primitive_desc()));
  MKLDNNStream::Get()->RegisterPrim(
          lrn_forward(pdesc_fwd, mkldnn::primitive::at(*data_mem),
          *ws_mem, *dst_temp));

  auto data_in_md = pdesc_fwd.src_primitive_desc().desc();
  auto diff_mem = out_grad.GetMKLDNNData();
  auto diff_md = diff_mem->get_primitive_desc().desc();
  auto pdesc_bwd = GetLRNBwd(param, data_in_md, diff_md, pdesc_fwd);

  auto diff_src_mem = CreateMKLDNNMem(in_grad,
          pdesc_bwd.diff_src_primitive_desc(), req);

  MKLDNNStream::Get()->RegisterPrim(
        lrn_backward(pdesc_bwd, mkldnn::primitive::at(*data_mem),
        mkldnn::primitive::at(*diff_mem), *ws_mem, *diff_src_mem.second));
  MKLDNNStream::Get()->Submit();
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_LRN_INL_H__
