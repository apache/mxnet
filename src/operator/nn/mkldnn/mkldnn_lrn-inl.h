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

inline algorithm GetMKLDNNLRNAlgo(const LRNParam &param) {
  // TODO(Patric): lrn_within_channel will cause core dump in MKLDNN backward
  //               Need to confirm with MKLDNN team and fix later
  return algorithm::lrn_across_channels;
}

inline lrn_forward::primitive_desc GetLRNFwd(const LRNParam &param,
                                             const bool is_train,
                                             const memory::desc &src_md) {
  const auto  engine = CpuEngine::Get()->get_engine();
  const auto  alg = GetMKLDNNLRNAlgo(param);
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
          const mkldnn::memory::desc &diff_in_md,
          const mkldnn::memory::desc &diff_md,
          const lrn_forward::primitive_desc &lrnFwd_desc) {
  const auto engine = CpuEngine::Get()->get_engine();
  const auto alg = GetMKLDNNLRNAlgo(param);
  const float alpha = param.alpha;
  const float beta = param.beta;
  const int nsize = param.nsize;
  const float k = param.knorm;

  lrn_backward::desc lrnBwd_desc(alg, diff_in_md,
                diff_md, nsize, alpha, beta, k);
  return mkldnn::lrn_backward::primitive_desc(lrnBwd_desc,
                               engine, lrnFwd_desc);
}

void MKLDNNLRNForward(const OpContext &ctx,
                      const LRNParam &param,
                      const NDArray &in_data,
                      const OpReqType req,
                      const NDArray &out_data) {
  auto src_mem = in_data.GetMKLDNNData();
  const auto src_md = src_mem->get_primitive_desc().desc();
  const auto pdesc = GetLRNFwd(param, ctx.is_train, src_md);
  auto dst_mem = const_cast<NDArray &>(out_data).CreateMKLDNNData(
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

void MKLDNNLRNBackward(const OpContext &ctx, const LRNParam &param,
                       const NDArray &out_grad,
                       const NDArray &in_data,
                       const OpReqType req,
                       const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }
  // Repeat FW for getting workspace
  auto data_mem = in_data.GetMKLDNNData();
  const auto data_md = data_mem->get_primitive_desc().desc();
  const auto pdesc_fwd = GetLRNFwd(param, ctx.is_train, data_md);

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

  const auto data_in_md = pdesc_fwd.src_primitive_desc().desc();
  auto diff_mem = out_grad.GetMKLDNNData();
  const auto diff_md = diff_mem->get_primitive_desc().desc();
  const auto pdesc_bwd = GetLRNBwd(param, data_in_md, diff_md, pdesc_fwd);
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
