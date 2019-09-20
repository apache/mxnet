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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_act-inl.h
 * \brief MKLDNN Activation operator
 * /author Zhiyuan Huang
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ACT_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ACT_INL_H_


#if MXNET_USE_MKLDNN == 100
#include <vector>
#include <utility>
#include "../activation-inl.h"

namespace mxnet {
namespace op {

mkldnn::algorithm GetMKLDNNActAlgo(const ActivationParam& param);
mkldnn::eltwise_forward::primitive_desc GetActFwdDescImpl(
    const ActivationParam& param, bool is_train, const mkldnn::memory &input_mem);

class MKLDNNActForward {
 public:
  const mkldnn::eltwise_forward::primitive_desc fwd_pd;

  MKLDNNActForward(const ActivationParam& param, bool is_train,
                   const NDArray &data, const mkldnn::memory &mem): fwd_pd(
                       GetActFwdDescImpl(param, is_train, mem)) {
    fwd_ = std::make_shared<mkldnn::eltwise_forward>(fwd_pd);
  }
  const inline mkldnn::eltwise_forward &GetFwd() const;

 private:
  std::shared_ptr<mkldnn::eltwise_forward> fwd_;
};

typedef ParamOpSign<ActivationParam> MKLDNNActSignature;
MKLDNNActForward &GetActForward(const ActivationParam& param,
                                const OpContext &ctx, const NDArray &in_data,
                                const mkldnn::memory &in_mem);

void MKLDNNActivationForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                             const NDArray &in_data, const OpReqType &req,
                             const NDArray &out_data);

mkldnn::eltwise_backward::primitive_desc GetActBwdDescImpl(
    const ActivationParam &param, const mkldnn::memory &input_mem,
    const mkldnn::memory &diff_dst_memory);

class MKLDNNActBackward {
 public:
  const mkldnn::eltwise_backward::primitive_desc pd;

  explicit MKLDNNActBackward(const ActivationParam &param, const NDArray &data,
                             const mkldnn::memory &mem,
                             const mkldnn::memory &diff_dst_memory): pd(
                                 GetActBwdDescImpl(param, mem, diff_dst_memory)) {
    bwd = std::make_shared<mkldnn::eltwise_backward>(pd);
  }
  const inline mkldnn::eltwise_backward &GetBwd() const;

 private:
  std::shared_ptr<mkldnn::eltwise_backward> bwd;
};
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 100
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ACT_INL_H_
