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
 * \brief MKLDNN(Quantized) Activation operator based on subgraph
 * /author Zhiyuan Huang
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ACT_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ACT_INL_H_


#if MXNET_USE_MKLDNN == 1
#include <vector>
#include <utility>
#include "../activation-inl.h"
#include "../../leaky_relu-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

struct MKLDNNActParam {
  mkldnn::algorithm alg;
  float slope = 0.f;

  bool operator==(const MKLDNNActParam& other) const {
    return this->alg == other.alg &&
           this->slope == other.slope;
  }
};

mkldnn::algorithm GetMKLDNNActAlgo(const ActivationParam& param);
mkldnn::algorithm GetMKLDNNActAlgo(const LeakyReLUParam& param);
mkldnn::eltwise_forward::primitive_desc GetActFwdDescImpl(
    const MKLDNNActParam& param, bool is_train,
    const mkldnn::memory &input_mem, int dtype);

class MKLDNNActForward {
 public:
  const mkldnn::eltwise_forward::primitive_desc fwd_pd;

  MKLDNNActForward(const MKLDNNActParam& param, bool is_train,
                   const NDArray &data, const mkldnn::memory &mem): fwd_pd(
                       GetActFwdDescImpl(param, is_train, mem, data.dtype())) {}
  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &output);
  const mkldnn::eltwise_forward &GetFwd() const;

 private:
  std::shared_ptr<mkldnn::eltwise_forward> fwd_;
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;
};

typedef ParamOpSign<MKLDNNActParam> MKLDNNActSignature;
MKLDNNActForward &GetActForward(const MKLDNNActParam& param,
                                const OpContext &ctx, const NDArray &in_data,
                                const mkldnn::memory &in_mem);

void MKLDNNActivationForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                             const NDArray &in_data, const OpReqType &req,
                             const NDArray &out_data);
void MKLDNNLeakyReluForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                             const NDArray &in_data, const OpReqType &req,
                             const NDArray &out_data);
}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::MKLDNNActParam> {
  size_t operator()(const mxnet::op::MKLDNNActParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, static_cast<size_t>(val.alg));
    ret = dmlc::HashCombine(ret, val.slope);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ACT_INL_H_
