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
 * \file dnnl_act-inl.h
 * \brief DNNL Activation operator
 * /author Zhiyuan Huang
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_ACT_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_ACT_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <utility>
#include <vector>

#include "operator/leaky_relu-inl.h"
#include "operator/nn/activation-inl.h"

namespace mxnet {
namespace op {

struct DNNLActParam {
  dnnl::algorithm alg;
  float slope = 0.f;

  bool operator==(const DNNLActParam& other) const {
    return this->alg == other.alg && this->slope == other.slope;
  }
};

dnnl::algorithm GetDNNLActAlgo(const ActivationParam& param);
dnnl::algorithm GetDNNLActAlgo(const LeakyReLUParam& param);

dnnl::eltwise_forward::primitive_desc GetActFwdDescImpl(const DNNLActParam& param,
                                                        bool is_train,
                                                        const dnnl::memory& input_mem);

class DNNLActForward {
 public:
  const dnnl::eltwise_forward::primitive_desc fwd_pd;

  DNNLActForward(const DNNLActParam& param,
                 bool is_train,
                 const NDArray& data,
                 const dnnl::memory& mem)
      : fwd_pd(GetActFwdDescImpl(param, is_train, mem)) {
    fwd_ = std::make_shared<dnnl::eltwise_forward>(fwd_pd);
  }
  const inline dnnl::eltwise_forward& GetFwd() const;

 private:
  std::shared_ptr<dnnl::eltwise_forward> fwd_;
};

typedef ParamOpSign<DNNLActParam> DNNLActSignature;
DNNLActForward& GetActForward(const DNNLActParam& param,
                              const OpContext& ctx,
                              const NDArray& in_data,
                              const dnnl::memory& in_mem);

dnnl::eltwise_backward::primitive_desc GetActBwdDescImpl(const DNNLActParam& param,
                                                         const dnnl::memory& input_mem,
                                                         const dnnl::memory& diff_dst_memory);

class DNNLActBackward {
 public:
  const dnnl::eltwise_backward::primitive_desc bwd_pd;

  explicit DNNLActBackward(const DNNLActParam& param,
                           const NDArray& data,
                           const dnnl::memory& mem,
                           const dnnl::memory& diff_dst_memory)
      : bwd_pd(GetActBwdDescImpl(param, mem, diff_dst_memory)) {
    bwd_prim_ = std::make_shared<dnnl::eltwise_backward>(bwd_pd);
  }
  const inline dnnl::eltwise_backward& GetBwd() const;

 private:
  std::shared_ptr<dnnl::eltwise_backward> bwd_prim_;
};

void DNNLActivationForward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const NDArray& in_data,
                           const OpReqType& req,
                           const NDArray& out_data);

void DNNLActivationBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs);

void DNNLLeakyReluForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const NDArray& in_data,
                          const OpReqType& req,
                          const NDArray& out_data);

void DNNLLeakyReluBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs);

}  // namespace op
}  // namespace mxnet

namespace std {
template <>
struct hash<mxnet::op::DNNLActParam> {
  size_t operator()(const mxnet::op::DNNLActParam& val) {
    size_t ret = 0;
    ret        = dmlc::HashCombine(ret, static_cast<size_t>(val.alg));
    ret        = dmlc::HashCombine(ret, val.slope);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_ACT_INL_H_
