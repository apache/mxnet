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
 * \file dnnl_eltwise-inl.h
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_ELTWISE_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_ELTWISE_INL_H_

#if MXNET_USE_ONEDNN == 1

#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/operator_common.h"

namespace mxnet {
namespace op {

using eltwise_fwd_t    = dnnl::eltwise_forward;
using eltwise_fwd_pd_t = dnnl::eltwise_forward::primitive_desc;

class DNNLEltwiseFwd {
 public:
  typedef OpSignature DNNLEltwiseSignature;

  static DNNLEltwiseFwd& GetCached(const NDArray& input,
                                   const NDArray& output,
                                   const dnnl::algorithm algorithm);

  explicit DNNLEltwiseFwd(const NDArray& input, const dnnl::algorithm algorithm);

  void Execute(const NDArray& input, const OpReqType& req, const NDArray& output);

 private:
  std::shared_ptr<eltwise_fwd_t> fwd;
  std::shared_ptr<eltwise_fwd_pd_t> fwd_pd;
};

template <dnnl::algorithm algorithm>
inline void DNNLEltwiseForward(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const NDArray& input,
                               const OpReqType& req,
                               const NDArray& output) {
  DNNLEltwiseFwd& fwd = DNNLEltwiseFwd::GetCached(input, output, algorithm);
  fwd.Execute(input, req, output);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_ELTWISE_INL_H_
