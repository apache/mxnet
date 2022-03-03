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
 * \file dnnl_power-inl.h
 * \author: Adam Grabowski, adam.grabowski@intel.com
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_POWER_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_POWER_INL_H_

#if MXNET_USE_ONEDNN == 1

#include "dnnl_base-inl.h"
#include "dnnl_ops-inl.h"
#include "operator/tensor/elemwise_binary_scalar_op.h"
#include "operator/tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

using eltwise_fwd_t    = dnnl::eltwise_forward;
using eltwise_fwd_pd_t = dnnl::eltwise_forward::primitive_desc;

class DNNLPowerFwd {
 public:
  static DNNLPowerFwd& GetPowerForward(const nnvm::NodeAttrs& attrs,
                                       const NDArray& input,
                                       const NDArray& outputs);

  DNNLPowerFwd(const NDArray& input, const float exponent);

  void Execute(const NDArray& input, const OpReqType& req, const NDArray& output);

 private:
  std::shared_ptr<eltwise_fwd_t> fwd;
  std::shared_ptr<eltwise_fwd_pd_t> fwd_pd;
};

typedef OpSignature DNNLPowerSignature;

void DNNLPowerForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const NDArray& input,
                      const OpReqType& req,
                      const NDArray& output);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_POWER_INL_H_
