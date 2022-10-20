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
 * \file dnnl_binary-inl.h
 * \author: Adam Grabowski, adam.grabowski@intel.com
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_BINARY_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_BINARY_INL_H_

#if MXNET_USE_ONEDNN == 1
#include "./dnnl_base-inl.h"
#include <vector>

#include "../../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

using binary_fwd_t    = dnnl::binary;
using binary_fwd_pd_t = dnnl::binary::primitive_desc;

class DNNLBinaryOpFwd {
 public:
  template <dnnl::algorithm alg>
  static DNNLBinaryOpFwd& GetBinaryOpForward(const std::vector<NDArray>& inputs,
                                             const std::vector<NDArray>& outputs);
  DNNLBinaryOpFwd(const dnnl::algorithm alg,
                  const std::vector<NDArray>& inputs,
                  const std::vector<NDArray>& outputs);

  void Execute(const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  std::shared_ptr<binary_fwd_t> fwd;
  std::shared_ptr<binary_fwd_pd_t> fwd_pd;
};

template <dnnl::algorithm alg>
DNNLBinaryOpFwd& DNNLBinaryOpFwd::GetBinaryOpForward(const std::vector<NDArray>& inputs,
                                                     const std::vector<NDArray>& outputs) {
  using binary_op_fwd_map = std::unordered_map<OpSignature, DNNLBinaryOpFwd, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local binary_op_fwd_map fwds;
#else
  static MX_THREAD_LOCAL binary_op_fwd_map fwds;
#endif
  OpSignature key;
  key.AddSign(static_cast<int>(alg));
  key.AddSign(inputs[0]);
  key.AddSign(inputs[1]);
  key.AddSign(outputs[0]);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    const DNNLBinaryOpFwd fwd(alg, inputs, outputs);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_BINARY_INL_H_
