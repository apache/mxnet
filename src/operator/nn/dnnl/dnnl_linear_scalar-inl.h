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
 * \file dnnl_linear_scalar-inl.h
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_LINEAR_SCALAR_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_LINEAR_SCALAR_INL_H_

#if MXNET_USE_ONEDNN == 1

#include "operator/tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

using eltwise_fwd_t    = dnnl::eltwise_forward;
using eltwise_fwd_pd_t = dnnl::eltwise_forward::primitive_desc;

template <typename OP>
inline void GetMultiplierAndComponent(float& multiplier, float& component, const float scalar);
template <>
inline void GetMultiplierAndComponent<op::mshadow_op::plus>(float& multiplier,
                                                            float& component,
                                                            const float scalar) {
  component = scalar;
}
template <>
inline void GetMultiplierAndComponent<op::mshadow_op::minus>(float& multiplier,
                                                             float& component,
                                                             const float scalar) {
  component = -scalar;
}
template <>
inline void GetMultiplierAndComponent<op::mshadow_op::rminus>(float& multiplier,
                                                              float& component,
                                                              const float scalar) {
  component  = scalar;
  multiplier = -1;
}
template <>
inline void GetMultiplierAndComponent<op::mshadow_op::mul>(float& multiplier,
                                                           float& component,
                                                           const float scalar) {
  multiplier = scalar;
}

class DNNLLinearScalarFwd {
 public:
  typedef OpSignature DNNLLinearScalarSignature;

  template <typename OP>
  static DNNLLinearScalarFwd& GetLinearScalarForward(const nnvm::NodeAttrs& attrs,
                                                     const NDArray& input,
                                                     const NDArray& output) {
    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
#if DMLC_CXX11_THREAD_LOCAL
    static thread_local std::unordered_map<DNNLLinearScalarSignature, DNNLLinearScalarFwd, OpHash>
        fwds;
#else
    static MX_THREAD_LOCAL
        std::unordered_map<DNNLLinearScalarSignature, DNNLLinearScalarFwd, OpHash>
            fwds;
#endif
    float multiplier = 1, component = 0;  // set to neutral values
    GetMultiplierAndComponent<OP>(multiplier, component, static_cast<float>(param.scalar));
    DNNLLinearScalarSignature key;
    key.AddSign(multiplier);
    key.AddSign(component);
    key.AddSign(input);
    key.AddSign(output);

    auto it = fwds.find(key);
    if (it == fwds.end()) {
      const DNNLLinearScalarFwd fwd(input, multiplier, component);
      it = AddToCache(&fwds, key, fwd);
    }
    return it->second;
  }

  DNNLLinearScalarFwd(const NDArray& input, const float multiplier, const float component);

  void Execute(const NDArray& input, const OpReqType& req, const NDArray& output);

 private:
  std::shared_ptr<eltwise_fwd_t> fwd;
  std::shared_ptr<eltwise_fwd_pd_t> fwd_pd;
};

template <typename OP>
void DNNLLinearScalarForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const NDArray& input,
                             const OpReqType& req,
                             const NDArray& output) {
  DNNLLinearScalarFwd& fwd = DNNLLinearScalarFwd::GetLinearScalarForward<OP>(attrs, input, output);
  fwd.Execute(input, req, output);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_LINEAR_SCALAR_INL_H_
