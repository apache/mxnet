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
 * \file np_kron-inl.h
 * \brief Function definition of matrix numpy-compatible kron operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_KRON_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_KRON_INL_H_

#include <vector>
#include "np_tensordot_op-inl.h"

namespace mxnet {
namespace op {

template<typename xpu>
inline void KronOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];

  TensordotIntAxesImpl<xpu>(0, ctx, a, b, out, req[0]);
}

template<typename xpu>
inline void KronOpBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;

  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& ograd = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];

  TensordotIntAxesBackwardImpl<xpu>(0, ctx, ograd, a, b, grad_a, grad_b, req);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_KRON_INL_H_
