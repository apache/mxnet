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
 * \file mkldnn_quantized_pooling.cc
 * \brief
 * \author Tao Lv, Xinyu Chen
*/

#if MXNET_USE_MKLDNN == 1

#include "../../nn/mkldnn/mkldnn_pooling-inl.h"

namespace mxnet {
namespace op {

static void MKLDNNQuantizedPoolingForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                                          const std::vector<NDArray> &in_data,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &out_data) {
  CHECK(in_data[0].dtype() == mshadow::kUint8
    || in_data[0].dtype() == mshadow::kInt8)
    << "mkldnn_quantized_pooling op only supports uint8 and int8 as input type";
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  auto fwd = GetPoolingFwd(param, ctx.is_train, in_data[0], out_data[0]);
  fwd.SetNewMem(in_data[0], out_data[0], req[0]);
  fwd.Execute(out_data[0]);
  out_data[1].data().dptr<float>()[0] = in_data[1].data().dptr<float>()[0];
  out_data[2].data().dptr<float>()[0] = in_data[2].data().dptr<float>()[0];
}

NNVM_REGISTER_OP(_contrib_quantized_pooling)
.set_attr<FComputeEx>("FComputeEx<cpu>", MKLDNNQuantizedPoolingForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
