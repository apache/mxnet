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
 * \file dnnl_quantized_act.cc
 * \brief DNNL(Quantized) Activation operator based on subgraph
 * /author Zhiyuan Huang
 */
#if MXNET_USE_ONEDNN == 1

#include "operator/nn/activation-inl.h"
#include "operator/nn/dnnl/dnnl_act-inl.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

static void DNNLQuantizedActForward(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& in_data,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& out_data) {
  CHECK(in_data[0].dtype() == mshadow::kUint8 || in_data[0].dtype() == mshadow::kInt8)
      << "_contrib_quantized_act op only supports uint8 and int8 as input "
         "type";

  DNNLRun(DNNLActivationForward, attrs, ctx, in_data[0], req[0], out_data[0]);
  out_data[1].data().dptr<float>()[0] = in_data[1].data().dptr<float>()[0];
  out_data[2].data().dptr<float>()[0] = in_data[2].data().dptr<float>()[0];
}

NNVM_REGISTER_OP(_contrib_quantized_act)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedActForward);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
