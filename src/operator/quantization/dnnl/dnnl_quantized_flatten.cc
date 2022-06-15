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
 * \file quantized_flatten.cc
 * \brief
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_reshape-inl.h"
#include "operator/quantization/quantization_utils.h"

namespace mxnet {
namespace op {

inline static bool FlattenStorageType(const nnvm::NodeAttrs& attrs,
                                      const int dev_mask,
                                      DispatchMode* dispatch_mode,
                                      std::vector<int>* in_attrs,
                                      std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

static void DNNLQuantizedFlattenForward(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& outputs) {
  if (SupportDNNL(inputs[0])) {
    DNNLRun(DNNLReshapeForward, attrs, ctx, inputs[0], req[0], outputs[0]);
  } else {
    FallBackCompute(UnaryOp::IdentityCompute<cpu>, attrs, ctx, inputs, req, outputs);
  }
  outputs[1].data().dptr<float>()[0] = inputs[1].data().dptr<float>()[0];
  outputs[2].data().dptr<float>()[0] = inputs[2].data().dptr<float>()[0];
}

NNVM_REGISTER_OP(_contrib_quantized_flatten)
    .set_attr<FInferStorageType>("FInferStorageType", FlattenStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedFlattenForward)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<bool>("TIsDNNL", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
