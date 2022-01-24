
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
 * \file dnnl_quantized_transpose.cc
 * \author: Rafal Litka, rafal.litka@intel.com
 */
#if MXNET_USE_ONEDNN == 1
#include "../../nn/dnnl/dnnl_transpose-inl.h"
#include "../../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

inline static bool QuantizedTransposeStorageType(const nnvm::NodeAttrs& attrs,
                                                 const int dev_mask,
                                                 DispatchMode* dispatch_mode,
                                                 std::vector<int>* in_attrs,
                                                 std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

bool SupportDNNLQuantizedTranspose(const NDArray& data) {
  auto data_ndim = data.shape().ndim();

  if (data_ndim > 4 || data_ndim == 0 || data.shape().Size() == 0)
    return false;

  return true;
}

static void DNNLQuantizedTransposeForward(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
  CHECK(inputs[0].dtype() == mshadow::kUint8 || inputs[0].dtype() == mshadow::kInt8)
      << "dnnl_quantized_transpose only supports uint8 and int8 as input type";
  if (req[0] == kNullOp) {
    return;
  }
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);
  if (SupportDNNLQuantizedTranspose(inputs[0])) {
    DNNLRun(DNNLTransposeForward<TransposeParam>, attrs, ctx, inputs[0], req[0], outputs[0]);
  } else {
    FallBackCompute(UnaryOp::IdentityCompute<cpu>, attrs, ctx, inputs, req, outputs);
  }
  outputs[1].data().dptr<float>()[0] = inputs[1].data().dptr<float>()[0];
  outputs[2].data().dptr<float>()[0] = inputs[2].data().dptr<float>()[0];
}

NNVM_REGISTER_OP(_contrib_quantized_transpose)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedTransposeStorageType)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& n) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedTransposeForward)
    .set_attr<bool>("TIsDNNL", true);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
