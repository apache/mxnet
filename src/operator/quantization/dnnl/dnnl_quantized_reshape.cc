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
 * \file dnnl_quantized_reshape.cc
 * \author: Adam Grabowski, adam.grabowski@intel.com
 */

#if MXNET_USE_ONEDNN == 1
#include "operator/quantization/quantized_reshape-inl.h"
#include "operator/nn/dnnl/dnnl_reshape-inl.h"

namespace mxnet {
namespace op {

static void DNNLQuantizedReshapeForward(const nnvm::NodeAttrs& attrs,
                                        const OpContext& ctx,
                                        const std::vector<NDArray>& inputs,
                                        const std::vector<OpReqType>& req,
                                        const std::vector<NDArray>& outputs) {
  CHECK(inputs[0].dtype() == mshadow::kUint8 || inputs[0].dtype() == mshadow::kInt8)
      << "dnnl_quantized_reshape op only supports uint8 and int8 as input type";

  if (SupportDNNLReshape(inputs[0])) {
    OpReqType reqType;
    if (inputs[0].GetDNNLData()->get_data_handle() != outputs[0].GetDNNLData()->get_data_handle())
      reqType = kWriteTo;
    else
      reqType = req[0];
    DNNLRun(DNNLReshapeForward, attrs, ctx, inputs[0], reqType, outputs[0]);
  } else {
    FallBackCompute(UnaryOp::IdentityCompute<cpu>, attrs, ctx, inputs, req, outputs);
  }

  *outputs[1].data().dptr<float>() = *inputs[1].data().dptr<float>();
  *outputs[2].data().dptr<float>() = *inputs[2].data().dptr<float>();
}

inline bool QuantizedReshapeStorageType(const nnvm::NodeAttrs& attrs,
                                        const int dev_mask,
                                        DispatchMode* dispatch_mode,
                                        std::vector<int>* in_attrs,
                                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_contrib_quantized_reshape)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedReshapeForward)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedReshapeStorageType);

NNVM_REGISTER_OP(_npx_quantized_reshape)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLQuantizedReshapeForward)
    .set_attr<FInferStorageType>("FInferStorageType", QuantizedReshapeStorageType);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
