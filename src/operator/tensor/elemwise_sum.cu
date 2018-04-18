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
 * Copyright (c) 2015 by Contributors
 * \file elemwise_sum.cu
 * \brief elementwise sum operator
*/
#include "./elemwise_sum.h"
#include "../../ndarray/ndarray_function.h"

namespace mxnet {
namespace op {

void ElementWiseSumComputeExGPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  CHECK(!inputs.empty());
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  CHECK_EQ(req[0], kWriteTo) << "ElementWiseSumComputeExGPU only supports req = kWriteTo";
  if (inputs[0].storage_type() == kRowSparseStorage) {
    mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
    NDArray out_nd = outputs[0];
    mxnet::ndarray::ElementwiseSum<gpu>(s, ctx.requested[0], inputs, &out_nd);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

NNVM_REGISTER_OP(add_n)
.set_attr<FCompute>("FCompute<gpu>", ElementWiseSumComputeWithHalf2<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ElementWiseSumComputeExGPU);

}  // namespace op
}  // namespace mxnet
