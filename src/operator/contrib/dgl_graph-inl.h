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

#ifndef MXNET_OPERATOR_CONTRIB_DGL_GRAPH_INL_H_
#define MXNET_OPERATOR_CONTRIB_DGL_GRAPH_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

template<typename xpu>
void DGLAdjacencyForwardEx(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(inputs[0].storage_type(), kCSRStorage);
  CHECK_EQ(outputs[0].storage_type(), kCSRStorage);
  const TBlob &in_idx = inputs[0].aux_data(csr::kIdx);
  const TBlob &in_indptr = inputs[0].aux_data(csr::kIndPtr);

  outputs[0].CheckAndAllocData(in_idx.shape_);
  outputs[0].CheckAndAllocAuxData(csr::kIdx, in_idx.shape_);
  outputs[0].CheckAndAllocAuxData(csr::kIndPtr, in_indptr.shape_);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  Fill<false>(s, outputs[0].data(), req[0], 1.0);
  mxnet_op::copy(s, outputs[0].aux_data(csr::kIdx), in_idx);
  mxnet_op::copy(s, outputs[0].aux_data(csr::kIndPtr), in_indptr);
}

}
}

#endif  // MXNET_OPERATOR_CONTRIB_DGL_GRAPH_INL_H_
