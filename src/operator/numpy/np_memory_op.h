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
 *  Copyright (c) 2019 by Contributors
 * \file np_memory_op.h
 * \brief Function definition of numpy memory op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_MEMORY_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_MEMORY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include "../operator_common.h"

namespace mxnet {
namespace op {

template<typename xpu>
void NumpyShareMemoryCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  Tensor<xpu, 1, bool> outdata = outputs[0].FlatTo1D<xpu, bool>(s);

  if (a.Size() == 0 || b.Size() == 0) {
    ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, false);
    return;
  }
  MSHADOW_TYPE_SWITCH_WITH_BOOL(a.type_flag_, AType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(b.type_flag_, BType, {
      uint64_t start1 = reinterpret_cast<uint64_t>(a.dptr_);
      uint64_t end1 = start1 + a.Size() * sizeof(AType);
      uint64_t start2 = reinterpret_cast<uint64_t>(b.dptr_);
      uint64_t end2 = start2 + b.Size() * sizeof(BType);
      if (!(start1 < end2 && start2 < end1 && start1 < end1 && start2 < end2)) {
        ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, false);
      } else {
        ASSIGN_DISPATCH(outdata, OpReqType::kWriteTo, true);
      }
    });
  });
  return;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_MEMORY_OP_H_
