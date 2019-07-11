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
 * \file np_init_op.cc
 * \brief Function definition of numpy init op
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_INIT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_INIT_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

template<int req>
struct identity {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const int n) {
    using namespace mxnet_op;

    auto j = unravel(i, mshadow::Shape2(n, n));
    if (j[0] == j[1]) {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(1));
    } else {
      KERNEL_ASSIGN(out_data[i], req, static_cast<DType>(0));
    }
  }
};

template<typename xpu>
void IdentityCompute(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 0U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_data = outputs[0];
  int n = out_data.shape_[0];
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<identity<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), n);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_INIT_OP_INL_H_
