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
 * Copyright (c) 2018 by Contributors
 * \file dynamic_shape_ops-inl.h
*/

#ifndef MXNET_OPERATOR_CONTRIB_DYNAMIC_SHAPE_OPS_INL_H_
#define MXNET_OPERATOR_CONTRIB_DYNAMIC_SHAPE_OPS_INL_H_

#include <vector>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

template<typename xpu>
inline void DynamicReshapeForward(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const std::vector<NDArray> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  const NDArray &out = outputs[0];
  const NDArray &idx = inputs[1];
  size_t idx_size = idx.shape()[0];
  mxnet::TShape shape_value = mxnet::TShape(idx_size, 0);
  std::vector<index_t> shapev(idx_size, 0);

  // Copy the target shape that is provided in inputs[1]
  // to the vector shapev
  MSHADOW_TYPE_SWITCH(idx.dtype(), DType, {
    DType* idx_dptr = idx.data().dptr<DType>();
    for (size_t i = 0; i < idx_size; i++) {
      shapev[i] = static_cast<index_t>(idx_dptr[i]);
    }
  });
  shape_value = InferReshapeShape(mxnet::Tuple<index_t>(shapev), inputs[0].shape(), false);
  const_cast<NDArray &>(out).Init(shape_value);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(out.dtype(), DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
          s, inputs[0].data().Size(), out.data().dptr<DType>(),
          inputs[0].data().dptr<DType>());
    });
  });
}

template<typename xpu>
inline void DynamicReshapeBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const std::vector<NDArray> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
          s, inputs[0].data().Size(), outputs[0].data().dptr<DType>(),
          inputs[0].data().dptr<DType>());
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_DYNAMIC_SHAPE_OPS_INL_H_
