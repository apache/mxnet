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
 * \file elemwise_binary_broadcast_op-inl.cuh
 * \brief CUDA specific Function definition of elementwise binary broadcast operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_CUH_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_CUH_
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "broadcast_reduce-inl.h"
namespace mxnet {
namespace op {
template<typename xpu, typename LOP, typename ROP>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, void>::type
BinaryBroadcastBackwardUseNone(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace broadcast;
  TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryOp::BackwardUseNone<gpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Stream<gpu> *s = ctx.get_stream<gpu>();
      const TBlob lhs = outputs[0].reshape(new_lshape);
      const TBlob rhs = outputs[1].reshape(new_rshape);
      const TBlob out = inputs[0].reshape(new_oshape);
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        // Request temporary storage
        size_t workspace_size = new_oshape.Size();
        Tensor<gpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<gpu, 1, char>(
                Shape1(workspace_size * sizeof(index_t)), s);
        Reduce<red::sum, NDim, DType, LOP>(s, lhs, req[0], workspace, out);
        Reduce<red::sum, NDim, DType, ROP>(s, rhs, req[1], workspace, out);
      });
    });
  }
}
} // namespace op
} // namespace mxnet
#endif
