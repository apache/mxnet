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
 * \file np_dot-inl.h
 * \brief Function definition of matrix numpy-compatible dot operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_DOT_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_DOT_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../tensor/dot-inl.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/broadcast_reduce_op.h"
#include "np_tensordot_op-inl.h"

namespace mxnet {
namespace op {

template<typename xpu>
inline void NumpyDotForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    if (b_shape.ndim() < 3) {
      // Case 1, 2, 3, 4, 5: a is N-D array (N >= 1) and b is vector or matrix, sum product
      //        over the last axis of a and the first axis of b
      TensordotIntAxesImpl<xpu>(1, ctx, a, b, out, req[0]);
    } else {
      // Case 3, 5.5: a is N-D array and b is M-D array (M > 2), sum product over the last axis
      //         of a and the 2nd-to-last axis of b
      const Tuple<int> a_axes_summed({a_shape.ndim() - 1});
      const Tuple<int> b_axes_summed({b_shape.ndim() - 2});
      size_t workspace_size = TensordotWorkspaceSize<xpu>(a_axes_summed,
                                                          b_axes_summed,
                                                          a, b, out,
                                                          req);
      Tensor<xpu, 1, char> workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(
        Shape1(workspace_size), ctx.get_stream<xpu>());
      TensordotImpl<xpu>(a_axes_summed, b_axes_summed, ctx, a, b, out, req, workspace);
    }
  });
}

template<typename xpu>
inline void NumpyDotBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& ograd = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;

  MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, DType, {
    if (b_shape.ndim() < 3) {
      // Case 1, 2, 3, 4, 5: a is N-D array (N >= 1) and b is vector or matrix, sum product
      //        over the last axis of a and the first axis of b
      TensordotIntAxesBackwardImpl<xpu>(1, ctx, ograd, a, b, grad_a, grad_b, req);
    } else {
      // Case 3, 5.5: a is N-D array and b is M-D array (M > 2), sum product over the last axis
      //         of a and the 2nd-to-last axis of b
      const Tuple<int> a_axes_summed({a_shape.ndim() - 1});
      const Tuple<int> b_axes_summed({b_shape.ndim() - 2});
      size_t workspace_size = TensordotBackwardWorkspaceSize<xpu>(a_axes_summed, b_axes_summed,
                                                                  ograd, a, b, grad_a,
                                                                  grad_b, req);
      Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size),
                                                       ctx.get_stream<xpu>());
      TensordotBackwardImpl<xpu>(a_axes_summed, b_axes_summed, ctx, ograd, a, b, grad_a,
                                 grad_b, req, workspace);
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_DOT_INL_H_
