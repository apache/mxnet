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
 * \file np_dot_forward.cc
 * \brief CPU Implementation of numpy-compatible dot
 */

#include "np_dot-inl.h"
#if MXNET_USE_ONEDNN == 1
#include "operator/nn/dnnl/dnnl_dot-inl.h"
#endif

namespace mxnet {
namespace op {

inline bool NumpyDotShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector* in_attrs,
                          mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);

  if (!ndim_is_known(a_shape) || !ndim_is_known(b_shape)) {
    return false;
  }

  if (a_shape.ndim() == 1 && b_shape.ndim() == 1) {
    // Case 1: both 1-D arrays, inner product of vectors
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, in_attrs->at(1));
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, in_attrs->at(0));
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
  } else if (a_shape.ndim() == 2 && b_shape.ndim() == 2) {
    // Case 2: both 2-D arrays, matrix multiplication
    mxnet::TShape tmp_shape(2, -1);
    tmp_shape[1] = b_shape[0];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tmp_shape);

    tmp_shape[0] = a_shape[1];
    tmp_shape[1] = -1;
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);

    tmp_shape[0] = a_shape[0];
    tmp_shape[1] = b_shape[1];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tmp_shape);
  } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
    // Case 3 + 3.5: either of them is a scalar, just scale by one of them
    mxnet::TShape oshape = (a_shape.ndim() == 0) ? b_shape : a_shape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  } else if (b_shape.ndim() == 1) {
    // Case 4: a is N-D array and b is 1-D array, sum product over the last axis
    TShape tmp_shape(a_shape.ndim(), -1);
    tmp_shape[a_shape.ndim() - 1] = b_shape[0];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tmp_shape);

    tmp_shape    = TShape(1, -1);
    tmp_shape[0] = a_shape[a_shape.ndim() - 1];
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);

    mxnet::TShape out_shape(a_shape.ndim() - 1, -1);
    for (int i = 0; i < a_shape.ndim() - 1; ++i) {
      out_shape[i] = a_shape[i];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  } else {
    // Case 5: a is N-D array and b is M-D array, sum product over the last axis
    //         of a and the 2nd-to-last axis of b
    TShape tmp_shape(a_shape.ndim(), -1);
    tmp_shape[a_shape.ndim() - 1] = b_shape[b_shape.ndim() - 2];
    SHAPE_ASSIGN_CHECK(*in_attrs, 0, tmp_shape);

    tmp_shape                     = TShape(b_shape.ndim(), -1);
    tmp_shape[b_shape.ndim() - 2] = a_shape[a_shape.ndim() - 1];
    SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);

    tmp_shape = TShape(a_shape.ndim() + b_shape.ndim() - 2, -1);
    for (int i = 0; i < a_shape.ndim() - 1; ++i) {
      tmp_shape[i] = a_shape[i];
    }
    for (int i = 0; i < b_shape.ndim() - 2; ++i) {
      tmp_shape[i + a_shape.ndim() - 1] = b_shape[i];
    }
    tmp_shape[tmp_shape.ndim() - 1] = b_shape[b_shape.ndim() - 1];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, tmp_shape);
  }
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

#if MXNET_USE_ONEDNN == 1
static void NumpyDotComputeExCPU(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
  if (SupportDNNLDot(inputs)) {
    DNNL_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    DNNLRun(DNNLDotForward<true>, attrs, ctx, inputs, req, outputs);
    DNNL_OPCHECK_RUN(NumpyDotForward<cpu>, attrs, ctx, inputs, req, outputs);
  } else {
    FallBackCompute(NumpyDotForward<cpu>, attrs, ctx, inputs, req, outputs);
  }
}

inline static bool NumpyDotStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}
#endif

NNVM_REGISTER_OP(_npi_dot)
    .describe(R"doc(Dot product of two arrays. Specifically,

- If both a and b are 1-D arrays, it is inner product of vectors.

- If both a and b are 2-D arrays, it is matrix multiplication.

- If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.

- If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.

- If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:

  Example ::

    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

)doc" ADD_FILELINE)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a", "b"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyDotShape)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>(1,
                                                                      ResourceRequest::kTempSpace);
                                })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyDotForward<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<FComputeEx>("FComputeEx<cpu>", NumpyDotComputeExCPU)
    .set_attr<FInferStorageType>("FInferStorageType", NumpyDotStorageType)
#endif
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_dot"})
    .add_argument("a", "NDArray-or-Symbol", "First input")
    .add_argument("b", "NDArray-or-Symbol", "Second input");

}  // namespace op
}  // namespace mxnet
