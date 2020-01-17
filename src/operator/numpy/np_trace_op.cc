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
 * \file np_trace_op.cc
 * \brief CPU Implementation of numpy-compatible trace operator
 */

#include "./np_trace_op-inl.h"

namespace mxnet {
namespace op {

inline bool NumpyTraceOpShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_attrs,
                              mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const int ndim((*in_attrs)[0].ndim());
  if (ndim < 2) {
    return false;
  }
  std::vector<int> oshape(ndim - 2);
  const NumpyTraceParam& param = nnvm::get<NumpyTraceParam>(attrs.parsed);
  int x1 = CheckAxis(param.axis1, (*in_attrs)[0].ndim());
  int x2 = CheckAxis(param.axis2, (*in_attrs)[0].ndim());
  CHECK_NE(x1, x2) << "axis1 and axis2 cannot refer to the the same axis " << x1;
  for ( int i = 0, j = 0; i < ndim; ++i ) {
    if (i != x1 && i != x2) {
      oshape[j++] = (*in_attrs)[0][i];
    }
  }
  mxnet::TShape tshape(oshape.begin(), oshape.end());
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, tshape);
  return true;
}

DMLC_REGISTER_PARAMETER(NumpyTraceParam);

NNVM_REGISTER_OP(_np_trace)
.describe(R"code(Computes the sum of the diagonal elements of a matrix.
Input is a tensor *A* of dimension *n >= 2*.

If *n=2*, we sum the diagonal elements. The result has shape ().

If *n>2*, *trace* is performed separately on the matrix defined by *axis1* and *axis2* for all
inputs (batch mode).

Examples::

   // Single matrix reduction
   A = [[1.0, 1.0], [1.0, 7.0]]
   trace(A) = 8.0

   // Batch matrix reduction
   A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
   trace(A) = [1.0, 18.0]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyTraceParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyTraceOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", NumpyTraceOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_np_trace"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyTraceParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_trace)
.set_attr_parser(ParamParser<NumpyTraceParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyTraceOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
