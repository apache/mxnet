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
 * Copyright (c) 2019 by Contributors
 * \file np_la_op.cc
 * \brief CPU implementation of Operators for advanced linear algebra.
 */

#include "./np_la_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyLaNormParam);

inline bool NumpyLaNormShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known((*in_attrs)[0])) return false;
  const NumpyLaNormParam& param = nnvm::get<NumpyLaNormParam>(attrs.parsed);
  const int ndim = (*in_attrs)[0].ndim();
  if ((!param.axis.has_value() && param.flag != 0 && ndim > 2) ||
      (param.axis.has_value() && param.axis.value().ndim() > 2))
    LOG(FATAL) << "Improper number of dimensions to norm.";
  if (!param.axis.has_value()) {
    if ((ndim == 0 && param.flag != 0) ||  // for scalar
        (ndim == 1 && (param.flag == 1 || param.flag ==2)) ||
        (ndim >= 2 && (param.ord == 0 || param.ord > 2 || param.ord < -2))) {
      LOG(FATAL) << "Invalid norm order for inputs.";
    }
  } else {
    if ((param.axis.value().ndim() == 0 && param.flag != 0) ||  // for scalar
        (param.axis.value().ndim() == 1 && (param.flag == 1 || param.flag ==2)) ||
        (param.axis.value().ndim() == 2 && (param.ord == 0 || param.ord > 2 || param.ord < -2))) {
      LOG(FATAL) << "Invalid norm order for inputs.";
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     ReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims, false));
  return true;
}

NNVM_REGISTER_OP(_npi_norm)
.describe(R"code(Computes the norm on an NDArray.

This operator computes the norm on an NDArray with the specified axis, depending
on the value of the ord parameter. By default, it computes the L2 norm on the entire
array. Currently only ord=2 supports sparse ndarrays.

Examples::

x = [[[1, 2],
    [3, 4]],
   [[2, 2],
    [5, 6]]]

norm(x, ord=2, axis=1) = [[3.1622777 4.472136 ]
                        [5.3851647 6.3245554]]

norm(x, ord=1, axis=1) = [[4., 6.],
                        [7., 8.]]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyLaNormParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyLaNormShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_numpylanorm"})
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
     return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", NumpyLaNormCompute<cpu>)
.add_argument("data", "NDArray-or-Symbol", "The input");

NNVM_REGISTER_OP(_backward_numpylanorm)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyLaNormParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
     return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", NumpyLpNormGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
