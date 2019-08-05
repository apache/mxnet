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
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_numpyLanorm"})
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<FCompute>("FCompute<cpu>", NumpyLaNormCompute<cpu>)
.add_argument("data", "NDArray-or-Symbol", "The input");

NNVM_REGISTER_OP(_backward_numpyLanorm)
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
