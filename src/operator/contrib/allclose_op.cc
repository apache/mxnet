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
 * \file allclose_op.cc
 * \brief CPU Implementation of allclose op
 * \author Andrei Ivanov
 */
#include "./allclose_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(AllCloseParam);

NNVM_REGISTER_OP(_contrib_allclose)
.describe(R"code(This operators implements the numpy.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)

.. math::

    f(x) = |a−b|≤atol+rtol|b|

where
:math:`a, b` are the input tensors of equal types an shapes
:math:`atol, rtol` the values of absolute and relative tolerance (by default, rtol=1e-05, atol=1e-08)

Examples::

  a = [1e10, 1e-7],
  b = [1.00001e10, 1e-8]
  y = allclose(a, b)
  y = False

  a = [1e10, 1e-8],
  b = [1.00001e10, 1e-9]
  y = allclose(a, b)
  y = True

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<AllCloseParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", AllCloseShape)
.set_attr<nnvm::FInferType>("FInferType", AllCloseType)
.set_attr<FCompute>("FCompute<cpu>", AllClose<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("a", "NDArray-or-Symbol", "Input array a")
.add_argument("b", "NDArray-or-Symbol", "Input array b")
.add_arguments(AllCloseParam::__FIELDS__());

template<>
size_t GetAdditionalMemoryLogical<cpu>(mshadow::Stream<cpu> *s, const int num_items) {
  return 0;
}

template<>
void GetResultLogical<cpu>(mshadow::Stream<cpu> *s, INTERM_DATA_TYPE *workMem,
                           size_t extraStorageBytes, int num_items, INTERM_DATA_TYPE *outPntr) {
  while (num_items-- > 0 && workMem[num_items]) {}
  outPntr[0] = num_items >= 0? 0 : 1;
}

}  // namespace op
}  // namespace mxnet
