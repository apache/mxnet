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
.set_attr<nnvm::FInferShape>("FInferShape", AllCloseShape)
.set_attr<nnvm::FInferType>("FInferType", AllCloseType)
.set_attr<FCompute>("FCompute<cpu>", AllClose<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("a", "NDArray-or-Symbol", "Input array a")
.add_argument("b", "NDArray-or-Symbol", "Input array b")
.add_arguments(AllCloseParam::__FIELDS__());

template<>
size_t GetAdditionalMemorySize<cpu>(const int num_items) {
  return 0;
}

template<>
void AllCloseAction<cpu>(mshadow::Stream<cpu> *s,
             int *workSpaceMemory,
             size_t extraStorageBytes,
             const TBlob& in0,
             const TBlob& in1,
             const std::vector<OpReqType>& req,
             const AllCloseParam& param,
             int *outPntr) {
  int num_items = in0.Size();
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(in0.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<allclose_forward<req_type>, cpu>::Launch(
        s, num_items, workSpaceMemory, in0.dptr<DType>(), in1.dptr<DType>(),
        param.rtol, param.atol, param.equal_nan);
    });
  });

  while (num_items-- > 0 && workSpaceMemory[num_items] > 0.5) {}
  outPntr[0] = num_items >= 0? 0 : 1;
}

}  // namespace op
}  // namespace mxnet
