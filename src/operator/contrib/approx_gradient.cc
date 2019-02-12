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
 * \file approx_gradient.cc
 * \brief CPU Implementation of calculation of numerical gradient approximation
 * \author Andrei Ivanov
 */
#include "./approx_gradient-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(ApproxGradientParam);

NNVM_REGISTER_OP(_contrib_approx_gradient)
.describe(R"code(This operators implements the calculation of numerical gradient approximation
.. math::
    grad[i] = (a−b).sum() / eps
OR
    for j in range(i):
          grad[j] = (a(i)−b(i)).sum() / eps
where
:math:`a, b` are the input tensors of equal types an shapes
:math:`eps` the values of used for calculation of the numerical approximation of derivative
:math:`grad` approximation of gradient
:math:`i` index of the approximated gradient OR size of the batch which was used

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<ApproxGradientParam>)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b", "grad"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", ApproxGradientShape)
.set_attr<nnvm::FInferType>("FInferType", ApproxGradientType)
.set_attr<FCompute>("FCompute<cpu>", ApproxGradient<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("a", "NDArray-or-Symbol", "Input array a")
.add_argument("b", "NDArray-or-Symbol", "Input array b")
.add_argument("grad", "NDArray", "Output array grad")
.add_arguments(ApproxGradientParam::__FIELDS__());


template<>
size_t GetAdditionalMemorySizeA<cpu>(const int num_items) {
  return 0;
}

template<>
void ApproxGradientAction<cpu>(mshadow::Stream<cpu> *s,
             float *workSpaceMemory,
             size_t extraStorageBytes,
             const TBlob& in0,
             const TBlob& in1,
             const TBlob& gradCoord,
             const std::vector<OpReqType>& req,
             const ApproxGradientParam& param) {
  int num_items = in0.Size();
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(in0.type_flag_, DType, {
      Kernel<vector_increment<kWriteTo>, cpu>::Launch(
        s, num_items, workSpaceMemory, in0.dptr<DType>(), in1.dptr<DType>(), param.eps);
  });

  const auto index = param.index;
  if (!param.batched_mode) {
    float sum = 0.f;
    while (num_items-- > 0)
      sum += workSpaceMemory[num_items];

    MSHADOW_TYPE_SWITCH(in0.type_flag_, DType,
                        *(gradCoord.dptr<DType>() + index) = sum;);
  } else {
    MSHADOW_TYPE_SWITCH(in0.type_flag_, DType, {
        Kernel <assign_gradient<kWriteTo>, cpu>::Launch(
          s, index, gradCoord.dptr<DType>(), workSpaceMemory, index);
    });
  }
}

}  // namespace op
}  // namespace mxnet
