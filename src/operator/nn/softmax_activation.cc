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
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief softmax_activation op
 * \author Junyuan Xie, Da Zheng
*/
#include "./softmax_activation-inl.h"
#include "../tensor/elemwise_unary_op.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SoftmaxActivationParam);

MXNET_OPERATOR_REGISTER_UNARY(SoftmaxActivation)
.describe(R"code(Applies softmax activation to input. This is intended for internal layers.

.. note::

  This operator has been deprecated, please use `softmax`.

If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.
This is the default mode.

If `mode` = ``channel``, this operator will compute a k-class softmax at each position
of each instance, where `k` = ``num_channel``. This mode can only be used when the input array
has at least 3 dimensions.
This can be used for `fully convolutional network`, `image segmentation`, etc.

Example::

  >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
  >>>                            [2., -.4, 7.,   3., 0.2]])
  >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
  >>> print softmax_act.asnumpy()
  [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]
   [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SoftmaxActivationParam>)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<FCompute>("FCompute<cpu>", SoftmaxActivationCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_SoftmaxActivation"})
.add_arguments(SoftmaxActivationParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_SoftmaxActivation)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr_parser(ParamParser<SoftmaxActivationParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxActivationGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
