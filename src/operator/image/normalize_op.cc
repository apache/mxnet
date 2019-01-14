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
 * \file normalize_op.cc
 * \brief CPU Implementation of Normalize op
 */
#include "./normalize_op-inl.h"

namespace mxnet {
namespace op {
namespace image {

DMLC_REGISTER_PARAMETER(NormalizeParam);

NNVM_REGISTER_OP(_image_normalize)
.describe(R"code(Normalize an tensor of shape (C x H x W) or (N x C x H x W) with mean and
    standard deviation.

    Given mean `(m1, ..., mn)` and std `(s\ :sub:`1`\ , ..., s\ :sub:`n`)` for `n` channels,
    this transform normalizes each channel of the input tensor with:

.. math::

        output[i] = (input[i] - m\ :sub:`i`\ ) / s\ :sub:`i`

    If mean or std is scalar, the same value will be applied to all channels.

    Default value for mean is 0.0 and stand deviation is 1.0.

Example:

    .. code-block:: python
        image = mx.nd.random.uniform(0, 1, (3, 4, 2))
        normalize(image, mean=(0, 1, 2), std=(3, 2, 1))
            [[[ 0.18293785  0.19761486]
              [ 0.23839645  0.28142193]
              [ 0.20092112  0.28598186]
              [ 0.18162774  0.28241724]]
             [[-0.2881726  -0.18821815]
              [-0.17705294 -0.30780914]
              [-0.2812064  -0.3512327 ]
              [-0.05411351 -0.4716435 ]]
             [[-1.0363373  -1.7273437 ]
              [-1.6165586  -1.5223348 ]
              [-1.208275   -1.1878313 ]
              [-1.4711051  -1.5200229 ]]]
            <NDArray 3x4x2 @cpu(0)>
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NormalizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", NormalizeOpShape)
.set_attr<nnvm::FInferType>("FInferType", NormalizeOpType)
.set_attr<FCompute>("FCompute<cpu>", NormalizeOpForward<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NormalizeParam::__FIELDS__());

}  // namespace image
}  // namespace op
}  // namespace mxnet
