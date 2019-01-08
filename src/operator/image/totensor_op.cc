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
 * \file totensor_op.cc
 * \brief CPU Implementation of ToTensor op
 */
#include "./totensor_op-inl.h"

namespace mxnet {
namespace op {
namespace image {

NNVM_REGISTER_OP(_image_to_tensor)
.describe(R"code(Converts an image NDArray of shape (H x W x C) or (N x H x W x C) 
with values in the range [0, 255] to a tensor NDArray of shape (C x H x W) or (N x C x H x W)
with values in the range [0, 1)

Example:
    .. code-block:: python
        image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
        to_tensor(image)
            [[[ 0.85490197  0.72156864]
              [ 0.09019608  0.74117649]
              [ 0.61960787  0.92941177]
              [ 0.96470588  0.1882353 ]]
             [[ 0.6156863   0.73725492]
              [ 0.46666667  0.98039216]
              [ 0.44705883  0.45490196]
              [ 0.01960784  0.8509804 ]]
             [[ 0.39607844  0.03137255]
              [ 0.72156864  0.52941179]
              [ 0.16470589  0.7647059 ]
              [ 0.05490196  0.70588237]]]
             <NDArray 3x4x2 @cpu(0)>
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", ToTensorShape)
.set_attr<nnvm::FInferType>("FInferType", ToTensorType)
.set_attr<FCompute>("FCompute<cpu>", ToTensor)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray");

}  // namespace image
}  // namespace op
}  // namespace mxnet
