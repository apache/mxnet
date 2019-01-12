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
 *  Copyright (c) 2018 by Contributors
 * \file random_resize_crop
 * \brief the image random_resize_crop operator registration
 */

#include "mxnet/base.h"
#include "random_resize_crop-inl.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
namespace image {

DMLC_REGISTER_PARAMETER(RandomResizeCropParam);

NNVM_REGISTER_OP(_image_random_resize_crop)
.describe("Crop the input image with random scale and aspect ratio.")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RandomResizeCropParam>)
.set_attr<nnvm::FInferShape>("FInferShape", RandomResizeCropShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", RandomResizeCrop)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {        
    return std::vector<ResourceRequest>{ResourceRequest::kRandom};
})
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(RandomResizeCropParam::__FIELDS__());

}  // namespace image
}  // namespace op
}  // namespace mxnet