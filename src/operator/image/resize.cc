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
* \file image_random.cc
* \brief
* \author
*/

#include <mxnet/base.h>
#include "./resize-inl.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
namespace image {

DMLC_REGISTER_PARAMETER(ResizeParam);

NNVM_REGISTER_OP(_image_resize)
.describe("Resize an image or a batch of image NDArray to the given size."
        "Input tensor should be with (H, W, C) or (N, H, W, C)")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ResizeParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ResizeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", Resize<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(ResizeParam::__FIELDS__());

}  // namespace image
}  // namespace op
}  // namespace mxnet
