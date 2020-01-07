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
 *  Copyright (c) 2016 by Contributors
 * \file center-crop.h
 * \brief the image center crop operator registration
 */
#include "mxnet/base.h"
#include "center_crop-inl.h"

namespace mxnet {
namespace op {
namespace image {

DMLC_REGISTER_PARAMETER(CenterCropParam);

NNVM_REGISTER_OP(_image_center_crop)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CenterCropParam>)
.set_attr<mxnet::FInferShape>("FInferShape", CenterCropShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", CenterCropOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_backward_image_center_crop" })
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(CenterCropParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_image_center_crop)
.set_attr_parser(ParamParser<CenterCropParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", CenterCropOpBackward<cpu>);

}  // namespace image
}  // namespace op
}  // namespace mxnet
