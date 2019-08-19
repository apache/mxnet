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
 *  Copyright (c) 2019 by Contributors
 * \file crop-cc.h
 * \brief the image crop operator registration
 */

#include "mxnet/base.h"
#include "crop-inl.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
namespace image {

DMLC_REGISTER_PARAMETER(CropParam);

NNVM_REGISTER_OP(_image_crop)
.add_alias("_npx__image_crop")
.describe(R"code(Crop an image NDArray of shape (H x W x C) or (N x H x W x C) 
to the given size.
Example:
    .. code-block:: python
        image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
        mx.nd.image.crop(image, 1, 1, 2, 2)
            [[[144  34   4]
              [ 82 157  38]]

             [[156 111 230]
              [177  25  15]]]
            <NDArray 2x2x3 @cpu(0)>
        image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)
        mx.nd.image.crop(image, 1, 1, 2, 2)            
            [[[[ 35 198  50]
               [242  94 168]]

              [[223 119 129]
               [249  14 154]]]


              [[[137 215 106]
                [ 79 174 133]]

               [[116 142 109]
                [ 35 239  50]]]]
            <NDArray 2x2x2x3 @cpu(0)>
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<CropParam>)
.set_attr<mxnet::FInferShape>("FInferShape", CropShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", CropOpForward)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_backward_image_crop" })
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(CropParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_image_crop)
.set_attr_parser(ParamParser<CropParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", CropOpBackward);

}  // namespace image
}  // namespace op
}  // namespace mxnet
