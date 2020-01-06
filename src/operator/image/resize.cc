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
 * Copyright (c) 2019 by Contributors
 * \file resize.cc
 * \brief resize operator cpu
 * \author Jake Lee
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
.add_alias("_npx__image_resize")
.describe(R"code(Resize an image NDArray of shape (H x W x C) or (N x H x W x C) 
to the given size
Example:
    .. code-block:: python
        image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
        mx.nd.image.resize(image, (3, 3))
            [[[124 111 197]
              [158  80 155]
              [193  50 112]]

             [[110 100 113]
              [134 165 148]
              [157 231 182]]

             [[202 176 134]
              [174 191 149]
              [147 207 164]]]
            <NDArray 3x3x3 @cpu(0)>
        image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)
        mx.nd.image.resize(image, (2, 2))            
            [[[[ 59 133  80]
               [187 114 153]]

              [[ 38 142  39]
               [207 131 124]]]


              [[[117 125 136]
               [191 166 150]]

              [[129  63 113]
               [182 109  48]]]]
            <NDArray 2x2x2x3 @cpu(0)>
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ResizeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ResizeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", Resize<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(ResizeParam::__FIELDS__());

}  // namespace image
}  // namespace op
}  // namespace mxnet
