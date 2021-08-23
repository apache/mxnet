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

#include "crop-inl.h"

namespace mxnet {
namespace op {
namespace image {

NNVM_REGISTER_OP(_image_crop)
.set_attr<FCompute>("FCompute<gpu>", CropOpForward<gpu>);

NNVM_REGISTER_OP(_backward_image_crop)
.set_attr<FCompute>("FCompute<gpu>", CropOpBackward<gpu>);

NNVM_REGISTER_OP(_image_random_crop)
.set_attr<FCompute>("FCompute<gpu>", RandomCropOpForward<gpu>);

NNVM_REGISTER_OP(_backward_image_random_crop)
.set_attr<FCompute>("FCompute<gpu>", RandomCropOpBackward<gpu>);

NNVM_REGISTER_OP(_image_random_resized_crop)
.set_attr<FCompute>("FCompute<gpu>", RandomResizedCropOpForward<gpu>);

NNVM_REGISTER_OP(_backward_image_random_resized_crop)
.set_attr<FCompute>("FCompute<gpu>", RandomResizedCropOpBackward<gpu>);
}  // namespace image
}  // namespace op
}  // namespace mxnet
