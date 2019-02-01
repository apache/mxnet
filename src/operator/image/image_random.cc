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
#include "./image_random-inl.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
namespace image {

DMLC_REGISTER_PARAMETER(NormalizeParam);
DMLC_REGISTER_PARAMETER(RandomEnhanceParam);
DMLC_REGISTER_PARAMETER(AdjustLightingParam);
DMLC_REGISTER_PARAMETER(RandomLightingParam);
DMLC_REGISTER_PARAMETER(RandomColorJitterParam);

NNVM_REGISTER_OP(_image_to_tensor)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ToTensorShape)
.set_attr<nnvm::FInferType>("FInferType", ToTensorType)
.set_attr<FCompute>("FCompute<cpu>", ToTensor)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })
.add_argument("data", "NDArray-or-Symbol", "The input.");

NNVM_REGISTER_OP(_image_normalize)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NormalizeParam>)
.set_attr<nnvm::FInferShape>("FInferShape", NormalizeShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", Normalize)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(NormalizeParam::__FIELDS__());

MXNET_REGISTER_IMAGE_AUG_OP(_image_flip_left_right)
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", FlipLeftRight);

MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_flip_left_right)
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", RandomFlipLeftRight);

MXNET_REGISTER_IMAGE_AUG_OP(_image_flip_top_bottom)
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", FlipTopBottom);

MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_flip_top_bottom)
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", RandomFlipTopBottom);

MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_brightness)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomEnhanceParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomBrightness)
.add_arguments(RandomEnhanceParam::__FIELDS__());

MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_contrast)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomEnhanceParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomContrast)
.add_arguments(RandomEnhanceParam::__FIELDS__());


MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_saturation)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomEnhanceParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomSaturation)
.add_arguments(RandomEnhanceParam::__FIELDS__());


MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_hue)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomEnhanceParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomHue)
.add_arguments(RandomEnhanceParam::__FIELDS__());


MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_color_jitter)
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomColorJitterParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomColorJitter)
.add_arguments(RandomColorJitterParam::__FIELDS__());


MXNET_REGISTER_IMAGE_AUG_OP(_image_adjust_lighting)
.describe(R"code(Adjust the lighting level of the input. Follow the AlexNet style.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<AdjustLightingParam>)
.set_attr<FCompute>("FCompute<cpu>", AdjustLighting)
.add_arguments(AdjustLightingParam::__FIELDS__());


MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_lighting)
.describe(R"code(Randomly add PCA noise. Follow the AlexNet style.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomLightingParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomLighting)
.add_arguments(RandomLightingParam::__FIELDS__());

}  // namespace image
}  // namespace op
}  // namespace mxnet
