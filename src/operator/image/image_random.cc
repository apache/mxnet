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
.add_alias("_npx__image_to_tensor")
.describe(R"code(Converts an image NDArray of shape (H x W x C) or (N x H x W x C) 
with values in the range [0, 255] to a tensor NDArray of shape (C x H x W) or (N x C x H x W)
with values in the range [0, 1]

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

        image = mx.nd.random.uniform(0, 255, (2, 4, 2, 3)).astype(dtype=np.uint8)
        to_tensor(image)
            [[[[0.11764706 0.5803922 ]
               [0.9411765  0.10588235]
               [0.2627451  0.73333335]
               [0.5647059  0.32156864]]
              [[0.7176471  0.14117648]
               [0.75686276 0.4117647 ]
               [0.18431373 0.45490196]
               [0.13333334 0.6156863 ]]
              [[0.6392157  0.5372549 ]
               [0.52156866 0.47058824]
               [0.77254903 0.21568628]
               [0.01568628 0.14901961]]]
             [[[0.6117647  0.38431373]
               [0.6784314  0.6117647 ]
               [0.69411767 0.96862745]
               [0.67058825 0.35686275]]
              [[0.21960784 0.9411765 ]
               [0.44705883 0.43529412]
               [0.09803922 0.6666667 ]
               [0.16862746 0.1254902 ]]
              [[0.6156863  0.9019608 ]
               [0.35686275 0.9019608 ]
               [0.05882353 0.6509804 ]
               [0.20784314 0.7490196 ]]]]
            <NDArray 2x3x4x2 @cpu(0)>
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ToTensorShape)
.set_attr<nnvm::FInferType>("FInferType", ToTensorType)
.set_attr<FCompute>("FCompute<cpu>", ToTensorOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{ "_copy" })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray");

NNVM_REGISTER_OP(_image_normalize)
.add_alias("_npx__image_normalize")
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

        image = mx.nd.random.uniform(0, 1, (2, 3, 4, 2))
        normalize(image, mean=(0, 1, 2), std=(3, 2, 1))
            [[[[ 0.18934818  0.13092826]
               [ 0.3085322   0.27869293]
               [ 0.02367868  0.11246539]
               [ 0.0290431   0.2160573 ]]
              [[-0.4898908  -0.31587923]
               [-0.08369008 -0.02142242]
               [-0.11092162 -0.42982462]
               [-0.06499392 -0.06495637]]
              [[-1.0213816  -1.526392  ]
               [-1.2008414  -1.1990893 ]
               [-1.5385206  -1.4795225 ]
               [-1.2194707  -1.3211205 ]]]
             [[[ 0.03942481  0.24021089]
               [ 0.21330701  0.1940066 ]
               [ 0.04778443  0.17912441]
               [ 0.31488964  0.25287187]]
              [[-0.23907584 -0.4470462 ]
               [-0.29266903 -0.2631998 ]
               [-0.3677222  -0.40683383]
               [-0.11288315 -0.13154092]]
              [[-1.5438497  -1.7834496 ]
               [-1.431566   -1.8647819 ]
               [-1.9812102  -1.675859  ]
               [-1.3823645  -1.8503251 ]]]]
            <NDArray 2x3x4x2 @cpu(0)>
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NormalizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NormalizeOpShape)
.set_attr<nnvm::FInferType>("FInferType", NormalizeOpType)
.set_attr<FCompute>("FCompute<cpu>", NormalizeOpForward<cpu>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_image_normalize"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NormalizeParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_image_normalize)
.set_attr_parser(ParamParser<NormalizeParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NormalizeOpBackward<cpu>);

MXNET_REGISTER_IMAGE_AUG_OP(_image_flip_left_right)
.add_alias("_npx__image_flip_left_right")
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", FlipLeftRight);

MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_flip_left_right)
.add_alias("_npx__image_random_flip_left_right")
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", RandomFlipLeftRight);

MXNET_REGISTER_IMAGE_AUG_OP(_image_flip_top_bottom)
.add_alias("_npx__image_flip_top_bottom")
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", FlipTopBottom);

MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_flip_top_bottom)
.add_alias("_npx__image_random_flip_top_bottom")
.describe(R"code()code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", RandomFlipTopBottom);

MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_brightness)
.add_alias("_npx__image_random_brightness")
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomEnhanceParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomBrightness)
.add_arguments(RandomEnhanceParam::__FIELDS__());

MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_contrast)
.add_alias("_npx__image_random_contrast")
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomEnhanceParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomContrast)
.add_arguments(RandomEnhanceParam::__FIELDS__());


MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_saturation)
.add_alias("_npx__image_random_saturation")
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomEnhanceParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomSaturation)
.add_arguments(RandomEnhanceParam::__FIELDS__());


MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_hue)
.add_alias("_npx__image_random_hue")
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomEnhanceParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomHue)
.add_arguments(RandomEnhanceParam::__FIELDS__());


MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_color_jitter)
.add_alias("_npx__image_random_color_jitter")
.describe(R"code()code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomColorJitterParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomColorJitter)
.add_arguments(RandomColorJitterParam::__FIELDS__());


MXNET_REGISTER_IMAGE_AUG_OP(_image_adjust_lighting)
.add_alias("_npx__image_adjust_lighting")
.describe(R"code(Adjust the lighting level of the input. Follow the AlexNet style.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<AdjustLightingParam>)
.set_attr<FCompute>("FCompute<cpu>", AdjustLighting)
.add_arguments(AdjustLightingParam::__FIELDS__());


MXNET_REGISTER_IMAGE_RND_AUG_OP(_image_random_lighting)
.add_alias("_npx__image_random_lighting")
.describe(R"code(Randomly add PCA noise. Follow the AlexNet style.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<RandomLightingParam>)
.set_attr<FCompute>("FCompute<cpu>", RandomLighting)
.add_arguments(RandomLightingParam::__FIELDS__());

}  // namespace image
}  // namespace op
}  // namespace mxnet
