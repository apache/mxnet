# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from coremltools.proto import NeuralNetwork_pb2 as _NeuralNetwork_pb2


def add_pooling_with_padding_types(builder, name, height, width, stride_height, stride_width,
        layer_type, padding_type, input_name, output_name,
        padding_top = 0, padding_bottom = 0, padding_left = 0, padding_right = 0,
        same_padding_asymmetry_mode = 'BOTTOM_RIGHT_HEAVY',
        exclude_pad_area = True, is_global = False):
    """
    Add a pooling layer to the model.

    This is our own implementation of add_pooling since current CoreML's version (0.5.0) of builder
    doesn't provide support for padding types apart from valid. This support will be added in the
    next release of coremltools. When that happens, this can be removed.

    Parameters

    ----------
    builder: NeuralNetworkBuilder
        A neural network builder object.
    name: str
        The name of this layer.
    height: int
        Height of pooling region.
    width: int
        Number of elements to be padded on the right side of the input blob.
    stride_height: int
        Stride along the height direction.
    stride_width: int
        Stride along the height direction.
    layer_type: str
        Type of pooling performed. Can either be 'MAX', 'AVERAGE' or 'L2'.
    padding_type: str
        Option for the output blob shape. Can be either 'VALID' , 'SAME' or 'INCLUDE_LAST_PIXEL'. Kindly look at NeuralNetwork.proto for details.
    input_name: str
        The input blob name of this layer.
    output_name: str
        The output blob name of this layer.

    padding_top, padding_bottom, padding_left, padding_right: int
        values of height (top, bottom) and width (left, right) padding to be used if padding type is "VALID" or "INCLUDE_LAST_PIXEL"

    same_padding_asymmetry_mode : str.
        Type of asymmetric padding to be used when  padding_type = 'SAME'. Kindly look at NeuralNetwork.proto for details. Can be either 'BOTTOM_RIGHT_HEAVY' or  'TOP_LEFT_HEAVY'.

    exclude_pad_area: boolean
        Whether to exclude padded area in the pooling operation. Defaults to True.

        - If True, the value of the padded area will be excluded.
        - If False, the padded area will be included.
        This flag is only used with average pooling.
    is_global: boolean
        Whether the pooling operation is global. Defaults to False.

        - If True, the pooling operation is global -- the pooling region is of the same size of the input blob.
        Parameters height, width, stride_height, stride_width will be ignored.

        - If False, the pooling operation is not global.

    See Also
    --------
    add_convolution, add_pooling, add_activation
    """

    spec = builder.spec
    nn_spec = builder.nn_spec

    # Add a new layer
    spec_layer = nn_spec.layers.add()
    spec_layer.name = name
    spec_layer.input.append(input_name)
    spec_layer.output.append(output_name)
    spec_layer_params = spec_layer.pooling

    # Set the parameters
    spec_layer_params.type = \
                _NeuralNetwork_pb2.PoolingLayerParams.PoolingType.Value(layer_type)

    if padding_type == 'VALID':
        height_border = spec_layer_params.valid.paddingAmounts.borderAmounts.add()
        height_border.startEdgeSize = padding_top
        height_border.endEdgeSize = padding_bottom
        width_border = spec_layer_params.valid.paddingAmounts.borderAmounts.add()
        width_border.startEdgeSize = padding_left
        width_border.endEdgeSize = padding_right
    elif padding_type == 'SAME':
        if not (same_padding_asymmetry_mode == 'BOTTOM_RIGHT_HEAVY' or  same_padding_asymmetry_mode == 'TOP_LEFT_HEAVY'):
            raise ValueError("Invalid value %d of same_padding_asymmetry_mode parameter" % same_padding_asymmetry_mode)
        spec_layer_params.same.asymmetryMode = _NeuralNetwork_pb2.SamePadding.SamePaddingMode.Value(same_padding_asymmetry_mode)
    elif padding_type == 'INCLUDE_LAST_PIXEL':
        if padding_top != padding_bottom or padding_left != padding_right:
            raise ValueError("Only symmetric padding is supported with the INCLUDE_LAST_PIXEL padding type")
        spec_layer_params.includeLastPixel.paddingAmounts.append(padding_top)
        spec_layer_params.includeLastPixel.paddingAmounts.append(padding_left)

    spec_layer_params.kernelSize.append(height)
    spec_layer_params.kernelSize.append(width)
    spec_layer_params.stride.append(stride_height)
    spec_layer_params.stride.append(stride_width)
    spec_layer_params.avgPoolExcludePadding = exclude_pad_area
    spec_layer_params.globalPooling = is_global
