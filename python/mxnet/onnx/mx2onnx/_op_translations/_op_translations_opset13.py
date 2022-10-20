#  Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
#  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Based on
#  https://github.com/NVIDIA/mxnet_to_onnx/blob/master/mx2onnx_converter/
# mx2onnx_converter_functions.py

# coding: utf-8
# pylint: disable=too-many-locals,no-else-return,too-many-lines
# pylint: disable=anomalous-backslash-in-string,eval-used
# pylint: disable=too-many-function-args
"""
Conversion Functions for common layers.
Add new functions here with a decorator.
"""

import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
try:
    import onnx
except ImportError:
    onnx = None

OPSET_VERSION = 13

def parse_helper(attrs, attrs_name, alt_value=None):
    """Helper function to parse operator attributes in required format."""
    tuple_re = re.compile(r'\([0-9L|,| ]+\)')
    if not attrs:
        return alt_value
    attrs_str = None if attrs.get(attrs_name) is None else str(attrs.get(attrs_name))
    if attrs_str is None:
        return alt_value
    attrs_match = tuple_re.search(attrs_str)
    if attrs_match is not None:
        if attrs_match.span() == (0, len(attrs_str)):
            dims = eval(attrs_str)
            return dims
        else:
            raise AttributeError(f"Malformed {attrs_name} dimensions: {str(attrs_str)}")
    return alt_value

def transform_padding(pad_width):
    """Helper function to convert padding format for pad operator.
    """
    num_pad_values = len(pad_width)
    onnx_pad_width = [0]*num_pad_values

    start_index = 0
    # num_pad_values will always be multiple of 2
    end_index = int(num_pad_values/2)
    for idx in range(0, num_pad_values):
        if idx % 2 == 0:
            onnx_pad_width[start_index] = pad_width[idx]
            start_index += 1
        else:
            onnx_pad_width[end_index] = pad_width[idx]
            end_index += 1

    return onnx_pad_width


def convert_string_to_list(string_val):
    """Helper function to convert string to list.
     Used to convert shape attribute string to list format.
    """
    result_list = []

    list_string = string_val.split(',')
    for val in list_string:
        val = str(val.strip())
        val = val.replace("(", "")
        val = val.replace(")", "")
        val = val.replace("L", "")
        val = val.replace("[", "")
        val = val.replace("]", "")
        if val == "None":
            result_list.append(None)
        elif val != "":
            result_list.append(int(val))

    return result_list

def get_boolean_attribute_value(attrs, attr_name):
    """ Helper function to convert a string version
    of Boolean attributes to integer for ONNX.
    Takes attribute dictionary and attr_name as
    parameters.
    """
    return 1 if attrs.get(attr_name, 0) in ["True", "1"] else 0

def get_inputs(node, kwargs):
    """Helper function to get inputs"""
    name = node["name"]
    outputs_lookup = kwargs["outputs_lookup"]
    inputs = node["inputs"]
    attrs = node.get("attrs", {})
    input_nodes = []
    for ip in inputs:
        input_node_name = outputs_lookup[ip[0]][ip[1]].name
        input_nodes.append(input_node_name)

    return name, input_nodes, attrs

def get_input_dtypes(node, kwargs):
    outputs_lookup = kwargs['outputs_lookup']
    inputs = node['inputs']
    input_dtypes = []
    for ip in inputs:
        input_node_dtype = outputs_lookup[ip[0]][ip[1]].dtype
        input_dtypes.append(input_node_dtype)
    return input_dtypes

def create_basic_op_node(op_name, node, kwargs):
    """Helper function to create a basic operator
    node that doesn't contain op specific attrs"""
    name, input_nodes, _ = get_inputs(node, kwargs)

    node = onnx.helper.make_node(
        op_name,
        input_nodes,
        [name],
        name=name
    )
    return [node]

def create_const_scalar_node(input_name, value, kwargs):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    from onnx.helper import make_tensor
    initializer = kwargs["initializer"]
    dtype = value.dtype
    if dtype == 'float16':
        # when using float16, we must convert it to np.uint16 view first
        value = np.float16(value).view(np.uint16)
    input_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    tensor_node = make_tensor(input_name, input_type, (), ([value]))
    initializer.append(tensor_node)

def create_const_node(input_name, value, kwargs):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    from onnx.helper import make_tensor
    initializer = kwargs["initializer"]
    dtype = value.dtype
    if dtype == 'float16':
        # when using float16, we must convert it to np.uint16 view first
        value = np.float16(value).view(np.uint16)
    input_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    input_shape = value.shape
    tensor_node = make_tensor(input_name, input_type, input_shape, value)
    initializer.append(tensor_node)

def create_tensor(tensor_list, tensor_name, initializer, dtype='int64'):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    tensor_np = np.array(tensor_list, dtype=dtype)
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[tensor_np.dtype]
    dims = np.shape(tensor_np)
    if dtype == np.float16:
        tensor_np = tensor_np.view(dtype=np.uint16)
    tensor = onnx.helper.make_tensor(
        name=tensor_name,
        data_type=data_type,
        dims=dims,
        vals=tensor_np.flatten().tolist(),
        raw=False
    )
    initializer.append(tensor)


def create_helper_trans_node(node_name, input_node):
    """create extra transpose node for dot operator"""
    trans_node = onnx.helper.make_node(
        'Transpose',
        inputs=[input_node],
        outputs=[node_name],
        name=node_name
    )
    return trans_node


def scalar_op_helper(node, op_name, **kwargs):
    """Helper function for scalar arithmetic operations"""
    from onnx import numpy_helper
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)

    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]

    scalar_value = np.array([attrs.get("scalar", 1)],
                            dtype=dtype)
    initializer = kwargs["initializer"]
    flag = True
    # If the input value is in initializer, just multiply with scalar input
    # and create a new initializer
    for i in initializer:
        if i.name == input_nodes[0]:
            if op_name == 'Mul':
                new_initializer = numpy_helper.to_array(i) * scalar_value[0]
            elif op_name == 'Sub':
                if name.startswith("_rminusscalar"):
                    new_initializer = scalar_value[0] - numpy_helper.to_array(i)
                else:
                    new_initializer = numpy_helper.to_array(i) - scalar_value[0]
            elif op_name == 'Add':
                new_initializer = numpy_helper.to_array(i) + scalar_value[0]
            elif op_name == 'Div':
                if name.startswith("_rdivscalar"):
                    new_initializer = scalar_value[0] / numpy_helper.to_array(i)
                else:
                    new_initializer = numpy_helper.to_array(i) / scalar_value[0]
            elif op_name == 'Pow':
                new_initializer = numpy_helper.to_array(i) ** scalar_value[0]
            flag = False
            break

    # else create a new tensor of the scalar value, add it in initializer
    if flag is True:
        dims = np.shape(scalar_value)

        scalar_op_name = "scalar_op" + str(kwargs["idx"])
        tensor_node = onnx.helper.make_tensor_value_info(scalar_op_name, dtype_t, dims)

        initializer.append(
            onnx.helper.make_tensor(
                name=scalar_op_name,
                data_type=dtype_t,
                dims=dims,
                vals=scalar_value,
                raw=False,
            )
        )

        mul_node = onnx.helper.make_node(
            op_name,
            [input_nodes[0], scalar_op_name],
            [name],
            name=name
        )

        return [tensor_node, mul_node]
    else:
        dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[new_initializer.dtype]
        dims = np.shape(new_initializer)

        tensor_node = onnx.helper.make_tensor_value_info(name, dtype_t, dims)

        initializer.append(
            onnx.helper.make_tensor(
                name=name,
                data_type=dtype_t,
                dims=dims,
                vals=new_initializer.flatten(),
                raw=False,
            )
        )
        return [tensor_node]


    return create_basic_op_node('Shape', node, kwargs)


@mx_op.register("_contrib_arange_like", OPSET_VERSION)
def convert_arange_like(node, **kwargs):
    """Map MXNet's arange_like operator attributes to onnx's Range and Reshape operators.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError("ONNX opset 11 or greater is required to export this operator")
    # use the same dtype as the that of the input node
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    axis = attrs.get('axis', 'None')
    start = attrs.get('start', 0.)
    step = attrs.get('step', 1.)
    repeat = int(attrs.get('repeat', 1))
    if repeat != 1:
        raise NotImplementedError("arange_like operator with repeat != 1 not yet implemented.")

    create_const_scalar_node(name+"_start", np.dtype(dtype).type(start), kwargs)
    create_const_scalar_node(name+"_step", np.dtype(dtype).type(step), kwargs)
    create_const_scalar_node(name+"_half_step", np.dtype(dtype).type(float(step)*0.5), kwargs)
    create_tensor([0], name+"_0", kwargs["initializer"], dtype='int64')
    nodes = []
    if axis == 'None':
        # output will be same shape as input
        nodes += [
            make_node("Shape", [input_nodes[0]], [name+"_shape0_out"]),
            make_node("ReduceProd", [name+"_shape0_out"], [name+"_redprod0_out"]),
            make_node("Squeeze", [name+"_redprod0_out", name+"_0"], [name+'_reshape0_out']),
            make_node("Cast", [name+"_reshape0_out"], [name+"_cast0_out"], to=dtype_t),
            make_node("Mul", [name+"_cast0_out", name+"_step"], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", name+"_start"], [name+"_add1_out"]),
            make_node("Sub", [name+"_add1_out", name+"_half_step"], [name+"_sub0_out"]),
            make_node("Range", [name+"_start", name+"_sub0_out", name+"_step"], [name+"_range0_out"]),
            make_node("Reshape", [name+"_range0_out", name+"_shape0_out"], [name], name=name)
        ]
    else:
        # determine shape of axis
        create_tensor([int(axis)], name+"_axis_start", kwargs["initializer"], dtype='int64')
        create_tensor([int(axis)+1], name+"_axis_end", kwargs["initializer"], dtype='int64')
        nodes += [
            make_node("Shape", [input_nodes[0]], [name+"_shape0_out"]),
            make_node("Slice", [name+"_shape0_out", name+"_axis_start", name+"_axis_end"], [name+"_slice0_out"]),
            make_node("ReduceProd", [name+"_slice0_out"], [name+"_reprod0_out"]),
            make_node("Squeeze", [name+"_reprod0_out", name+"_0"], [name+"_reshape0_out"]),
            make_node("Cast", [name+"_reshape0_out"], [name+"_cast0_out"], to=dtype_t),
            make_node("Mul", [name+"_cast0_out", name+"_step"], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", name+"_start"], [name+"_add1_out"]),
            make_node("Sub", [name+"_add1_out", name+"_half_step"], [name+"_sub0_out"]),
            make_node("Range", [name+"_start", name+"_sub0_out", name+"_step"], [name], name=name)
        ]

    return nodes


@mx_op.register("LayerNorm", OPSET_VERSION)
def convert_layer_norm(node, **kwargs):
    """Map MXNet's LayerNorm operator attributes to onnx operators.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)

    dtype = input_dtypes[0]

    axes = int(attrs.get('axis', -1))
    eps = attrs.get('eps', 9.99999975e-06)

    create_tensor([axes], name+"_axes", kwargs["initializer"])
    create_tensor([axes+1], name+"_axes+1", kwargs["initializer"])
    create_tensor([0], name+"_0", kwargs["initializer"], dtype='int64')
    create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)
    create_const_scalar_node(name+"_2_s", np.int64(2).astype(dtype), kwargs)
    create_const_scalar_node(name+"_eps", np.float32(eps), kwargs)

    nodes = [
        make_node("ReduceMean", [input_nodes[0]], [name+"_rm0_out"], axes=[axes]),
        make_node("Sub", [input_nodes[0], name+"_rm0_out"], [name+"_sub0_out"]),
        make_node("Pow", [name+"_sub0_out", name+"_2_s"], [name+"_pow0_out"]),
        make_node("ReduceMean", [name+"_pow0_out"], [name+"_rm1_out"], axes=[axes]),
        make_node("Add", [name+"_rm1_out", name+"_eps"], [name+"_add0_out"]),
        make_node("Sqrt", [name+"_add0_out"], [name+"_sqrt0_out"]),
        make_node("Div", [name+"_sub0_out", name+"_sqrt0_out"], [name+"_div0_out"]),
    ]

    if axes == -1:
        nodes += [
            make_node("Mul", [name+"_div0_out", input_nodes[1]], [name+"_mul0_out"]),
            # make_node("Add", [name+"_mul0_out", input_nodes[2]], [name])
            # the Add operator triggers a weird NaN issue in onnxruntime
            # a workaround is to use Neg + Sub
            make_node('Neg', [input_nodes[2]], [name+'_neg']),
            make_node("Sub", [name+"_mul0_out", name+'_neg'], [name])
        ]
    else:
        nodes += [
            make_node("Shape", [input_nodes[0]], [name+"_shape0_out"]),
            make_node("Shape", [name+"_shape0_out"], [name+"_in_dim"]),
            make_node("Squeeze", [name+"_in_dim", name+"_0"], [name+"_in_dim_s"]),
            make_node("Range", [name+"_0_s", name+"_in_dim_s", name+"_1_s"], [name+"_range"]),
            make_node("Equal", [name+"_range", name+"_axes"], [name+"_equal"]),
            make_node("Cast", [name+"_equal"], [name+"_one_hot"], to=int(TensorProto.INT64)),
            make_node("Slice", [name+"_shape0_out", name+"_axes", name+"_axes+1"], [name+"_slice_out"]),
            make_node("Squeeze", [name+"_slice_out", name+"_0"], [name+"_slice_out_s"]),
            make_node("Sub", [name+"_slice_out_s", name+"_1_s"], [name+"_sub1_out"]),
            make_node("Mul", [name+"_one_hot", name+"_sub1_out"], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", name+"_1_s"], [name+"_add1_out"]),
            make_node('Reshape', [input_nodes[1], name+"_add1_out"], [name+"gamma_exp"]),
            make_node('Reshape', [input_nodes[2], name+"_add1_out"], [name+"beta_exp"]),
            make_node('Expand', [name+"gamma_exp", name+"_shape0_out"], [name+"gamma_exp1"]),
            make_node('Expand', [name+"beta_exp", name+"_shape0_out"], [name+"beta_exp1"]),
            make_node("Mul", [name+"_div0_out", name+"gamma_exp1"], [name+"_mul1_out"]),
            make_node("Add", [name+"_mul1_out", name+"beta_exp1"], [name], name=name)
        ]

    return nodes


@mx_op.register("broadcast_axis", OPSET_VERSION)
def convert_broadcast_axis(node, **kwargs):
    """Map MXNet's broadcast_axis
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = convert_string_to_list(attrs.get('axis', '()'))
    size = convert_string_to_list(attrs.get('size', '()'))
    assert len(axis) == len(size)

    shape_name = name+'_shape_0'

    create_tensor([0], name+'_0', kwargs["initializer"])
    create_tensor([1], name+'_1', kwargs["initializer"])
    create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)

    nodes = [
        make_node('Shape', [input_nodes[0]], [shape_name]),
        make_node('Shape', [shape_name], [name+'_in_dim']),
        make_node('Squeeze', [name+'_in_dim', name+'_0'], [name+'_in_dim_s']),
        make_node('Range', [name+'_0_s', name+'_in_dim_s', name+'_1_s'], [name+'_range']),
    ]

    for i, axis in enumerate(axis):
        if axis not in (0, 1):
            create_tensor([axis], name+'_'+str(axis), kwargs["initializer"])
        create_tensor([size[i]-1], name+'_size_'+str(i), kwargs["initializer"])
        nodes += [
            make_node('Equal', [name+'_range', name+'_'+str(axis)], [name+'_equal_'+str(i)]),
            make_node('Cast', [name+'_equal_'+str(i)], [name+'_cast_'+str(i)], to=int(TensorProto.INT64)),
            make_node('Mul', [name+'_size_'+str(i), name+'_cast_'+str(i)], [name+'_mul_'+str(i)]),
            make_node('Add', [name+'_mul_'+str(i), name+'_1'], [name+'_add_'+str(i)]),
            make_node('Mul', [name+'_add_'+str(i), shape_name], [name+'_shape_'+str(i+1)])
        ]
        shape_name = name+'_shape_'+str(i+1)

    nodes += [
        make_node('Expand', [input_nodes[0], shape_name], [name], name=name)
    ]

    return nodes


@mx_op.register("SequenceMask", OPSET_VERSION)
def convert_sequencemask(node, **kwargs):
    """Map MXNet's SequenceMask operator
    """
    from onnx.helper import make_node
    from onnx import TensorProto

    name, input_nodes, attrs = get_inputs(node, kwargs)

    use_sequence_length = attrs.get('use_sequence_length', 'False')
    mask_val = float(attrs.get('value', '0'))
    axis = int(attrs.get('axis', '0'))

    if(use_sequence_length == 'False'):
        return [make_node('Identity', [input_nodes[0]], [name], name=name)]

    create_tensor([0], name+'_0', kwargs["initializer"])
    create_tensor([1], name+'_1', kwargs["initializer"])
    create_tensor([2], name+'_2', kwargs["initializer"])
    create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
    create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)
    create_const_scalar_node(name+'_2_s', np.int64(2), kwargs)
    create_tensor([mask_val], name+'_mask_val', kwargs["initializer"], dtype='float32')

    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_in_shape']),
        make_node('Slice', [name+'_in_shape', name+'_0', name+'_1'], [name+'_slice_0']),
        make_node('Slice', [name+'_in_shape', name+'_1', name+'_2'], [name+'_slice_1']),
        make_node('Concat', [name+'_slice_0', name+'_1'], [name+'_shape_0'], axis=0),
        make_node('Shape', [name+'_in_shape'], [name+'_in_dim']),
        make_node('Squeeze', [name+'_in_dim', name+'_0'], [name+'_in_dim_s']),
        make_node('Range', [name+'_0_s', name+'_in_dim_s', name+'_1_s'], [name+'_range_0']),
        make_node('Less', [name+'_range_0', name+'_2'], [name+'_less_0']),
        make_node('Where', [name+'_less_0', name+'_in_shape', name+'_1'], [name+'_shape_1'])
    ]

    if(axis == 0):
        nodes += [
            make_node('Squeeze', [name+'_slice_0', name+'_0'], [name+'_max_len']),
            make_node('Range', [name+'_0_s', name+'_max_len', name+'_1_s'], [name+'_range_1']),
            make_node('Reshape', [name+'_range_1', name+'_shape_0'], [name+"_reshape_0"]),
            make_node('Cast', [input_nodes[1]], [name+'_cast'], to=int(TensorProto.INT64)),
            make_node('Less', [name+'_reshape_0', name+'_cast'], [name+'_less_1']),
            make_node('Reshape', [name+'_less_1', name+'_shape_1'], [name+"_reshape_1"]),
            make_node('Where', [name+'_reshape_1', input_nodes[0], name+'_mask_val'], [name], name=name),
        ]
    else:
        nodes += [
            make_node('Squeeze', [name+'_slice_1', name+'_0'], [name+'_max_len']),
            make_node('Range', [name+'_0_s', name+'_max_len', name+'_1_s'], [name+'_range_1']),
            make_node('Reshape', [input_nodes[1], name+'_shape_0'], [name+"_reshape_0"]),
            make_node('Cast', [name+"_reshape_0"], [name+'_cast'], to=int(TensorProto.INT64)),
            make_node('Less', [name+'_range_1', name+'_cast'], [name+'_less_1']),
            make_node('Reshape', [name+'_less_1', name+'_shape_1'], [name+"_reshape_1"]),
            make_node('Where', [name+'_reshape_1', input_nodes[0], name+'_mask_val'], [name], name=name),
        ]
    return nodes


@mx_op.register("expand_dims", OPSET_VERSION)
def convert_expand_dims(node, **kwargs):
    """Map MXNet's expand_dims operator attributes to onnx's Unsqueeze operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get("axis"))
    create_tensor([axis], name+"_axis", kwargs["initializer"])
    input_nodes.append(name+"_axis")
    node = onnx.helper.make_node(
        "Unsqueeze",
        input_nodes,
        [name],
        name=name,
    )
    return [node]


@mx_op.register("stack", OPSET_VERSION)
def convert_stack(node, **kwargs):
    """Map MXNet's stack operator to onnx operators.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get("axis", 0))
    create_tensor([axis], name+"_axis", kwargs["initializer"])
    idx = 0
    nodes = []
    for input_node in input_nodes:
        nodes.append(onnx.helper.make_node(
            "Unsqueeze",
            inputs=[input_node, name+"_axis"],
            outputs=[name+"_unsqueeze"+str(idx)]
        ))
        idx += 1

    nodes.append(onnx.helper.make_node(
        "Concat",
        inputs=[name+"_unsqueeze"+str(i) for i in range(len(nodes))],
        outputs=[name],
        name=name,
        axis=axis
    ))
    return nodes


@mx_op.register("softmax", OPSET_VERSION)
def convert_softmax(node, **kwargs):
    """Map MXNet's softmax operator attributes to onnx's Softmax operator
    and return the created node.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)

    axis = int(attrs.get("axis", -1))
    temperature = str(attrs.get("temperature", 'None'))
    if temperature == 'None':
        temperature = 1.
    else:
        temperature = float(temperature)

    use_length = str(attrs.get("use_length", 'None'))
    use_length = use_length in ['1', 'True']
    dtype = input_dtypes[0]
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    data = input_nodes[0]

    create_tensor([0], name+"_0", kwargs["initializer"])
    if axis == -1 and temperature == 1.:
        nodes = []
        if use_length:
            # magic number, this is fp16 min
            create_tensor([-65500.0], name+"_mask_val", kwargs["initializer"], dtype=dtype)
            create_tensor([1], name+"_1", kwargs["initializer"])
            create_tensor([-1], name+"_-1", kwargs["initializer"])
            create_const_scalar_node(name+"_0_s", np.int64(0), kwargs)
            create_const_scalar_node(name+"_1_s", np.int64(1), kwargs)
            nodes += [
                make_node("Shape", [data], [name+"_shape"]),
                make_node("Shape", [name+"_shape"], [name+"_dim"]),
                make_node("Sub", [name+"_dim", name+"_1"], [name+"_dim_m1"]),
                make_node("Slice", [name+"_shape", name+"_dim_m1", name+"_dim"],
                          [name+"_dim_last_"]),
                make_node("Squeeze", [name+"_dim_last_", name+"_0"], [name+"_dim_last"]),
                make_node("Range", [name+"_0_s", name+"_dim_last", name+"_1_s"], [name+"_range"]),
                make_node("Cast", [input_nodes[1]], [name+"_len"], to=int(TensorProto.INT64)),
                make_node("Unsqueeze", [name+"_len", name+"_-1"], [name+"_len_unsqueezed"]),
                make_node("Less", [name+"_range", name+"_len_unsqueezed"], [name+"_less"]),
                make_node("Where", [name+'_less', data, name+"_mask_val"], [name+"_data_masked"])
            ]
            data = name+"_data_masked"

        nodes += [
            make_node("Softmax", [data], [name], axis=-1)
        ]

        return nodes

    create_tensor([axis], name+"_axes", kwargs["initializer"])
    create_tensor([temperature], name+"_tmp", kwargs["initializer"], dtype=dtype)
    nodes = [
        make_node("Div", [data, name+"_tmp"], [name+'_data']),
    ]
    if len(input_nodes) == 1:
        nodes += [
            make_node("Softmax", [name+'_data'], [name], axis=axis)
        ]
        return nodes
    elif use_length:
        length = input_nodes[1]
        create_tensor([1], name+"_1", kwargs["initializer"])
        create_const_scalar_node(name+'_-1_s', np.int64(-1), kwargs)
        create_const_scalar_node(name+'_0_s', np.int64(0), kwargs)
        create_const_scalar_node(name+'_1_s', np.int64(1), kwargs)
        nodes += [
            # cast data type
            make_node("Cast", [length], [name+"_length"], to=int(TensorProto.INT64)),
            make_node("Cast", [name+"_0"], [name+"_0_itype"], to=dtype_t),
            make_node("Cast", [name+"_1"], [name+"_1_itype"], to=dtype_t),
            # softmax output
            make_node("Softmax", [name+'_data'], [name+"_softmax_out"], axis=axis),
            # update axis
            make_node("Shape", [data], [name+"_shape0_out"]),
            make_node("Shape", [name+"_shape0_out"], [name+"_in_dim"]),
            make_node("Add", [name+"_in_dim", name+"_axes"], [name+"_dim+axis"]),
            make_node("Less", [name+"_axes", name+"_0_s"], [name+"_less0_out"]),
            make_node("Where", [name+"_less0_out", name+"_dim+axis", name+"_axes"], [name+"_final_axis"]),
            # data mask
            make_node("Add", [name+"_final_axis", name+"_1_s"], [name+"_final_axis+1"]),
            make_node("Slice", [name+"_shape0_out", name+"_final_axis", name+"_final_axis+1"], [name+"_axis_dim"]),
            make_node("Squeeze", [name+"_axis_dim", name+"_0"], [name+"_axis_dim_s"]),
            make_node("Range", [name+"_0_s", name+"_axis_dim_s", name+"_1_s"], [name+"_range0_out"]),
            # one hot for axis
            make_node("Squeeze", [name+"_in_dim", name+"_0"], [name+"_in_dim_s"]),
            make_node("Range", [name+"_0_s", name+"_in_dim_s", name+"_1_s"], [name+"_range1_out"]),
            make_node("Equal", [name+"_range1_out", name+"_final_axis"], [name+"_equal_out"]),
            make_node("Cast", [name+"_equal_out"], [name+"_one_hot"], to=int(TensorProto.INT64)),
            # reshape data mask for less
            make_node("Sub", [name+"_axis_dim_s", name+"_1_s"], [name+"_sub0_out"]),
            make_node("Mul", [name+"_one_hot", name+"_sub0_out"], [name+"_mul0_out"]),
            make_node("Add", [name+"_mul0_out", name+"_1_s"], [name+"_add0_out"]),
            make_node('Reshape', [name+"_range0_out", name+"_add0_out"], [name+"_reshape0_out"]),
            # reshape length for less
            make_node("Mul", [name+"_one_hot", name+"_-1_s"], [name+"_mul1_out"]),
            make_node("Add", [name+"_mul1_out", name+"_1_s"], [name+"_add1_out"]),
            make_node("Sub", [name+"_shape0_out", name+"_1_s"], [name+"_sub1_out"]),
            make_node("Mul", [name+"_add1_out", name+"_sub1_out"], [name+"_mul2_out"]),
            make_node("Add", [name+"_mul2_out", name+"_1_s"], [name+"_add2_out"]),
            make_node('Reshape', [name+"_length", name+"_add2_out"], [name+"_reshape1_out"]),
            # mask output
            make_node("Less", [name+"_reshape0_out", name+"_reshape1_out"], [name+"_less_out"]),
            make_node("Cast", [name+"_less_out"], [name+"_mask"], to=dtype_t),
            make_node("Mul", [name+"_softmax_out", name+"_mask"], [name+"_mul3_out"]),
            make_node("ReduceSum", [name+"_mul3_out", name+"_axes"], [name+"_rsum1_out"], keepdims=1),
            make_node("Equal", [name+"_rsum1_out", name+"_0_itype"], [name+"_equal1_out"]),
            make_node("Where", [name+"_equal1_out", name+"_1_itype", name+"_rsum1_out"], [name+"_where_out"]),
            make_node("Div", [name+"_mul3_out", name+"_where_out"], [name], name=name)
        ]
        return nodes

    else:
        raise NotImplementedError("use_length must be true when both data and length are paased in.")


@mx_op.register("reverse", OPSET_VERSION)
def convert_reverse(node, **kwargs):
    """Map MXNet's reverse operator attributes to ONNX
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    axis = int(attrs.get('axis', 0))

    # Transpose takes perm as a parameter, so we must 'pad' the input to a known dim (8 here)
    perm = [i for i in range(8)]
    perm[0], perm[axis] = axis, 0

    create_tensor([8], name+'_8', kwargs['initializer'])
    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([-1], name+'_m1', kwargs['initializer'])
    create_tensor([axis], name+'_axis', kwargs['initializer'])
    create_tensor([axis+1], name+'_axis_p1', kwargs['initializer'])
    create_const_scalar_node(name+'_m1_s', np.int64(-1), kwargs)

    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Shape', [name+'_shape'], [name+'_dim']),
        make_node('Sub', [name+'_8', name+'_dim'], [name+'_sub']),
        make_node('Concat', [name+'_0', name+'_sub'], [name+'_concat'], axis=0),
        make_node('Pad', [name+'_shape', name+'_concat', name+'_1'], [name+'_shape_8_dim']),
        make_node('Reshape', [input_nodes[0], name+'_shape_8_dim'], [name+'_data_8_dim']),
        make_node('Transpose', [name+'_data_8_dim'], [name+'_data_t'], perm=perm),
        make_node('Slice', [name+'_shape', name+'_axis', name+'_axis_p1'], [name+'_axis_len']),
        make_node('Sub', [name+'_axis_len', name+'_1'], [name+'_axis_len_m1']),
        make_node('Squeeze', [name+'_axis_len_m1', name+'_0'], [name+'_axis_len_m1_s']),
        make_node('Range', [name+'_axis_len_m1_s', name+'_m1_s', name+'_m1_s'], [name+'_indices']),
        make_node('Gather', [name+'_data_t', name+'_indices'], [name+'_gather']),
        make_node('Transpose', [name+'_gather'], [name+'_data_reversed'], perm=perm),
        make_node('Reshape', [name+'_data_reversed', name+'_shape'], [name], name=name)
    ]

    return nodes


@mx_op.register('repeat', OPSET_VERSION)
def convert_repeat(node, **kwargs):
    """Map MXNet's repeat operator attributes to onnx's Tile operator.
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')

    repeats = int(attrs.get('repeats', 1))
    axis = attrs.get('axis', 'None')

    if repeats <= 0:
        raise NotImplementedError('repeat operator does not support parameter repeats==0')

    nodes = []
    if axis == 'None':
        create_tensor([-1], name+'_-1', kwargs['initializer'])
        create_tensor([repeats], name+'_rep', kwargs['initializer'])
        create_tensor([1, repeats], name+'_repeats', kwargs['initializer'])
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('ReduceProd', [name+'_shape'], [name+'_size']),
            make_node('Reshape', [input_nodes[0], name+'_size'], [name+'_flat']),
            make_node('Unsqueeze', [name+'_flat', name+'_-1'], [name+'_unsqueeze']),
            make_node('Tile', [name+'_unsqueeze', name+'_repeats'], [name+'_tile']),
            make_node('Mul', [name+'_size', name+'_rep'], [name+'_new_size']),
            make_node('Reshape', [name+'_tile', name+'_new_size'], [name], name=name)
        ]
    else:
        axis = int(axis)
        repeats -= 1
        create_tensor([repeats], name+'_repeats', kwargs['initializer'])
        create_tensor([1], name+'_1', kwargs['initializer'])
        create_tensor([0], name+'_0', kwargs['initializer'])
        create_tensor([axis], name+'_axis', kwargs['initializer'])
        create_const_scalar_node(name+"_0_s", np.int64(0), kwargs)
        create_const_scalar_node(name+"_1_s", np.int64(1), kwargs)
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_shape']),
            make_node('Shape', [name+'_shape'], [name+'_dim']),
            make_node('Squeeze', [name+'_dim', name+'_0'], [name+'_dim_s']),
            make_node('Range', [name+'_0_s', name+'_dim_s', name+'_1_s'], [name+'_range'])
        ]
        if axis < 0:
            nodes += [
                make_node('Add', [name+'_axis', name+'_dim'], [name+'_true_axis']),
                make_node('Equal', [name+'_range', name+'_true_axis'], [name+'_one_hot'])
                ]
        else:
            nodes += [
                make_node('Equal', [name+'_range', name+'_axis'], [name+'_one_hot'])
                ]
        nodes += [
            make_node('Cast', [name+'_one_hot'], [name+'_one_hot_int'], to=int(TensorProto.INT64)),
            make_node('Mul', [name+'_repeats', name+'_one_hot_int'], [name+'_mul']),
            make_node('Add', [name+'_mul', name+'_1'], [name+'_add']),
            make_node('Concat', [name+'_1', name+'_add'], [name+'_repeats_tensor'], axis=0)
            ]
        if axis == -1:
            nodes += [
                make_node('Concat', [name+'_shape', name+'_1'], [name+'_unsqueeze_shape'], axis=0),
                make_node('Reshape', [input_nodes[0], name+'_unsqueeze_shape'],
                          [name+'_unsqueeze'])
                ]
        else:
            create_tensor([axis+1], name+'_axis+1', kwargs['initializer'])
            nodes += [
                make_node('Unsqueeze', [input_nodes[0], name+'_axis+1'], [name+'_unsqueeze'])
                ]
        nodes += [
            make_node('Tile', [name+'_unsqueeze', name+'_repeats_tensor'], [name+'_tile']),
            make_node('Mul', [name+'_shape', name+'_add'], [name+'_new_shape']),
            make_node('Reshape', [name+'_tile', name+'_new_shape'], [name], name=name)
            ]

    return nodes


@mx_op.register('_contrib_box_nms', OPSET_VERSION)
def convert_contrib_box_nms(node, **kwargs):
    """Map MXNet's _contrib_box_nms operator to ONNX
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)

    dtype = input_dtypes[0]
    #dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]

    opset_version = kwargs['opset_version']
    if opset_version < 11:
        raise AttributeError('ONNX opset 11 or greater is required to export this operator')

    overlap_thresh = float(attrs.get('overlap_thresh', '0.5'))
    valid_thresh = float(attrs.get('valid_thresh', '0'))
    topk = int(attrs.get('topk', '-1'))
    coord_start = int(attrs.get('coord_start', '2'))
    score_index = int(attrs.get('score_index', '1'))
    id_index = int(attrs.get('id_index', '-1'))
    force_suppress = attrs.get('force_suppress', 'True')
    background_id = int(attrs.get('background_id', '-1'))
    in_format = attrs.get('in_format', 'corner')
    out_format = attrs.get('out_format', 'corner')

    center_point_box = 0 if in_format == 'corner' else 1

    if topk == -1:
        topk = 2**31-1

    if in_format != out_format:
        raise NotImplementedError('box_nms does not currently support in_fomat != out_format')

    if background_id != -1:
        raise NotImplementedError('box_nms does not currently support background_id != -1')

    if id_index != -1 or force_suppress == 'False':
        logging.warning('box_nms: id_idex != -1 or/and force_suppress == False detected. '
                        'However, due to ONNX limitations, boxes of different categories will NOT '
                        'be exempted from suppression. This might lead to different behavior than '
                        'native MXNet')

    create_tensor([coord_start], name+'_cs', kwargs['initializer'])
    create_tensor([coord_start+4], name+'_cs_p4', kwargs['initializer'])
    create_tensor([score_index], name+'_si', kwargs['initializer'])
    create_tensor([score_index+1], name+'_si_p1', kwargs['initializer'])
    create_tensor([topk], name+'_topk', kwargs['initializer'])
    create_tensor([overlap_thresh], name+'_ot', kwargs['initializer'], dtype=np.float32)
    create_tensor([valid_thresh], name+'_vt', kwargs['initializer'], dtype=np.float32)
    create_tensor([-1], name+'_m1', kwargs['initializer'])
    create_tensor([-1], name+'_m1_f', kwargs['initializer'], dtype=dtype)
    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([2], name+'_2', kwargs['initializer'])
    create_tensor([3], name+'_3', kwargs['initializer'])
    create_tensor([0, 1, -1], name+'_scores_shape', kwargs['initializer'])
    create_tensor([0, 0, 1, 0], name+'_pad', kwargs['initializer'])
    create_tensor([0, -1], name+'_bat_spat_helper', kwargs['initializer'])
    create_const_scalar_node(name+"_0_s", np.int64(0), kwargs)
    create_const_scalar_node(name+"_1_s", np.int64(1), kwargs)

    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Shape', [name+'_shape'], [name+'_dim']),
        make_node('Sub', [name+'_dim', name+'_2'], [name+'_dim_m2']),
        make_node('Slice', [name+'_shape', name+'_dim_m2', name+'_dim'], [name+'_shape_last2']),
        make_node('Concat', [name+'_m1', name+'_shape_last2'], [name+'_shape_3d'], axis=0),
        make_node('Reshape', [input_nodes[0], name+'_shape_3d'], [name+'_data_3d']),
        make_node('Slice', [name+'_data_3d', name+'_cs', name+'_cs_p4', name+'_m1'],
                  [name+'_boxes']),
        make_node('Slice', [name+'_data_3d', name+'_si', name+'_si_p1', name+'_m1'],
                  [name+'_scores_raw']),
        make_node('Reshape', [name+'_scores_raw', name+'_scores_shape'], [name+'_scores']),
        make_node('Shape', [name+'_scores'], [name+'_scores_shape_actual']),
        make_node('NonMaxSuppression',
                  [name+'_boxes', name+'_scores', name+'_topk', name+'_ot', name+'_vt'],
                  [name+'_nms'], center_point_box=center_point_box),
        make_node('Slice', [name+'_nms', name+'_0', name+'_3', name+'_m1', name+'_2'],
                  [name+'_nms_sliced']),
        make_node('GatherND', [name+'_data_3d', name+'_nms_sliced'], [name+'_candidates']),
        make_node('Pad', [name+'_candidates', name+'_pad', name+'_m1_f'], [name+'_cand_padded']),
        make_node('Shape', [name+'_nms'], [name+'_nms_shape']),
        make_node('Slice', [name+'_nms_shape', name+'_0', name+'_1'], [name+'_cand_cnt']),
        make_node('Squeeze', [name+'_cand_cnt', name+'_0'], [name+'_cc_s']),
        make_node('Range', [name+'_0_s', name+'_cc_s', name+'_1_s'], [name+'_cand_indices']),
        make_node('Slice', [name+'_scores_shape_actual', name+'_0', name+'_3', name+'_m1',
                            name+'_2'], [name+'_shape_bat_spat']),
        make_node('Slice', [name+'_shape_bat_spat', name+'_1', name+'_2'], [name+'_spat_dim']),
        make_node('Expand', [name+'_cand_cnt', name+'_shape_bat_spat'], [name+'_base_indices']),
        make_node('ScatterND', [name+'_base_indices', name+'_nms_sliced', name+'_cand_indices'],
                  [name+'_indices']),
        make_node('TopK', [name+'_indices', name+'_spat_dim'], [name+'_indices_sorted', name+'__'],
                  largest=0, axis=-1, sorted=1),
        make_node('Gather', [name+'_cand_padded', name+'_indices_sorted'], [name+'_gather']),
        make_node('Reshape', [name+'_gather', name+'_shape'], [name+'0'])
    ]

    return nodes


@mx_op.register('_contrib_ROIAlign', OPSET_VERSION)
def convert_contrib_roialign(node, **kwargs):
    """Map MXNet's _contrib_ROIAlign
    """
    from onnx.helper import make_node
    from onnx import TensorProto
    name, input_nodes, attrs = get_inputs(node, kwargs)

    pooled_size = convert_string_to_list(str(attrs.get('pooled_size')))
    spatial_scale = float(attrs.get('spatial_scale'))
    sample_ratio = int(attrs.get('sample_ratio', '0'))
    position_sensitive = attrs.get('position_sensitive', 'False')
    aligned = attrs.get('aligned', 'False')

    if position_sensitive != 'False':
        raise NotImplementedError('_contrib_ROIAlign does not currently support \
                                   position_sensitive!=False')
    if aligned != 'False':
        raise NotImplementedError('_contrib_ROIAlign does not currently support \
                                   aligned!=False')

    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([0], name+'_0_s', kwargs['initializer'], dtype='float32')
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([5], name+'_5', kwargs['initializer'])
    create_tensor([2, 3], name+'_2_3', kwargs['initializer'])

    nodes = [
        make_node('Slice', [input_nodes[1], name+'_1', name+'_5', name+'_1'], [name+'_rois']),
        make_node('Slice', [input_nodes[1], name+'_0', name+'_1', name+'_1'], [name+'_inds___']),
        make_node('Squeeze', [name+'_inds___', name+'_1'], [name+'_inds__']),
        make_node('Relu', [name+'_inds__'], [name+'_inds_']),
        make_node('Cast', [name+'_inds_'], [name+'_inds'], to=int(TensorProto.INT64)),
        make_node('RoiAlign', [input_nodes[0], name+'_rois', name+'_inds'], [name+'_roi'],
                  mode='avg', output_height=pooled_size[0], output_width=pooled_size[1],
                  sampling_ratio=sample_ratio, spatial_scale=spatial_scale),
        make_node('Unsqueeze', [name+'_inds___', name+'_2_3'], [name+'_unsq']),
        make_node('Less', [name+'_unsq', name+'_0_s'], [name+'_less']),
        make_node('Where', [name+'_less', name+'_0_s', name+'_roi'], [name])
    ]

    return nodes


@mx_op.register("sum", OPSET_VERSION)
@mx_op.register("_npi_sum", OPSET_VERSION)
def convert_sum(node, **kwargs):
    """Map MXNet's sum operator attributes to onnx's ReduceSum operator
    and return the created node.
    """
    from onnx.helper import make_node

    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = attrs.get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis not in [None, 'None'] else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes:
        create_tensor(axes, name+'_axes', kwargs['initializer'])
        input_nodes.append(name+'_axes')
        node = make_node(
            'ReduceSum',
            inputs=input_nodes,
            outputs=[name],
            keepdims=keepdims,
            name=name
        )
        return [node]
    else:
        create_tensor([1], name+'_1', kwargs['initializer'])
        nodes = [
            onnx.helper.make_node(
                'ReduceSum',
                inputs=input_nodes,
                outputs=[name],
                keepdims=keepdims,
            )
        ]
    return nodes


@mx_op.register("RNN", OPSET_VERSION)
def convert_RNN(node, **kwargs):
    """Map MXNet's RNN operator attributes to onnx's operators
    and return the created node.
    """
    from onnx.helper import make_node, make_tensor
    from onnx import TensorProto

    name, input_nodes, attrs = get_inputs(node, kwargs)

    mode = str(attrs.get('mode'))
    bidirectional = str(attrs.get('bidirectional', 'False'))
    if bidirectional != 'False' and mode not in ['lstm']:
        raise NotImplementedError('Currently RNN onnx export only supports bidirectional is False')

    num_layers = int(attrs.get('num_layers', '1'))

    use_sequence_length = str(attrs.get('use_sequence_length', 'False'))
    if use_sequence_length != 'False':
        raise NotImplementedError('Currently RNN onnx export only supports use_sequence_length equals to False')

    projection_size = str(attrs.get('projection_size', 'None'))
    if projection_size != 'None':
        raise NotImplementedError('Currently RNN onnx export only supports projection_size equals to None')

    state_outputs = str(attrs.get('state_outputs', 'False'))
    if state_outputs != 'True':
        raise NotImplementedError('Currently RNN onnx export only supports state_outputs equals to True')

    state_size = int(attrs.get('state_size'))

    direction = 1
    if bidirectional != 'False':
        direction = 2

    data = input_nodes[0]
    param = input_nodes[1]
    dtype = get_input_dtypes(node, kwargs)[2]

    create_tensor([0], name+'_0', kwargs['initializer'])
    create_tensor([1], name+'_1', kwargs['initializer'])
    create_tensor([state_size], name+'_state_size', kwargs['initializer'])
    create_tensor([direction], name+'_direction', kwargs['initializer'])

    tensor_1 = make_tensor(name+'_1_f', onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype], [1], [1])

    nodes = [
        make_node('Shape', [data], [name+'_data_shape']),
        make_node('Split', [name+'_data_shape'], [name+'_seq_length', name+'_batch_size', name+'_input_size']),
        make_node('Concat', [name+'_direction', name+'_batch_size', name+'_state_size'], [name+'_concat'], axis=0),
        make_node('ConstantOfShape', [name+'_concat'], [name+'_COS'], value=tensor_1),
        make_node('Mul', [input_nodes[2], name+'_COS'], [name+'initial_h']),

    ]

    if mode == 'lstm':
        nodes += [
            make_node('Mul', [input_nodes[3], name+'_COS'], [name+'initial_c']),
        ]

        if num_layers == 2:
            if bidirectional != 'False':
                raise NotImplementedError('Currently lstm onnx export only supports bidirectional when num_layers = 1')
            create_tensor([8*state_size], name+'_8*state_size', kwargs['initializer'])
            create_tensor([4*state_size*state_size], name+'_4*state_size^2', kwargs['initializer'])
            create_tensor([1, 4*state_size, state_size], name+'_WR_shape', kwargs['initializer'])
            create_tensor([1, 8*state_size], name+'_B_shape', kwargs['initializer'])
            create_tensor([4*4*state_size*state_size], name+'_WR_offset', kwargs['initializer'])

            nodes += [
                # Layer 0
                # get W
                make_node('Slice', [param, name+'_0', name+'_4*state_size^2'], [name+'_W0_1d']),
                make_node('Split', [name+'_W0_1d'], [name+'_W00', name+'_W01', name+'_W02', name+'_W03']),
                make_node('Concat', [name+'_W00', name+'_W03', name+'_W01', name+'_W02'], [name+'_W0_'], axis=0),
                make_node('Reshape', [name+'_W0_', name+'_WR_shape'], [name+'_W0']),
                # get R
                make_node('Add', [name+'_4*state_size^2', name+'_4*state_size^2'], [name+'_R0_offset']),
                make_node('Slice', [param, name+'_4*state_size^2', name+'_R0_offset'], [name+'_R0_1d']),
                make_node('Split', [name+'_R0_1d'], [name+'_R00', name+'_R01', name+'_R02', name+'_R03']),
                make_node('Concat', [name+'_R00', name+'_R03', name+'_R01', name+'_R02'], [name+'_R0_'], axis=0),
                make_node('Reshape', [name+'_R0_', name+'_WR_shape'], [name+'_R0']),
                # get B
                make_node('Add', [name+'_WR_offset', name+'_8*state_size'], [name+'_B0_offset']),
                make_node('Slice', [param, name+'_WR_offset', name+'_B0_offset'], [name+'_B0_1d']),
                make_node('Split', [name+'_B0_1d'], [name+'_B00', name+'_B01', name+'_B02', name+'_B03',
                                                     name+'_B04', name+'_B05', name+'_B06', name+'_B07']),
                make_node('Concat', [name+'_B00', name+'_B03', name+'_B01', name+'_B02',
                                     name+'_B04', name+'_B07', name+'_B05', name+'_B06'], [name+'_B0_'], axis=0),
                make_node('Reshape', [name+'_B0_', name+'_B_shape'], [name+'_B0']),
                # get initial states
                make_node('Split', [name+'initial_h'], [name+'_initial_h0', name+'_initial_h1'], axis=0),
                make_node('Split', [name+'initial_c'], [name+'_initial_c0', name+'_initial_c1'], axis=0),
                # get seq_len
                make_node('Tile', [name+'_seq_length', name+'_batch_size'], [name+'_seq_len_']),
                make_node("Cast", [name+'_seq_len_'], [name+"_seq_len"], to=int(TensorProto.INT32)),
                # Layer 0 LSTM
                make_node('LSTM', [data, name+'_W0', name+'_R0', name+'_B0', name+'_seq_len',
                                   name+'_initial_h0', name+'_initial_c0'],
                          [name+'_lstm0_out_', name+'_lstm0_h', name+'_lstm0_c'], hidden_size=state_size),
                make_node('Squeeze', [name+'_lstm0_out_', name+'_1'], [name+'_lstm0_out']),

                # Layer 1
                # get W
                make_node('Add', [name+'_R0_offset', name+'_4*state_size^2'], [name+'_W1_offset']),
                make_node('Slice', [param, name+'_R0_offset', name+'_W1_offset'], [name+'_W1_1d']),
                make_node('Split', [name+'_W1_1d'], [name+'_W10', name+'_W11', name+'_W12', name+'_W13']),
                make_node('Concat', [name+'_W10', name+'_W13', name+'_W11', name+'_W12'], [name+'_W1_'], axis=0),
                make_node('Reshape', [name+'_W1_', name+'_WR_shape'], [name+'_W1']),
                # get R
                make_node('Slice', [param, name+'_W1_offset', name+'_WR_offset'], [name+'_R1_1d']),
                make_node('Split', [name+'_R1_1d'], [name+'_R10', name+'_R11', name+'_R12', name+'_R13']),
                make_node('Concat', [name+'_R10', name+'_R13', name+'_R11', name+'_R12'], [name+'_R1_'], axis=0),
                make_node('Reshape', [name+'_R1_', name+'_WR_shape'], [name+'_R1']),
                # get B
                make_node('Add', [name+'_B0_offset', name+'_8*state_size'], [name+'_B1_offset']),
                make_node('Slice', [param, name+'_B0_offset', name+'_B1_offset'], [name+'_B1_1d']),
                make_node('Split', [name+'_B1_1d'], [name+'_B10', name+'_B11', name+'_B12', name+'_B13',
                                                     name+'_B14', name+'_B15', name+'_B16', name+'_B17']),
                make_node('Concat', [name+'_B10', name+'_B13', name+'_B11', name+'_B12',
                                     name+'_B14', name+'_B17', name+'_B15', name+'_B16'], [name+'_B1_'], axis=0),
                make_node('Reshape', [name+'_B1_', name+'_B_shape'], [name+'_B1']),
                # Layer 1 LSTM
                make_node('LSTM', [name+'_lstm0_out', name+'_W1', name+'_R1', name+'_B1', name+'_seq_len',
                                   name+'_initial_h1', name+'_initial_c1'],
                          [name+'_lstm1_out_', name+'_lstm1_h', name+'_lstm1_c'], hidden_size=state_size),
                make_node('Squeeze', [name+'_lstm1_out_', name+'_1'], [name]),
                make_node('Concat', [name+'_lstm0_h', name+'_lstm1_h'], [name+'1'], axis=0),
                make_node('Concat', [name+'_lstm0_c', name+'_lstm1_c'], [name+'2'], axis=0),
            ]
        elif num_layers == 1:
            if bidirectional == 'False':
                create_tensor([4*state_size], name+'_4*state_size', kwargs['initializer'])
                create_tensor([8*state_size], name+'_8*state_size', kwargs['initializer'])
                create_tensor([4*state_size*state_size], name+'_4*state_size^2', kwargs['initializer'])
                create_tensor([1, 4*state_size, state_size], name+'_R_shape', kwargs['initializer'])
                create_tensor([1, 8*state_size], name+'_B_shape', kwargs['initializer'])

                nodes += [
                    # get W
                    make_node('Mul', [name+'_4*state_size', name+'_input_size'], [name+'_mul0']),
                    make_node('Slice', [param, name+'_0', name+'_mul0'], [name+'_W_1d']),
                    make_node('Split', [name+'_W_1d'], [name+'_W0', name+'_W1', name+'_W2', name+'_W3']),
                    make_node('Concat', [name+'_W0', name+'_W3', name+'_W1', name+'_W2'], [name+'_W_'], axis=0),
                    make_node('Concat', [name+'_1', name+'_4*state_size', name+'_input_size'],
                              [name+'_W_shape'], axis=0),
                    make_node('Reshape', [name+'_W_', name+'_W_shape'], [name+'_W']),
                    # get R
                    make_node('Add', [name+'_mul0', name+'_4*state_size^2'], [name+'_add0']),
                    make_node('Slice', [param, name+'_mul0', name+'_add0'], [name+'_R_1d']),
                    make_node('Split', [name+'_R_1d'], [name+'_R0', name+'_R1', name+'_R2', name+'_R3']),
                    make_node('Concat', [name+'_R0', name+'_R3', name+'_R1', name+'_R2'], [name+'_R_'], axis=0),
                    make_node('Reshape', [name+'_R_', name+'_R_shape'], [name+'_R']),
                    # get B
                    make_node('Add', [name+'_add0', name+'_8*state_size'], [name+'_add1']),
                    make_node('Slice', [param, name+'_add0', name+'_add1'], [name+'_B_1d']),
                    make_node('Split', [name+'_B_1d'], [name+'_B0', name+'_B1', name+'_B2', name+'_B3',
                                                        name+'_B4', name+'_B5', name+'_B6', name+'_B7']),
                    make_node('Concat', [name+'_B0', name+'_B3', name+'_B1', name+'_B2',
                                         name+'_B4', name+'_B7', name+'_B5', name+'_B6'], [name+'_B_'], axis=0),
                    make_node('Reshape', [name+'_B_', name+'_B_shape'], [name+'_B']),
                    # get seq_len
                    make_node('Tile', [name+'_seq_length', name+'_batch_size'], [name+'_seq_len_']),
                    make_node("Cast", [name+'_seq_len_'], [name+"_seq_len"], to=int(TensorProto.INT32)),
                    # compute LSTM
                    make_node('LSTM', [data, name+'_W', name+'_R', name+'_B',
                                       name+'_seq_len', name+'initial_h', name+'initial_c'],
                              [name+'0_', name+'1', name+'2'], hidden_size=state_size),
                    make_node('Squeeze', [name+'0_', name+'_1'], [name]),
                ]
            else:
                create_tensor([-1], name+'_-1', kwargs['initializer'])
                create_tensor([4*state_size], name+'_4*state_size', kwargs['initializer'])
                create_tensor([8*state_size], name+'_8*state_size', kwargs['initializer'])
                create_tensor([4*state_size*state_size], name+'_4*state_size^2', kwargs['initializer'])
                create_tensor([1, 4*state_size, state_size], name+'_R_shape', kwargs['initializer'])
                create_tensor([1, 8*state_size], name+'_B_shape', kwargs['initializer'])

                nodes += [
                    # get W_fwd
                    make_node('Mul', [name+'_4*state_size', name+'_input_size'], [name+'_mul0']),
                    make_node('Slice', [param, name+'_0', name+'_mul0'], [name+'_W_1d']),
                    make_node('Split', [name+'_W_1d'], [name+'_W0', name+'_W1', name+'_W2', name+'_W3']),
                    make_node('Concat', [name+'_W0', name+'_W3', name+'_W1', name+'_W2'],
                              [name+'_W_'], axis=0),
                    make_node('Concat', [name+'_1', name+'_4*state_size', name+'_input_size'],
                              [name+'_W_shape'], axis=0),
                    make_node('Reshape', [name+'_W_', name+'_W_shape'], [name+'_W_fwd']),
                    # get R_fwd
                    make_node('Add', [name+'_mul0', name+'_4*state_size^2'], [name+'_add0']),
                    make_node('Slice', [param, name+'_mul0', name+'_add0'], [name+'_R_1d']),
                    make_node('Split', [name+'_R_1d'], [name+'_R0', name+'_R1', name+'_R2', name+'_R3']),
                    make_node('Concat', [name+'_R0', name+'_R3', name+'_R1', name+'_R2'], [name+'_R_'], axis=0),
                    make_node('Reshape', [name+'_R_', name+'_R_shape'], [name+'_R_fwd']),
                    # get W_bwd
                    make_node('Add', [name+'_add0', name+'_mul0'], [name+'_add1']),
                    make_node('Slice', [param, name+'_add0', name+'_add1'], [name+'_W_1d_bwd']),
                    make_node('Split', [name+'_W_1d_bwd'],
                              [name+'_W0_bwd', name+'_W1_bwd', name+'_W2_bwd', name+'_W3_bwd']),
                    make_node('Concat', [name+'_W0_bwd', name+'_W3_bwd', name+'_W1_bwd', name+'_W2_bwd'],
                              [name+'_W_bwd_'], axis=0),
                    make_node('Reshape', [name+'_W_bwd_', name+'_W_shape'], [name+'_W_bwd']),
                    # get R_bwd
                    make_node('Add', [name+'_add1', name+'_4*state_size^2'], [name+'_add2']),
                    make_node('Slice', [param, name+'_add1', name+'_add2'], [name+'_R_1d_bwd']),
                    make_node('Split', [name+'_R_1d_bwd'],
                              [name+'_R0_bwd', name+'_R1_bwd', name+'_R2_bwd', name+'_R3_bwd']),
                    make_node('Concat', [name+'_R0_bwd', name+'_R3_bwd', name+'_R1_bwd', name+'_R2_bwd'],
                              [name+'_R_bwd_'], axis=0),
                    make_node('Reshape', [name+'_R_bwd_', name+'_R_shape'], [name+'_R_bwd']),
                    # get B_fwd
                    make_node('Add', [name+'_add2', name+'_8*state_size'], [name+'_add3']),
                    make_node('Slice', [param, name+'_add2', name+'_add3'], [name+'_B_1d']),
                    make_node('Split', [name+'_B_1d'], [name+'_B0', name+'_B1', name+'_B2', name+'_B3',
                                                        name+'_B4', name+'_B5', name+'_B6', name+'_B7']),
                    make_node('Concat', [name+'_B0', name+'_B3', name+'_B1', name+'_B2',
                                         name+'_B4', name+'_B7', name+'_B5', name+'_B6'], [name+'_B_'], axis=0),
                    make_node('Reshape', [name+'_B_', name+'_B_shape'], [name+'_B_fwd']),
                    # get B_bwd
                    make_node('Add', [name+'_add3', name+'_8*state_size'], [name+'_add4']),
                    make_node('Slice', [param, name+'_add3', name+'_add4'], [name+'_B_1d_bwd']),
                    make_node('Split', [name+'_B_1d_bwd'],
                              [name+'_B0_bwd', name+'_B1_bwd', name+'_B2_bwd', name+'_B3_bwd',
                               name+'_B4_bwd', name+'_B5_bwd', name+'_B6_bwd', name+'_B7_bwd']),
                    make_node('Concat', [name+'_B0_bwd', name+'_B3_bwd', name+'_B1_bwd', name+'_B2_bwd',
                                         name+'_B4_bwd', name+'_B7_bwd', name+'_B5_bwd', name+'_B6_bwd'],
                              [name+'_B_bwd_'], axis=0),
                    make_node('Reshape', [name+'_B_bwd_', name+'_B_shape'], [name+'_B_bwd']),
                    # get seq_len
                    make_node('Tile', [name+'_seq_length', name+'_batch_size'], [name+'_seq_len_']),
                    make_node("Cast", [name+'_seq_len_'], [name+"_seq_len"], to=int(TensorProto.INT32)),
                    # compute LSTM
                    make_node('Concat', [name+'_W_fwd', name+'_W_bwd'], [name+'_W'], axis=0),
                    make_node('Concat', [name+'_R_fwd', name+'_R_bwd'], [name+'_R'], axis=0),
                    make_node('Concat', [name+'_B_fwd', name+'_B_bwd'], [name+'_B'], axis=0),
                    make_node('LSTM', [data, name+'_W', name+'_R', name+'_B',
                                       name+'_seq_len', name+'initial_h', name+'initial_c'],
                              [name+'0_', name+'1', name+'2'], hidden_size=state_size, direction='bidirectional'),
                    make_node('Transpose', [name+'0_'], [name+'0_t'], perm=[0, 2, 1, 3]),
                    make_node('Concat', [name+'_seq_length', name+'_batch_size', name+'_-1'],
                              [name+'_shape_out'], axis=0),
                    make_node('Reshape', [name+'0_t', name+'_shape_out'], [name]),
                ]
        else:
            raise NotImplementedError('Currently RNN onnx export only supports num_layers equals to 1 or 2')

    elif mode == 'gru':
        if num_layers == 2:
            create_tensor([6*state_size], name+'_6*state_size', kwargs['initializer'])
            create_tensor([3*state_size*state_size], name+'_3*state_size^2', kwargs['initializer'])
            create_tensor([1, 3*state_size, state_size], name+'_WR_shape', kwargs['initializer'])
            create_tensor([1, 6*state_size], name+'_B_shape', kwargs['initializer'])
            create_tensor([4*3*state_size*state_size], name+'_WR_offset', kwargs['initializer'])

            nodes += [
                # Layer 0
                # get W
                make_node('Slice', [param, name+'_0', name+'_3*state_size^2'], [name+'_W0_1d']),
                make_node('Split', [name+'_W0_1d'], [name+'_W00', name+'_W01', name+'_W02']),
                make_node('Concat', [name+'_W01', name+'_W00', name+'_W02'], [name+'_W0_'], axis=0),
                make_node('Reshape', [name+'_W0_', name+'_WR_shape'], [name+'_W0']),
                # get R
                make_node('Add', [name+'_3*state_size^2', name+'_3*state_size^2'], [name+'_R0_offset']),
                make_node('Slice', [param, name+'_3*state_size^2', name+'_R0_offset'], [name+'_R0_1d']),
                make_node('Split', [name+'_R0_1d'], [name+'_R00', name+'_R01', name+'_R02']),
                make_node('Concat', [name+'_R01', name+'_R00', name+'_R02'], [name+'_R0_'], axis=0),
                make_node('Reshape', [name+'_R0_', name+'_WR_shape'], [name+'_R0']),
                # get B
                make_node('Add', [name+'_WR_offset', name+'_6*state_size'], [name+'_B0_offset']),
                make_node('Slice', [param, name+'_WR_offset', name+'_B0_offset'], [name+'_B0_1d']),
                make_node('Split', [name+'_B0_1d'], [name+'_B00', name+'_B01', name+'_B02',
                                                     name+'_B03', name+'_B04', name+'_B05']),
                make_node('Concat', [name+'_B01', name+'_B00', name+'_B02',
                                     name+'_B04', name+'_B03', name+'_B05'], [name+'_B0_'], axis=0),
                make_node('Reshape', [name+'_B0_', name+'_B_shape'], [name+'_B0']),
                # get initial states
                make_node('Split', [name+'initial_h'], [name+'_initial_h0', name+'_initial_h1'], axis=0),
                # get seq_len
                make_node('Tile', [name+'_seq_length', name+'_batch_size'], [name+'_seq_len_']),
                make_node("Cast", [name+'_seq_len_'], [name+"_seq_len"], to=int(TensorProto.INT32)),
                # Layer 0 GRU
                make_node('GRU', [data, name+'_W0', name+'_R0', name+'_B0', name+'_seq_len',
                                  name+'_initial_h0'],
                          [name+'_gru0_out_', name+'_gru0_h'], hidden_size=state_size, linear_before_reset=1),
                make_node('Squeeze', [name+'_gru0_out_', name+'_1'], [name+'_gru0_out']),

                # Layer 1
                # get W
                make_node('Add', [name+'_R0_offset', name+'_3*state_size^2'], [name+'_W1_offset']),
                make_node('Slice', [param, name+'_R0_offset', name+'_W1_offset'], [name+'_W1_1d']),
                make_node('Split', [name+'_W1_1d'], [name+'_W10', name+'_W11', name+'_W12']),
                make_node('Concat', [name+'_W11', name+'_W10', name+'_W12'], [name+'_W1_'], axis=0),
                make_node('Reshape', [name+'_W1_', name+'_WR_shape'], [name+'_W1']),
                # get R
                make_node('Slice', [param, name+'_W1_offset', name+'_WR_offset'], [name+'_R1_1d']),
                make_node('Split', [name+'_R1_1d'], [name+'_R10', name+'_R11', name+'_R12']),
                make_node('Concat', [name+'_R11', name+'_R10', name+'_R12'], [name+'_R1_'], axis=0),
                make_node('Reshape', [name+'_R1_', name+'_WR_shape'], [name+'_R1']),
                # get B
                make_node('Add', [name+'_B0_offset', name+'_6*state_size'], [name+'_B1_offset']),
                make_node('Slice', [param, name+'_B0_offset', name+'_B1_offset'], [name+'_B1_1d']),
                make_node('Split', [name+'_B1_1d'], [name+'_B10', name+'_B11', name+'_B12',
                                                     name+'_B13', name+'_B14', name+'_B15']),
                make_node('Concat', [name+'_B11', name+'_B10', name+'_B12',
                                     name+'_B14', name+'_B13', name+'_B15'], [name+'_B1_'], axis=0),
                make_node('Reshape', [name+'_B1_', name+'_B_shape'], [name+'_B1']),
                # Layer 1 GRU
                make_node('GRU', [name+'_gru0_out', name+'_W1', name+'_R1', name+'_B1', name+'_seq_len',
                                  name+'_initial_h1'],
                          [name+'_gru1_out_', name+'_gru1_h'], hidden_size=state_size, linear_before_reset=1),
                make_node('Squeeze', [name+'_gru1_out_', name+'_1'], [name]),
                make_node('Concat', [name+'_gru0_h', name+'_gru1_h'], [name+'1'], axis=0)
            ]

        elif num_layers == 1:
            create_tensor([3*state_size], name+'_3*state_size', kwargs['initializer'])
            create_tensor([6*state_size], name+'_6*state_size', kwargs['initializer'])
            create_tensor([3*state_size*state_size], name+'_3*state_size^2', kwargs['initializer'])
            create_tensor([1, 3*state_size, state_size], name+'_R_shape', kwargs['initializer'])
            create_tensor([1, 6*state_size], name+'_B_shape', kwargs['initializer'])

            nodes += [
                # get W
                make_node('Mul', [name+'_3*state_size', name+'_input_size'], [name+'_mul0']),
                make_node('Slice', [param, name+'_0', name+'_mul0'], [name+'_W_1d']),
                make_node('Split', [name+'_W_1d'], [name+'_W0', name+'_W1', name+'_W2']),
                make_node('Concat', [name+'_W1', name+'_W0', name+'_W2'], [name+'_W_'], axis=0),
                make_node('Concat', [name+'_1', name+'_3*state_size', name+'_input_size'], [name+'_W_shape'], axis=0),
                make_node('Reshape', [name+'_W_', name+'_W_shape'], [name+'_W']),
                # get R
                make_node('Add', [name+'_mul0', name+'_3*state_size^2'], [name+'_add0']),
                make_node('Slice', [param, name+'_mul0', name+'_add0'], [name+'_R_1d']),
                make_node('Split', [name+'_R_1d'], [name+'_R0', name+'_R1', name+'_R2']),
                make_node('Concat', [name+'_R1', name+'_R0', name+'_R2'], [name+'_R_'], axis=0),
                make_node('Reshape', [name+'_R_', name+'_R_shape'], [name+'_R']),
                # get B
                make_node('Add', [name+'_add0', name+'_6*state_size'], [name+'_add1']),
                make_node('Slice', [param, name+'_add0', name+'_add1'], [name+'_B_1d']),
                make_node('Split', [name+'_B_1d'], [name+'_B0', name+'_B1', name+'_B2',
                                                    name+'_B3', name+'_B4', name+'_B5']),
                make_node('Concat', [name+'_B1', name+'_B0', name+'_B2',
                                     name+'_B4', name+'_B3', name+'_B5'], [name+'_B_'], axis=0),
                make_node('Reshape', [name+'_B_', name+'_B_shape'], [name+'_B']),
                # get seq_len
                make_node('Tile', [name+'_seq_length', name+'_batch_size'], [name+'_seq_len_']),
                make_node("Cast", [name+'_seq_len_'], [name+"_seq_len"], to=int(TensorProto.INT32)),
                # compute GRU
                make_node('GRU', [data, name+'_W', name+'_R', name+'_B', name+'_seq_len', name+'initial_h'],
                          [name+'0_', name+'1'], hidden_size=state_size, linear_before_reset=1),
                make_node('Squeeze', [name+'0_', name+'_1'], [name]),
            ]
        else:
            raise NotImplementedError('Currently RNN onnx export only supports num_layers equals to 1 or 2')

    elif mode in ['rnn_tanh', 'rnn_relu']:
        activations = ['Tanh']
        if mode == 'rnn_relu':
            activations = ['Relu']
        if num_layers == 2:
            create_tensor([2*state_size], name+'_2*state_size', kwargs['initializer'])
            create_tensor([state_size*state_size], name+'_state_size^2', kwargs['initializer'])
            create_tensor([1, state_size, state_size], name+'_WR_shape', kwargs['initializer'])
            create_tensor([1, 2*state_size], name+'_B_shape', kwargs['initializer'])
            create_tensor([4*state_size*state_size], name+'_WR_offset', kwargs['initializer'])

            nodes += [
                # Layer 0
                # get W
                make_node('Slice', [param, name+'_0', name+'_state_size^2'], [name+'_W0_1d']),
                make_node('Reshape', [name+'_W0_1d', name+'_WR_shape'], [name+'_W0']),
                # get R
                make_node('Add', [name+'_state_size^2', name+'_state_size^2'], [name+'_R0_offset']),
                make_node('Slice', [param, name+'_state_size^2', name+'_R0_offset'], [name+'_R0_1d']),
                make_node('Reshape', [name+'_R0_1d', name+'_WR_shape'], [name+'_R0']),
                # get B
                make_node('Add', [name+'_WR_offset', name+'_2*state_size'], [name+'_B0_offset']),
                make_node('Slice', [param, name+'_WR_offset', name+'_B0_offset'], [name+'_B0_1d']),
                make_node('Reshape', [name+'_B0_1d', name+'_B_shape'], [name+'_B0']),
                # get initial states
                make_node('Split', [name+'initial_h'], [name+'_initial_h0', name+'_initial_h1'], axis=0),
                # get seq_len
                make_node('Tile', [name+'_seq_length', name+'_batch_size'], [name+'_seq_len_']),
                make_node("Cast", [name+'_seq_len_'], [name+"_seq_len"], to=int(TensorProto.INT32)),
                # Layer 0 RNN
                make_node('RNN', [data, name+'_W0', name+'_R0', name+'_B0', name+'_seq_len',
                                  name+'_initial_h0'], [name+'_rnn0_out_', name+'_rnn0_h'],
                          hidden_size=state_size, activations=activations),
                make_node('Squeeze', [name+'_rnn0_out_', name+'_1'], [name+'_rnn0_out']),

                # Layer 1
                # get W
                make_node('Add', [name+'_R0_offset', name+'_state_size^2'], [name+'_W1_offset']),
                make_node('Slice', [param, name+'_R0_offset', name+'_W1_offset'], [name+'_W1_1d']),
                make_node('Reshape', [name+'_W1_1d', name+'_WR_shape'], [name+'_W1']),
                # get R
                make_node('Slice', [param, name+'_W1_offset', name+'_WR_offset'], [name+'_R1_1d']),
                make_node('Reshape', [name+'_R1_1d', name+'_WR_shape'], [name+'_R1']),
                # get B
                make_node('Add', [name+'_B0_offset', name+'_2*state_size'], [name+'_B1_offset']),
                make_node('Slice', [param, name+'_B0_offset', name+'_B1_offset'], [name+'_B1_1d']),
                make_node('Reshape', [name+'_B1_1d', name+'_B_shape'], [name+'_B1']),
                # Layer 1 RNN
                make_node('RNN', [name+'_rnn0_out', name+'_W1', name+'_R1', name+'_B1', name+'_seq_len',
                                  name+'_initial_h1'], [name+'_rnn1_out_', name+'_rnn1_h'],
                          hidden_size=state_size, activations=activations),
                make_node('Squeeze', [name+'_rnn1_out_', name+'_1'], [name]),
                make_node('Concat', [name+'_rnn0_h', name+'_rnn1_h'], [name+'1'], axis=0)
            ]

        elif num_layers == 1:
            create_tensor([2*state_size], name+'_2*state_size', kwargs['initializer'])
            create_tensor([state_size*state_size], name+'_state_size^2', kwargs['initializer'])
            create_tensor([1, state_size, state_size], name+'_R_shape', kwargs['initializer'])
            create_tensor([1, 2*state_size], name+'_B_shape', kwargs['initializer'])

            nodes += [
                # get W
                make_node('Mul', [name+'_state_size', name+'_input_size'], [name+'_mul0']),
                make_node('Slice', [param, name+'_0', name+'_mul0'], [name+'_W_1d']),
                make_node('Concat', [name+'_1', name+'_state_size', name+'_input_size'], [name+'_W_shape'], axis=0),
                make_node('Reshape', [name+'_W_1d', name+'_W_shape'], [name+'_W']),
                # get R
                make_node('Add', [name+'_mul0', name+'_state_size^2'], [name+'_add0']),
                make_node('Slice', [param, name+'_mul0', name+'_add0'], [name+'_R_1d']),
                make_node('Reshape', [name+'_R_1d', name+'_R_shape'], [name+'_R']),
                # get B
                make_node('Add', [name+'_add0', name+'_2*state_size'], [name+'_add1']),
                make_node('Slice', [param, name+'_add0', name+'_add1'], [name+'_B_1d']),
                make_node('Reshape', [name+'_B_1d', name+'_B_shape'], [name+'_B']),
                # get seq_len
                make_node('Tile', [name+'_seq_length', name+'_batch_size'], [name+'_seq_len_']),
                make_node("Cast", [name+'_seq_len_'], [name+"_seq_len"], to=int(TensorProto.INT32)),
                # compute RNN
                make_node('RNN', [data, name+'_W', name+'_R', name+'_B', name+'_seq_len', name+'initial_h'],
                          [name+'0_', name+'1'], hidden_size=state_size, activations=activations),
                make_node('Squeeze', [name+'0_', name+'_1'], [name]),
            ]
        else:
            raise NotImplementedError('Currently RNN onnx export only supports num_layers equals to 1 or 2')
    else:
        raise NotImplementedError(f"Currently RNN onnx export does not support {mode} mode")
    return nodes


@mx_op.register('SliceChannel', OPSET_VERSION)
def convert_slice_channel(node, **kwargs):
    """Map MXNet's SliceChannel operator attributes to onnx's Squeeze or Split
    operator based on squeeze_axis attribute
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    num_outputs = int(attrs.get('num_outputs'))
    axis = int(attrs.get('axis', 1))
    squeeze_axis = attrs.get('squeeze_axis', 'False')

    create_tensor([axis], name+'_axis', kwargs['initializer'])

    nodes = []
    if squeeze_axis in ['True', '1']:
        nodes += [
            make_node('Split', [input_nodes[0]], [name+str(i)+'_' for i in range(num_outputs)],
                      axis=axis)
        ]
        for i in range(num_outputs):
            nodes += [
                make_node('Squeeze', [name+str(i)+'_', name+'_axis'], [name+str(i)])
            ]
    else:
        nodes += [
            make_node('Split', [input_nodes[0]], [name+str(i) for i in range(num_outputs)],
                      axis=axis)
        ]

    return nodes


@mx_op.register("max", OPSET_VERSION)
def convert_max(node, **kwargs):
    """Map MXNet's max operator attributes to onnx's ReduceMax operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        if keepdims:
            node = make_node('ReduceMax', input_nodes, [name], axes=axes, keepdims=keepdims)
            return [node]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            create_tensor([len(axes)], name+'_axes_dim', kwargs['initializer'])
            nodes = [
                make_node('ReduceMax', input_nodes, [name+'_rmax'], axes=axes, keepdims=keepdims),
                make_node('Shape', [name+'_rmax'], [name+'_rmax_shape']),
                make_node('Shape', [name+'_rmax_shape'], [name+'_rmax_dim']),
                make_node('Shape', [input_nodes[0]], [name+'_in_shape']),
                make_node('Shape', [name+'_in_shape'], [name+'_in_dim']),
                make_node('Equal', [name+'_axes_dim', name+'_in_dim'], [name+'_equal']),
                make_node('Where', [name+'_equal', name+'_1', name+'_rmax_dim'], [name+'_where0']),
                make_node('Tile', [name+'_0', name+'_where0'], [name+'_tile']),
                make_node('Unsqueeze', [name+'_0', name+'_0'], [name+'_unsqueeze']),
                make_node('Where', [name+'_equal', name+'_1', name+'_0'], [name+'_where1']),
                make_node('ScatterND', [name+'_tile', name+'_unsqueeze', name+'_where1'], [name+'_SND']),
                make_node('Reshape', [name+'_rmax', name+'_SND'], [name]),
            ]
            return nodes
    else:
        if keepdims:
            node = make_node('ReduceMax', input_nodes, [name], keepdims=keepdims)
            return [node]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node('ReduceMax', input_nodes, [name+'_rmax'], keepdims=keepdims),
                make_node('Reshape', [name+'_rmax', name+'_1'], [name])
            ]
            return nodes


@mx_op.register("min", OPSET_VERSION)
def convert_min(node, **kwargs):
    """Map MXNet's min operator attributes to onnx's ReduceMin operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        if keepdims:
            node = make_node('ReduceMin', input_nodes, [name], axes=axes, keepdims=keepdims)
            return [node]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            create_tensor([len(axes)], name+'_axes_dim', kwargs['initializer'])
            nodes = [
                make_node('ReduceMin', input_nodes, [name+'_rmin'], axes=axes, keepdims=keepdims),
                make_node('Shape', [name+'_rmin'], [name+'_rmin_shape']),
                make_node('Shape', [name+'_rmin_shape'], [name+'_rmin_dim']),
                make_node('Shape', [input_nodes[0]], [name+'_in_shape']),
                make_node('Shape', [name+'_in_shape'], [name+'_in_dim']),
                make_node('Equal', [name+'_axes_dim', name+'_in_dim'], [name+'_equal']),
                make_node('Where', [name+'_equal', name+'_1', name+'_rmin_dim'], [name+'_where0']),
                make_node('Tile', [name+'_0', name+'_where0'], [name+'_tile']),
                make_node('Unsqueeze', [name+'_0', name+'_0'], [name+'_unsqueeze']),
                make_node('Where', [name+'_equal', name+'_1', name+'_0'], [name+'_where1']),
                make_node('ScatterND', [name+'_tile', name+'_unsqueeze', name+'_where1'], [name+'_SND']),
                make_node('Reshape', [name+'_rmin', name+'_SND'], [name]),
            ]
            return nodes
    else:
        if keepdims:
            node = make_node('ReduceMin', input_nodes, [name], keepdims=keepdims)
            return [node]

        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node('ReduceMin', input_nodes, [name+'_rmin'], keepdims=keepdims),
                make_node('Reshape', [name+'_rmin', name+'_1'], [name])
            ]
            return nodes


@mx_op.register("mean", OPSET_VERSION)
def convert_mean(node, **kwargs):
    """Map MXNet's mean operator attributes to onnx's ReduceMean operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        if keepdims:
            node = make_node('ReduceMean', input_nodes, [name], axes=axes, keepdims=keepdims)
            return [node]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            create_tensor([len(axes)], name+'_axes_dim', kwargs['initializer'])
            nodes = [
                make_node('ReduceMean', input_nodes, [name+'_reduce'], axes=axes, keepdims=keepdims),
                make_node('Shape', [name+'_reduce'], [name+'_reduce_shape']),
                make_node('Shape', [name+'_reduce_shape'], [name+'_reduce_dim']),
                make_node('Shape', [input_nodes[0]], [name+'_in_shape']),
                make_node('Shape', [name+'_in_shape'], [name+'_in_dim']),
                make_node('Equal', [name+'_axes_dim', name+'_in_dim'], [name+'_equal']),
                make_node('Where', [name+'_equal', name+'_1', name+'_reduce_dim'], [name+'_where0']),
                make_node('Tile', [name+'_0', name+'_where0'], [name+'_tile']),
                make_node('Unsqueeze', [name+'_0', name+'_0'], [name+'_unsqueeze']),
                make_node('Where', [name+'_equal', name+'_1', name+'_0'], [name+'_where1']),
                make_node('ScatterND', [name+'_tile', name+'_unsqueeze', name+'_where1'], [name+'_SND']),
                make_node('Reshape', [name+'_reduce', name+'_SND'], [name]),
            ]
            return nodes
    else:
        if keepdims:
            node = make_node('ReduceMean', input_nodes, [name], keepdims=keepdims)
            return [node]

        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node('ReduceMean', input_nodes, [name+'_reduce'], keepdims=keepdims),
                make_node('Reshape', [name+'_reduce', name+'_1'], [name])
            ]
            return nodes


@mx_op.register("prod", OPSET_VERSION)
def convert_prod(node, **kwargs):
    """Map MXNet's prod operator attributes to onnx's ReduceProd operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        if keepdims:
            node = make_node('ReduceProd', input_nodes, [name], axes=axes, keepdims=keepdims)
            return [node]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            create_tensor([len(axes)], name+'_axes_dim', kwargs['initializer'])
            nodes = [
                make_node('ReduceProd', input_nodes, [name+'_reduce'], axes=axes, keepdims=keepdims),
                make_node('Shape', [name+'_reduce'], [name+'_reduce_shape']),
                make_node('Shape', [name+'_reduce_shape'], [name+'_reduce_dim']),
                make_node('Shape', [input_nodes[0]], [name+'_in_shape']),
                make_node('Shape', [name+'_in_shape'], [name+'_in_dim']),
                make_node('Equal', [name+'_axes_dim', name+'_in_dim'], [name+'_equal']),
                make_node('Where', [name+'_equal', name+'_1', name+'_reduce_dim'], [name+'_where0']),
                make_node('Tile', [name+'_0', name+'_where0'], [name+'_tile']),
                make_node('Unsqueeze', [name+'_0', name+'_0'], [name+'_unsqueeze']),
                make_node('Where', [name+'_equal', name+'_1', name+'_0'], [name+'_where1']),
                make_node('ScatterND', [name+'_tile', name+'_unsqueeze', name+'_where1'], [name+'_SND']),
                make_node('Reshape', [name+'_reduce', name+'_SND'], [name]),
            ]
            return nodes
    else:
        if keepdims:
            node = make_node('ReduceProd', input_nodes, [name], keepdims=keepdims)
            return [node]

        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node('ReduceProd', input_nodes, [name+'_reduce'], keepdims=keepdims),
                make_node('Reshape', [name+'_reduce', name+'_1'], [name])
            ]
            return nodes


@mx_op.register("squeeze", OPSET_VERSION)
@mx_op.register("_npi_squeeze", OPSET_VERSION)
def convert_squeeze(node, **kwargs):
    """Map MXNet's squeeze operator attributes to onnx's squeeze operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    if not axes:
        node = onnx.helper.make_node(
            "Squeeze",
            input_nodes,
            [name],
            name=name
        )
    else:
        create_tensor(axes, name+'_axes', kwargs['initializer'])
        node = onnx.helper.make_node(
            "Squeeze",
            [input_nodes[0], name+'_axes'],
            [name],
            name=name,
        )
    return [node]


@mx_op.register("SoftmaxOutput", OPSET_VERSION)
def convert_softmax_output(node, **kwargs):
    """Map MXNet's SoftmaxOutput operator attributes to onnx's Softmax operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, _ = get_inputs(node, kwargs)

    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Flatten', [input_nodes[0]], [name+'_flat'], axis=1),
        make_node('Softmax', [name+'_flat'], [name+'_sm'], axis=1),
        make_node('Reshape', [name+'_sm', name+'_shape'], [name])
    ]

    return nodes


@mx_op.register("norm", OPSET_VERSION)
def convert_norm(node, **kwargs):
    """Map MXNet's norm operator attributes to onnx's ReduceL1 and ReduceL2 operators
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = attrs.get("axis", None)
    axes = convert_string_to_list(str(mx_axis)) if mx_axis else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")
    ord = int(attrs.get("ord", 2))

    onnx_op_name = "ReduceL1" if ord == 1 else "ReduceL2"

    if axes:
        if keepdims:
            reduce_node = make_node(onnx_op_name, input_nodes, [name], axes=axes, keepdims=keepdims)
            return [reduce_node]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            create_tensor([len(axes)], name+'_axes_dim', kwargs['initializer'])
            nodes = [
                make_node(onnx_op_name, input_nodes, [name+'_reduce'], axes=axes, keepdims=keepdims),
                make_node('Shape', [name+'_reduce'], [name+'_reduce_shape']),
                make_node('Shape', [name+'_reduce_shape'], [name+'_reduce_dim']),
                make_node('Shape', [input_nodes[0]], [name+'_in_shape']),
                make_node('Shape', [name+'_in_shape'], [name+'_in_dim']),
                make_node('Equal', [name+'_axes_dim', name+'_in_dim'], [name+'_equal']),
                make_node('Where', [name+'_equal', name+'_1', name+'_reduce_dim'], [name+'_where0']),
                make_node('Tile', [name+'_0', name+'_where0'], [name+'_tile']),
                make_node('Unsqueeze', [name+'_0', name+'_0'], [name+'_unsqueeze']),
                make_node('Where', [name+'_equal', name+'_1', name+'_0'], [name+'_where1']),
                make_node('ScatterND', [name+'_tile', name+'_unsqueeze', name+'_where1'], [name+'_SND']),
                make_node('Reshape', [name+'_reduce', name+'_SND'], [name]),
            ]
            return nodes
    else:

        if keepdims:
            reduce_node = make_node(onnx_op_name, input_nodes, [name], keepdims=keepdims)
            return [reduce_node]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node(onnx_op_name, input_nodes, [name+'_norm'], keepdims=keepdims),
                make_node('Reshape', [name+'_norm', name+'_1'], [name])
            ]
            return nodes


@mx_op.register("log_softmax", OPSET_VERSION)
def convert_logsoftmax(node, **kwargs):
    """Map MXNet's log_softmax operator attributes to onnx's LogSoftMax operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)

    # Converting to int
    axis = int(attrs.get("axis", -1))
    temp = attrs.get('temperature', 'None')
    use_length = attrs.get('use_length', 'False')

    if temp != 'None':
        raise AttributeError('LogSoftMax currently does not support temperature!=None')

    if use_length in ['1', 'True']:
        raise AttributeError('LogSoftMax currently does not support use_length==True')

    node = onnx.helper.make_node(
        'LogSoftmax',
        input_nodes,
        [name],
        axis=axis,
        name=name
    )

    return [node]


@mx_op.register('_split_v2', OPSET_VERSION)
def convert_contrib_split_v2(node, **kwargs):
    """Map MXNet's _split_v2 operator
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get('axis', 0))
    squeeze_axis = attrs.get('squeeze_axis', 'False')
    sections = int(attrs.get('sections', 0))
    indices = convert_string_to_list(attrs.get('indices', '[]'))
    if sections <= 0 and len(indices) == 0:
        raise NotImplementedError('section or indices must be set')
    if sections > 0:
        output_nodes = [name+str(i) for i in range(sections)]
        if squeeze_axis == 'False':
            nodes = [
                make_node('Split', input_nodes, output_nodes, axis=axis),
            ]
        else:
            output_nodes_ = [name+str(i)+'_' for i in range(sections)]
            create_tensor([axis], name+'_axis', kwargs['initializer'])
            nodes = [
                make_node('Split', input_nodes, output_nodes_, axis=axis),
            ]
            for i in range(sections):
                nodes += [
                    make_node("Squeeze", [output_nodes_[i], name+'_axis'], [output_nodes[i]]),
                ]
    else:
        indices.sort()
        split = []
        for i in range(1, len(indices)):
            if indices[i] >= indices[i-1]:
                split.append(indices[i] - indices[i-1])

        output_nodes = [name+str(i) for i in range(len(split)+1)]
        create_tensor([0], name+'_0', kwargs['initializer'])
        create_tensor([axis], name+'_axis', kwargs['initializer'])
        create_tensor([axis+1], name+'_axis+1', kwargs['initializer'])
        create_tensor(split, name+'_split_', kwargs['initializer'])
        create_tensor([sum(split)], name+'_sum', kwargs['initializer'])
        nodes = [
            make_node('Shape', input_nodes, [name+'_shape']),
            make_node('Slice', [name+'_shape', name+'_axis', name+'_axis+1', name+'_0'], [name+'_dim']),
            make_node('Sub', [name+'_dim', name+'_sum'], [name+'_sub']),
            make_node('Concat', [name+'_split_', name+'_sub'], [name+'_concat'], axis=0),
            make_node('Less', [name+'_concat', name+'_0'], [name+'_less']),
            make_node('Where', [name+'_less', name+'_0', name+'_concat'], [name+'_split']),
            ]
        if squeeze_axis == 'False':
            nodes += [
                make_node('Split', [input_nodes[0], name+'_split'], output_nodes, axis=axis),
            ]
        else:
            output_nodes_ = [name+str(i)+'_' for i in range(len(split)+1)]
            nodes += [
                make_node('Split', [input_nodes[0], name+'_split'], output_nodes_, axis=axis),
            ]
            for i, output_node in enumerate(output_nodes):
                nodes += [
                    make_node("Squeeze", [output_nodes_[i], name+'_axis'], [output_node]),
                ]

    return nodes


@mx_op.register("_npi_mean", OPSET_VERSION)
def convert_npi_mean(node, **kwargs):
    """Map MXNet's mean operator attributes to onnx's ReduceMean operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    dtype = np.dtype('float32')
    dtype_t = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        create_tensor(axes, name+'_axes', kwargs['initializer'])
        if keepdims:
            nodes = [
                make_node('Cast', input_nodes, [name+'_cast'], to=dtype_t),
                make_node('ReduceMean', [name+'_cast'], [name], axes=axes,
                          keepdims=keepdims),
            ]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            nodes = [
                make_node('Cast', input_nodes, [name+'_cast'], to=dtype_t),
                make_node('ReduceMean', [name+'_cast'], [name+'_reduce'], axes=axes,
                          keepdims=keepdims),
                make_node('Shape', [name+'_reduce'], [name+'_reduce_shape']),
                make_node('Concat', [name+'_1', name+'_reduce_shape'], [name+'_concat'], axis=0),
                make_node('Reshape', [name+'_reduce', name+'_concat'], [name+'_reshape']),
                make_node('Squeeze', [name+'_reshape', name+'_0'], [name]),
            ]
    else:
        if keepdims:
            nodes = [
                make_node('Cast', input_nodes, [name+'_cast'], to=dtype_t),
                make_node('ReduceMean', [name+'_cast'], [name], keepdims=keepdims),
            ]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node('Cast', input_nodes, [name+'_cast'], to=dtype_t),
                make_node('ReduceMean', [name+'_cast'], [name], keepdims=keepdims),
            ]
    return nodes, (dtype,)


@mx_op.register("_npi_prod", OPSET_VERSION)
def convert_npi_prod(node, **kwargs):
    """Map MXNet's prod operator attributes to onnx's ReduceProd operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        create_tensor(axes, name+'_axes', kwargs['initializer'])
        if keepdims:
            nodes = [
                make_node('ReduceProd', [input_nodes[0]], [name], axes=axes,
                          keepdims=keepdims),
            ]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            nodes = [
                make_node('ReduceProd', [input_nodes[0]], [name+'_reduce'], axes=axes,
                          keepdims=keepdims),
                make_node('Shape', [name+'_reduce'], [name+'_reduce_shape']),
                make_node('Concat', [name+'_1', name+'_reduce_shape'], [name+'_concat'], axis=0),
                make_node('Reshape', [name+'_reduce', name+'_concat'], [name+'_reshape']),
                make_node('Squeeze', [name+'_reshape', name+'_0'], [name]),
            ]
    else:
        if keepdims:
            nodes = [
                make_node('ReduceProd', [input_nodes[0]], [name], keepdims=keepdims),
            ]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node('ReduceProd', [input_nodes[0]], [name], keepdims=keepdims),
            ]
    return nodes


@mx_op.register("_npi_min", OPSET_VERSION)
def convert_npi_min(node, **kwargs):
    """Map MXNet's min operator attributes to onnx's ReduceMin operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        create_tensor(axes, name+'_axes', kwargs['initializer'])
        if keepdims:
            nodes = [
                make_node('ReduceMin', [input_nodes[0]], [name], axes=axes,
                          keepdims=keepdims),
            ]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            nodes = [
                make_node('ReduceMin', [input_nodes[0]], [name+'_reduce'], axes=axes,
                          keepdims=keepdims),
                make_node('Shape', [name+'_reduce'], [name+'_reduce_shape']),
                make_node('Concat', [name+'_1', name+'_reduce_shape'], [name+'_concat'], axis=0),
                make_node('Reshape', [name+'_reduce', name+'_concat'], [name+'_reshape']),
                make_node('Squeeze', [name+'_reshape', name+'_0'], [name]),
            ]
    else:
        if keepdims:
            nodes = [
                make_node('ReduceMin', [input_nodes[0]], [name], keepdims=keepdims),
            ]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node('ReduceMin', [input_nodes[0]], [name], keepdims=keepdims),
            ]
    return nodes


@mx_op.register("_npi_max", OPSET_VERSION)
def convert_npi_max(node, **kwargs):
    """Map MXNet's min operator attributes to onnx's ReduceMin operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)

    mx_axis = str(attrs.get("axis", 'None'))
    axes = convert_string_to_list(mx_axis) if mx_axis != 'None' else None

    keepdims = get_boolean_attribute_value(attrs, "keepdims")

    if axes is not None:
        create_tensor(axes, name+'_axes', kwargs['initializer'])
        if keepdims:
            nodes = [
                make_node('ReduceMax', [input_nodes[0]], [name], axes=axes,
                          keepdims=keepdims),
            ]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            create_tensor([0], name+'_0', kwargs['initializer'])
            nodes = [
                make_node('ReduceMax', [input_nodes[0]], [name+'_reduce'], axes=axes,
                          keepdims=keepdims),
                make_node('Shape', [name+'_reduce'], [name+'_reduce_shape']),
                make_node('Concat', [name+'_1', name+'_reduce_shape'], [name+'_concat'], axis=0),
                make_node('Reshape', [name+'_reduce', name+'_concat'], [name+'_reshape']),
                make_node('Squeeze', [name+'_reshape', name+'_0'], [name]),
            ]
    else:
        if keepdims:
            nodes = [
                make_node('ReduceMax', [input_nodes[0]], [name], keepdims=keepdims),
            ]
        else:
            create_tensor([1], name+'_1', kwargs['initializer'])
            nodes = [
                make_node('ReduceMax', [input_nodes[0]], [name], keepdims=keepdims),
            ]
    return nodes
