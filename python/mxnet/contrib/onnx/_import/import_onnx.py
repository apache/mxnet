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

# coding: utf-8
# pylint: disable=invalid-name,too-many-locals,no-self-use
""" Support import export formats."""
from __future__ import absolute_import as _abs
from .... import symbol
from .... import ndarray as nd
from .import_helper import _convert_map

def _convert_operator(op_name, attrs, inputs, convert_map=None):
    """Convert from onnx operator to mxnet operator.
    The converter must specify conversions explicitly for incompatible name, and
    apply handlers to operator attributes.

    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, FullyConnected
    attrs : dict
        Dict of operator attributes
    inputs: list
        list of inputs to the operator
    convert_map : dict
        Dict of name : callable, where name is the op's name that
        require conversion to mxnet, callable are functions which
        take attrs and return (new_op_name, new_attrs, inputs)

    Returns
    -------
    (op_name, attrs)
        Converted (op_name, attrs) for mxnet.
    """
    convert_map = convert_map if convert_map else _convert_map
    if op_name in convert_map:
        op_name, new_attrs, inputs = convert_map[op_name](op_name, attrs, inputs)
    else:
        raise NotImplementedError("Operator {} not implemented.".format(op_name))
    op = getattr(symbol, op_name, None)
    if not op:
        raise RuntimeError("Unable to map op_name {} to sym".format(op_name))
    return op, new_attrs, inputs

class GraphProto(object): # pylint: disable=too-few-public-methods
    """A helper class for handling mxnet symbol copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._renames = {}
        self._num_input = 0
        self._num_param = 0

    def from_onnx(self, graph):
        """Construct symbol from onnx graph.
        The inputs from onnx graph is vague, only providing "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        Returns
        -------
        sym :symbol.Symbol
            The returned mxnet symbol
        params : dict
            A dict of name: nd.array pairs, used as pretrained weights
        """
        # parse network inputs, aka parameters
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            self._params[init_tensor.name] = self._parse_array(init_tensor)

        # converting GraphProto message
        for i in graph.input:
            if i.name in self._params:
                # i is a param instead of input
                name_param = 'param_{}'.format(self._num_param)
                self._num_param += 1
                self._params[name_param] = self._params.pop(i.name)
                self._nodes[name_param] = symbol.Variable(name=name_param,
                                                          shape=self._params[name_param].shape)
                self._renames[i.name] = name_param
            else:
                name_input = 'input_{}'.format(self._num_input)
                self._num_input += 1
                self._nodes[name_input] = symbol.Variable(name=name_input)
                self._renames[i.name] = name_input

        # constructing nodes, nodes are stored as directed acyclic graph
        # converting NodeProto message
        for node in graph.node:
            op_name = node.op_type
            node_name = node.name.strip()
            node_name = node_name if node_name else None
            onnx_attr = self._parse_attr(node.attribute)
            inputs = [self._nodes[self._renames.get(i, i)] for i in node.input]
            new_op, mx_attr, inputs = _convert_operator(op_name, onnx_attr, inputs)

            op = new_op(name=node_name, *inputs, **mx_attr)

            assert len(node.output) == len(op.list_outputs()), (
                "Output dimension mismatch between the onnx operator and the mxnet symbol " +
                "{} vs {} for the operator - {}.".format(
                    len(node.output), len(op.list_outputs()), op_name))
            for k, i in zip(list(node.output), range(len(node.output))):
                self._nodes[k] = op[i]
        # now return the outputs
        out = [self._nodes[i.name] for i in graph.output]
        if len(out) > 1:
            out = symbol.Group(out)
        else:
            out = out[0]
        return out, self._params

    def _parse_array(self, tensor_proto):
        """Grab data in TensorProto and convert to numpy array."""
        try:
            from onnx.numpy_helper import to_array
        except ImportError as e:
            raise ImportError("Unable to import onnx which is required {}".format(e))
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        return nd.array(np_array)

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ['f', 'i', 's']:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ['floats', 'ints', 'strings']:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ['t', 'g']:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ['tensors', 'graphs']:
                if list(getattr(a, f)):
                    raise NotImplementedError("Filed {} is not supported in mxnet.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs
