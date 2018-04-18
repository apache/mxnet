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
from ....base import string_types
from .import_helper import _convert_map as convert_map

class GraphProto(object): # pylint: disable=too-few-public-methods
    """A helper class for handling mxnet symbol copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._num_input = 0
        self._num_param = 0

    def _convert_operator(self, node_name, op_name, attrs, inputs):
        """Convert from onnx operator to mxnet operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        :param node_name : str
            name of the node to be translated.
        :param op_name : str
            Operator name, such as Convolution, FullyConnected
        :param attrs : dict
            Dict of operator attributes
        :param inputs: list
            list of inputs to the operator
        Returns
        -------
        :return mxnet_sym
            Converted mxnet symbol
        """
        if op_name in convert_map:
            op_name, new_attrs, inputs = convert_map[op_name](attrs, inputs, self)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        if isinstance(op_name, string_types):
            new_op = getattr(symbol, op_name, None)
            if not new_op:
                raise RuntimeError("Unable to map op_name {} to sym".format(op_name))
            if node_name is None:
                mxnet_sym = new_op(*inputs, **new_attrs)
            else:
                mxnet_sym = new_op(name=node_name, *inputs, **new_attrs)
            return mxnet_sym
        return op_name

    def from_onnx(self, graph):
        """Construct symbol from onnx graph.

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
                self._nodes[i.name] = symbol.Variable(name=i.name,
                                                      shape=self._params[i.name].shape)
            else:
                self._nodes[i.name] = symbol.Variable(name=i.name)

        # For storing arg  and aux params for the graph.
        auxDict = {}
        argDict = {}

        # constructing nodes, nodes are stored as directed acyclic graph
        # converting NodeProto message
        for node in graph.node:
            op_name = node.op_type
            node_name = node.name.strip()
            node_name = node_name if node_name else None
            onnx_attr = self._parse_attr(node.attribute)
            inputs = [self._nodes[i] for i in node.input]
            mxnet_sym = self._convert_operator(node_name, op_name, onnx_attr, inputs)

            for k, i in zip(list(node.output), range(len(mxnet_sym.list_outputs()))):
                self._nodes[k] = mxnet_sym[i]

            # splitting params into args and aux params
            for args in mxnet_sym.list_arguments():
                if args in self._params:
                    argDict.update({args: nd.array(self._params[args])})
            for aux in mxnet_sym.list_auxiliary_states():
                if aux in self._params:
                    auxDict.update({aux: nd.array(self._params[aux])})

        # now return the outputs
        out = [self._nodes[i.name] for i in graph.output]
        if len(out) > 1:
            out = symbol.Group(out)
        else:
            out = out[0]
        return out, argDict, auxDict

    def _parse_array(self, tensor_proto):
        """Grab data in TensorProto and convert to numpy array."""
        try:
            from onnx.numpy_helper import to_array
        except ImportError:
            raise ImportError("Onnx and protobuf need to be installed. "
                              + "Instructions to install - https://github.com/onnx/onnx")
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        return nd.array(np_array)

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ['f', 'i', 's']:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
                    # Needed for supporting python version  > 3.5
                    if isinstance(attrs[a.name], bytes):
                        attrs[a.name] = attrs[a.name].decode(encoding='utf-8')
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
