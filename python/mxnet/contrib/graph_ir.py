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
"""NNVM GRAPH IR"""
from __future__ import absolute_import
from __future__ import division

import ctypes
import json
from ..base import _LIB, check_call, string_types
from ..base import mx_uint, SymbolHandle, c_array, c_str, py_str
from ..symbol import Variable, Symbol, Group as _Group


GraphHandle = ctypes.c_void_p


class GraphIndex(object):
    """Index for quickly accessing graph attributes.
    Parameters
    ----------
    graph : Graph
        The graph to create index.
    """

    def __init__(self, graph):
        jgraph = json.loads(create(graph).apply("SaveJSON").json_attr("json"))
        self.nodes = jgraph["nodes"]
        self.entry_ptr = jgraph["node_row_ptr"]
        self._name2nodeid = {n["name"]: i for i, n in enumerate(self.nodes)}
        self.input_names = graph.symbol.list_input_names()
        self.output_entries = jgraph["heads"]

    @property
    def num_nodes(self):
        """Number of nodes in graph."""
        return len(self.entry_ptr) - 1

    @property
    def num_node_entries(self):
        """Number of nodes in graph."""
        return self.entry_ptr[-1]

    def node_id(self, key):
        """Get the node index for a given key.
        Parameters
        ----------
        key : str or int
            The node key or index
        Returns
        -------
        index : int
            The entry index
        """
        return self._name2nodeid[key]

    def entry_id(self, key, value_index=0):
        """Get the entry id of a node entry.
        Parameters
        ----------
        key : str or int
            The node key or index
        value_index : int
            The value index of output
        Returns
        -------
        index : int
            The entry index
        """
        if isinstance(key, (list, tuple)):
            if len(key) != 3:
                raise ValueError("Expect entry index to be tuple of 3 elems")
            key, value_index, _ = key
        idx = self.node_id(key) if isinstance(key, str) else key
        assert value_index < self.entry_ptr[idx + 1]
        return self.entry_ptr[idx] + value_index


class Graph(object):
    """Graph is the graph object that can be used to apply optimization pass.
    It contains additional graphwise attribute besides the internal symbol.
    """
    _tvm_tcode = 17

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle
        Parameters
        ----------
        handle : GraphHandle
            the handle to the underlying C++ Graph
        """
        self.handle = handle
        self._index = None

    def __del__(self):
        check_call(_LIB.MXNNGraphFree(self.handle))

    def json_attr(self, key):
        """Get attribute string from the graph.
        Parameters
        ----------
        key : str
            The key to get attribute from.
        Returns
        -------
        value : str
            The attribute value of the key, returns None if attribute do not exist.
        """
        ret = ctypes.c_char_p()
        success = ctypes.c_int()
        check_call(_LIB.MXNNGraphGetJSONAttr(
            self.handle, c_str(key), ctypes.byref(ret), ctypes.byref(success)))
        if success.value != 0:
            json_str = py_str(ret.value)
            return json.loads(json_str)[1]
        return None

    def _set_symbol_list_attr(self, key, value):
        """Set the attribute of the graph.
        Parameters
        ----------
        key : string
            The key of the attribute
        value : value
            The any type that can be dumped to json
        type_name : string
            The typename registered on c++ side.
        """
        if isinstance(value, list):
            value = _Group(value)
        if not isinstance(value, Symbol):
            raise ValueError("value need to be grouped symbol")
        check_call(_LIB.NNGraphSetNodeEntryListAttr_(
            self.handle, c_str(key), value.handle))

    def _set_json_attr(self, key, value, type_name=None):
        """Set the attribute of the graph.
        Parameters
        ----------
        key : string
            The key of the attribute
        value : value
            The any type that can be dumped to json
        type_name : string
            The typename registered on c++ side.
        """
        if isinstance(value, string_types):
            type_name = 'str'
        elif type_name is None:
            raise ValueError("Need to specify type_name")
        json_value = json.dumps([type_name, value])
        check_call(_LIB.MXNNGraphSetJSONAttr(
            self.handle, c_str(key), c_str(json_value)))

    @property
    def _tvm_handle(self):
        return self.handle.value

    @property
    def symbol(self):
        shandle = SymbolHandle()
        check_call(_LIB.MXNNGraphGetSymbol(self.handle, ctypes.byref(shandle)))
        return Symbol(shandle)

    def json(self):
        """Get JSON representation of the graph
        Returns
        -------
        json : str
            JSON representation of the graph
        """
        return self.apply("SaveJSON").json_attr("json")

    def _tvm_graph_json(self):
        """Get TVM graph json"""
        return self.json()

    @property
    def index(self):
        if not self._index:
            self._index = GraphIndex(self)
        return self._index

    def ir(self, join_entry_attrs=None, join_node_attrs=None):
        """Get text form of graph ir.
        Parameters
        ----------
        join_entry_attrs : list of str
            List of graph NodeEntry attribute to be
            printed along each operator.
        join_node_attrs : list of str
            List of graph node attribute to be
            printed along each operator.
        """
        if join_entry_attrs:
            self._set_json_attr("join_entry_attrs",
                                join_entry_attrs, "list_str")
        if join_node_attrs:
            self._set_json_attr("join_node_attrs", join_node_attrs, "list_str")
        return self.apply("PrintGraphIR").json_attr("graphir")

    def apply(self, passes):
        """Apply passes to the graph
        Parameters
        ----------
        passes : str or list of str
            The passes to be applied
        Returns
        -------
        g : Graph
            The transformed graph.
        """
        if isinstance(passes, string_types):
            passes = [passes]
        cpass = c_array(ctypes.c_char_p, [c_str(key) for key in passes])
        ghandle = GraphHandle()
        npass = mx_uint(len(passes))
        check_call(_LIB.MXNNGraphApplyPasses(
            self.handle, npass, cpass, ctypes.byref(ghandle)))
        return Graph(ghandle)


def load_json(json_str):
    """Create a new graph by loading from json
    Parameters
    ----------
    json_str : str
        The json string
    Returns
    -------
    graph : Graph
        The loaded graph
    """
    ret = create(Variable("x"))
    ret._set_json_attr("json", json_str)
    return ret.apply("LoadJSON")


def create(symbol):
    """Create a new graph from symbol.
    Parameters
    ----------
    symbol : Symbol
        The symbolic graph used to create Graph object.
    Returns
    -------
    graph : Graph
        A generated new graph object.
    """
    ghandle = GraphHandle()
    check_call(_LIB.MXNNGraphCreate(
        symbol.handle, ctypes.byref(ghandle)))
    return Graph(ghandle)
