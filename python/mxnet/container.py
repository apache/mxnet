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
# pylint: disable=undefined-variable
"""
Container data structures.
Acknowledgement: This file originates from incubator-tvm
"""
from ._ffi.object import Object, register_object, getitem_helper, PyNativeObject
from ._ffi.function import _init_api

@register_object("MXNet.ADT")
class ADT(Object):
    """Algebatic data type(ADT) object.

    Parameters
    ----------
    tag : int
        The tag of ADT.

    fields : list[Object] or tuple[Object]
        The source tuple.
    """
    def __init__(self, tag, fields):
        for f in fields:
            assert isinstance(f, (Object)), "Expect object" \
            ", but received : {0}".format(type(f))
        self.__init_handle_by_constructor__(_ADT, tag, *fields)

    @property
    def tag(self):
        return _GetADTTag(self)

    def __getitem__(self, idx):
        return getitem_helper(
            self, _GetADTFields, len(self), idx)

    def __len__(self):
        return _GetADTSize(self)

@register_object("MXNet.Map")
class Map(Object):
    """Map container of MXNet.

    You do not need to create Map explicitly.
    Normally python dict will be converted automaticall to Map during mxnet function call.
    You can use convert to create a dict[Object-> Object] into a Map
    """

    def __getitem__(self, k):
        return _MapGetItem(self, k)

    def __contains__(self, k):
        return _MapCount(self, k) != 0

    def items(self):
        """Get the items from the map"""
        akvs = _MapItems(self)
        return [(akvs[i], akvs[i+1]) for i in range(0, len(akvs), 2)]

    def __len__(self):
        return _MapSize(self)

    def get(self, key, default=None):
        """Get an element with a default value.

        Parameters
        ----------
        key : object
            The attribute key.

        default : object
            The default object.

        Returns
        -------
        value: object
            The result value.
        """
        return self[key] if key in self else default

@register_object("MXNet.String")
class String(str, PyNativeObject):
    """String object, represented as a python str.

    Parameters
    ----------
    content : str
        The content string used to construct the object.
    """

    __slots__ = ["__mxnet_object__"]

    def __new__(cls, content):
        """Construct from string content."""
        val = str.__new__(cls, content)
        val.__init_mxnet_object_by_constructor__(_String, content)
        return val

    # pylint: disable=no-self-argument
    def __from_mxnet_object__(cls, obj):
        """Construct from a given mxnet object."""
        content = _GetFFIString(obj)
        val = str.__new__(cls, content)
        val.__mxnet_object__ = obj
        return val

_init_api("mxnet.container")
