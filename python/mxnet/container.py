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
from ._ffi.object import Object, register_object, getitem_helper
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

_init_api("mxnet.container")
