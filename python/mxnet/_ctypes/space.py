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
"""ConfigSpace ctypes API."""
import ctypes

from ..base import _LIB
from ..base import c_str_array, c_array
from ..base import check_call

class COtherOptionEntity(ctypes.Structure):
    """ctypes data structure for OtherOptionEntity"""
    _fields_ = [("val", ctypes.c_int)]


class COtherOptionSpace(ctypes.Structure):
    """ctypes data structure for OtherOptionSpace"""
    _fields_ = [("entities", ctypes.POINTER(COtherOptionEntity)),
                ("entities_size", ctypes.c_int)]


class CConfigSpace(ctypes.Structure):
    """ctypes data structure for ConfigSpace"""
    _fields_ = [("entity_map_size", ctypes.c_int),
                ("entity_map_key", ctypes.POINTER(ctypes.c_char_p)),
                ("entity_map_val", ctypes.POINTER(COtherOptionEntity)),
                ("space_map_size", ctypes.c_int),
                ("space_map_key", ctypes.POINTER(ctypes.c_char_p)),
                ("space_map_val", ctypes.POINTER(COtherOptionSpace))]


class CConfigSpaces(ctypes.Structure):
    """ctypes data structure for ConfigSpaces"""
    _fields_ = [("spaces_size", ctypes.c_int),
                ("spaces_key", ctypes.POINTER(ctypes.c_char_p)),
                ("spaces_val", ctypes.POINTER(CConfigSpace))]


def c_other_option_entity(x):
    """constructor for OtherOptionEntity"""
    ret = COtherOptionEntity()
    ret.val = x.val
    return ret


def c_other_option_space(x):
    """constructor for OtherOptionSpace"""
    ret = COtherOptionSpace()
    ret.entities = c_array(COtherOptionEntity,
                           [c_other_option_entity(e) for e in x.entities])
    ret.entities_size = len(x.entities)
    return ret


def c_config_space(x):
    """constructor for ConfigSpace"""
    ret = CConfigSpace()
    ret.entity_map_key = c_str_array(x._entity_map.keys())
    ret.entity_map_val = c_array(COtherOptionEntity,
                                 [c_other_option_entity(e) for e in x._entity_map.values()])
    ret.entity_map_size = len(x._entity_map)
    ret.space_map_key = c_str_array(x.space_map.keys())
    ret.space_map_val = c_array(COtherOptionSpace,
                                [c_other_option_space(v) for v in x.space_map.values()])
    ret.space_map_size = len(x.space_map)
    return ret


def c_config_spaces(x):
    """constructor for ConfigSpaces"""
    ret = CConfigSpaces()
    ret.spaces_size = len(x.spaces)
    ret.spaces_key = c_str_array(x.spaces.keys())
    ret.spaces_val = c_array(CConfigSpace, [c_config_space(c) for c in x.spaces.values()])
    return ret


def _set_tvm_op_config(x):
    """ctypes implementation of populating the config singleton"""
    check_call(_LIB.MXLoadTVMConfig(c_config_spaces(x)))
    return x
