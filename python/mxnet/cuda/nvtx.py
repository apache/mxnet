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
"""Utilities for NVTX usage in MXNet"""

from ..base import _LIB, mx_uint, c_str, check_call

# Palette of colors
RED = 0xFF0000
GREEN = 0x00FF00
BLUE = 0x0000FF
YELLOW = 0xB58900
ORANGE = 0xCB4B16
RED1 = 0xDC322F
MAGENTA = 0xD33682
VIOLET = 0x6C71C4
BLUE1 = 0x268BD2
CYAN = 0x2AA198
GREEN1 = 0x859900

def range_push(name, color=ORANGE):
    """Starts a new named NVTX range."""
    check_call(_LIB.MXNVTXRangePush(
        c_str(name),
        mx_uint(color)))

def range_pop():
    """Ends a NVTX range."""
    check_call(_LIB.MXNVTXRangePop())

class range:
    def __init__(self, name, color=ORANGE):
        self.name = name
        self.color = color

    def __enter__(self):
        range_push(self.name, self.color)

    def __exit__(self, exc_type, exc_val, exc_tb):
        range_pop()
