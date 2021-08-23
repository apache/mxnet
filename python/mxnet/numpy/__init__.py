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

"""MXNet NumPy module."""


from . import random
from . import linalg
from .multiarray import *  # pylint: disable=wildcard-import
from . import _op
from . import _register
from ._op import *  # pylint: disable=wildcard-import
from .utils import *  # pylint: disable=wildcard-import
from .function_base import *  # pylint: disable=wildcard-import
from .stride_tricks import *  # pylint: disable=wildcard-import
from .io import *  # pylint: disable=wildcard-import
from .arrayprint import *  # pylint: disable=wildcard-import

__all__ = []
