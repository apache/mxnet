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
"""Structured error classes in MXNet.

Each error class takes an error message as its input.
See the example sections for for suggested message conventions.
To make the code more readable, we recommended developers to
copy the examples and raise errors with the same message convention.
"""
from .base import MXNetError, register_error

__all__ = ['MXNetError', 'register']

register = register_error

@register_error
class InternalError(MXNetError):
    """Internal error in the system.

    Examples
    --------
    .. code :: c++

        // Example code C++
        LOG(FATAL) << "InternalError: internal error detail.";

    .. code :: python

        # Example code in python
        raise InternalError("internal error detail")
    """
    def __init__(self, msg):
        # Patch up additional hint message.
        if "MXNet hint:" not in msg:
            msg += ("\nMXNet hint: You hit an internal error. Please open an issue in "
                    "https://github.com/apache/mxnet/issues/new/choose"
                    " to report it.")
        super(InternalError, self).__init__(msg)


register_error("ValueError", ValueError)
register_error("TypeError", TypeError)
register_error("AttributeError", AttributeError)
register_error("IndexError", IndexError)
register_error("NotImplementedError", NotImplementedError)
register_error("InternalError", InternalError)
register_error("IOError", IOError)
register_error("FloatingPointError", FloatingPointError)
register_error("RuntimeError", RuntimeError)
