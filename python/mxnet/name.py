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
"""Automatic naming support for symbolic API."""
import contextvars


class NameManager:
    """NameManager to do automatic naming.

    Developers can also inherit from this class to change naming behavior.
    """
    def __init__(self):
        self._counter = {}
        self._old_manager = None

    def get(self, name, hint):
        """Get the canonical name for a symbol.

        This is the default implementation.
        If the user specifies a name,
        the user-specified name will be used.

        When user does not specify a name, we automatically generate a
        name based on the hint string.

        Parameters
        ----------
        name : str or None
            The name specified by the user.

        hint : str
            A hint string, which can be used to generate name.

        Returns
        -------
        full_name : str
            A canonical name for the symbol.
        """
        if name:
            return name
        if hint not in self._counter:
            self._counter[hint] = 0
        name = f'{hint}{self._counter[hint]}'
        self._counter[hint] += 1
        return name

    def __enter__(self):
        # Token can't be pickled and Token.old_value is Token.MISSING if _current.get() uses default value
        self._old_manager = _current.get()
        _current.set(self)
        return self

    def __exit__(self, ptype, value, trace):
        _current.set(self._old_manager)


class Prefix(NameManager):
    """A name manager that attaches a prefix to all names.

    Examples
    --------
    >>> import mxnet as mx
    >>> data = mx.symbol.Variable('data')
    >>> with mx.name.Prefix('mynet_'):
            net = mx.symbol.FullyConnected(data, num_hidden=10, name='fc1')
    >>> net.list_arguments()
    ['data', 'mynet_fc1_weight', 'mynet_fc1_bias']
    """
    def __init__(self, prefix):
        super().__init__()
        self._prefix = prefix

    def get(self, name, hint):
        name = super().get(name, hint)
        return self._prefix + name


_current = contextvars.ContextVar('namemanager', default=NameManager())


def current():
    """Returns the current name manager."""
    return _current.get()
