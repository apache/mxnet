# coding: utf-8
"""Automatic naming support for symbolic API."""
from __future__ import absolute_import

class NameManager(object):
    """NameManager to do automatic naming.

    Developers can also inherit from this class to change naming behavior.
    """
    current = None

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
        name = '%s%d' % (hint, self._counter[hint])
        self._counter[hint] += 1
        return name

    def __enter__(self):
        self._old_manager = NameManager.current
        NameManager.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_manager
        NameManager.current = self._old_manager


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
        super(Prefix, self).__init__()
        self._prefix = prefix

    def get(self, name, hint):
        name = super(Prefix, self).get(name, hint)
        return self._prefix + name

# initialize the default name manager
NameManager.current = NameManager()
