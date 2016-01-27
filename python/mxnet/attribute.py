# coding: utf-8
"""Attribute scoping support for symbolic API."""
from __future__ import absolute_import

from .base import string_types

class AttrScope(object):
    """Attribute manager for scoping.

    User can also inherit this object to change naming behavior.

    Parameters
    ----------
    kwargs
        The attributes to set for all symbol creations in the scope.
    """
    current = None

    def __init__(self, **kwargs):
        self._old_scope = None
        for value in kwargs.values():
            if not isinstance(value, string_types):
                raise ValueError("Attributes need to be string")
        self._attr = kwargs

    def get(self, attr):
        """
        Get the attribute dict given the attribute set by the symbol.

        Parameters
        ----------
        attr : dict of string to string
            The attribute passed in by user during symbol creation.

        Returns
        -------
        attr : dict of string to string
            Updated attributes to add other scope related attributes.
        """
        if self._attr:
            ret = self._attr.copy()
            if attr:
                ret.update(attr)
            return ret
        else:
            return attr

    def __enter__(self):
        # pylint: disable=protected-access
        self._old_scope = AttrScope.current
        attr = AttrScope.current._attr.copy()
        attr.update(self._attr)
        self._attr = attr
        AttrScope.current = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope
        AttrScope.current = self._old_scope

AttrScope.current = AttrScope()

