"""Registry wrapper for all detection blocks.
"""
from mxnet import registry
from mxnet import gluon

register = registry.get_register_func(gluon.Block, 'object_detection')
alias = registry.get_alias_func(gluon.Block, 'object_detection')
create = registry.get_create_func(gluon.Block, 'object_detection')
