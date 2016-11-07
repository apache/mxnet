import mxnet as mx


class ScaleInitializer(mx.init.Initializer):
    """
    Customized initializer for scale layer
    """
    def __init__(self):
        pass

    def _init_default(self, name, arr):
        if name.endswith("scale"):
            self._init_one(name, arr)
        else:
            raise ValueError('Unknown initialization pattern for %s' % name)
