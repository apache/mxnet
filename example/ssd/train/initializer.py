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
            
class CustomInitializer(mx.init.Xavier):
    """
    Customized initializer for ssd models
    """
    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        super(CustomInitializer, self).__init__(rnd_type, factor_type, magnitude)

    def _init_default(self, name, arr):
        if name.endswith('init'):
            self._init_zero(name, arr)
        elif name.endswith('scale'):
            try:
                s = float(name[:-6].split('_')[-1])  # read scale from name
                arr[:] = s
            except:
                self._init_one(name, arr)
        else:
            raise ValueError('Unknown initialization pattern for %s' % name)
