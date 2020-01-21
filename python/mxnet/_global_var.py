_ndarray_cls = None
_np_ndarray_cls = None

def _set_ndarray_class(cls):
    global _ndarray_cls
    _ndarray_cls = cls


def _set_np_ndarray_class(cls):
    global _np_ndarray_cls
    _np_ndarray_cls = cls
