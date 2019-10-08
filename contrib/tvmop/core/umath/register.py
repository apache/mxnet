from .core import *
from .operator import *
from ...opdef import defop
from ...utils import AllTypes

two2one_cpu_attrs = {
    'target': 'cpu',
    'dtype': AllTypes,
    'ndim': [5],
    'req': ['kWriteTo', 'kAddTo'],
    'attrs': ['req'],
}


@defop(name="equal_cpu", target="cpu", auto_broadcast=True, ndim=[5],
       dtype=AllTypes+['bool'],
       req=['kWriteTo', 'kAddTo'], attrs=['req'])
def equal_cpu(dtype, ndim):
    return two2one_cpu(equal, dtype, dtype, ndim)


@defop(name="equal_cpu", target="cpu", auto_broadcast=True, ndim=[5],
       dtype=["float32", "float64", "uint8", "int8", "int32", "int64", "bool"],
       req=['kWriteTo', 'kAddTo'], attrs=['req'])
def equal_gpu(dtype, ndim):
    return two2one_gpu(equal, dtype, dtype, ndim)
