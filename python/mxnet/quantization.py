from __future__ import absolute_import

import ctypes
from .base import _LIB, string_types, numeric_types, check_call
from .base import c_array, py_str, c_str, mx_real_t, mx_uint
from .base import NDArrayHandle, ExecutorHandle, SymbolHandle
from .symbol import Symbol
from . import ndarray as nd
from .contrib import ndarray as cnd

def quantize(param):
    max_range = nd.max(param)
    min_range = nd.min(param)
    return cnd.quantize(param, min_range, max_range)

def quantize_params(qsym, params):
    inputs_name = qsym.list_arguments()
    quantized_params = {}
    for name in inputs_name:
        if name.endswith(('weight_quantize', 'bias_quantize')):
            origin_name = name.replace('_quantize', '')
            val, vmin, vmax = quantize(params[origin_name])
            quantized_params[name] = val
            quantized_params[name+'_min'] = vmin
            quantized_params[name+'_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
    return quantized_params

def quantize_graph(sym, ignore_symbols=None, offline_params=None):
    num_ignore = 0
    ignore_handles = []
    if ignore_symbols is not None:
        assert isinstance(ignore_symbols, list)
        num_ignore = len(ignore_symbols)
        for s in ignore_symbols:
            ignore_handles.append(s.handle)

    num_offline = 0
    offline = []
    if offline_params is not None:
        num_offline = len(offline_params)
        for k in offline_params:
            offline.append(c_str(k))

    out = SymbolHandle()
    check_call(_LIB.MXQuantizeGraph(sym.handle,
                                    ctypes.byref(out),
                                    mx_uint(num_ignore),
                                    c_array(SymbolHandle, ignore_handles),
                                    mx_uint(num_offline),
                                    c_array(ctypes.c_char_p, offline)))
    return Symbol(out)

